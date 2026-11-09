import sys
import time
import importlib.util
import traceback
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# 健壮加载带连字符的 agent 入口（普通 `import agent.praisonai-Agent` 会被连字符
# 解析失败）。直接用 importlib 按路径加载，等价于 eval/runner.load_agent_entry。
# ---------------------------------------------------------------------------
def _load_build_agent():
    entry = Path(__file__).resolve().parent / "agent" / "praisonai-agent.py"
    agent_dir = str(entry.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("agentic_rag_entry", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_agent


@st.cache_resource
def get_agent():
    """构造 Agent 并缓存（全局仅一次），构造异常会被上层 try 捕获。"""
    return _load_build_agent()()


st.title("智能问答机器人")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入您的问题…")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    try:
        agent = get_agent()
        with st.spinner("智能客服思考中…"):
            # praisonaiagents 此版本无 execute_stream；iter_stream 才是
            # "App-friendly streaming"接口：产出 chunk 且不向终端打印。
            res_stream = agent.iter_stream(prompt)

            chunks = []

            def capture(generator, cache):
                for chunk in generator:
                    cache.append(chunk)
                    for ch in chunk:          # 打字机效果
                        time.sleep(0.02)
                        yield ch

            st.chat_message("assistant").write_stream(capture(res_stream, chunks))

        # 关键修复：拼接所有流式子块，得到完整回复（原代码只存了最后一个子块）。
        answer = "".join(chunks).strip()
        if not answer:
            answer = "（模型未返回内容，请稍后重试）"
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    except Exception as exc:
        # 友好提示 + 折叠式完整堆栈，方便定位是 API 限流 / 网络 / 代码问题。
        st.error(f"⚠️ 回答生成失败：{type(exc).__name__}: {exc}")
        st.exception(exc)
