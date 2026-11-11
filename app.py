import sys
import time
from pathlib import Path

import streamlit as st

# agent/ 无 __init__.py，无法稳定走 `from agent.agent_factory` 的命名空间包导入
# （streamlit 运行脚本时 sys.path 行为与普通 python 不同，易触发 ModuleNotFoundError）。
# 改为把 agent 目录注入 sys.path 后直接 import 模块；模块内部会自行把 agent/ 与项目根
# 加入 path，自动解析 model / utils / tools 等绝对导入。
_AGENT_DIR = str(Path(__file__).resolve().parent / "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from agent_factory import productor  # noqa: E402


@st.cache_resource
def get_agent():
    """构造 Agent 并缓存（全局仅一次）。"""
    return productor

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
