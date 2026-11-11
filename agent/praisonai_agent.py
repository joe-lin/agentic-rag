import warnings
# 必须放在所有 import 之前：LeafSpec 的 FutureWarning 在 import praisonaiagents
# （→ litellm/pytree）阶段就触发，过滤器装晚了就拦不住。
# module="typing_extensions" 不可靠（stacklevel 把警告归因到调用方模块），
# 改按消息内容匹配。
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

from praisonaiagents import Agent
from praisonaiagents.escalation.loop_guard import LoopGuard, LoopGuardConfig
from praisonaiagents.tools import tavily_search as web_search
from types import GeneratorType


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.factory import praison_llm as llm
from utils.prompt_loader import load_product_expert_prompt
from utils.trace import RunResult, begin, end
from tools.agent_tools import rag_summarize
from tools.wechat_publisher import publish_wechat_oa

# 说明：tavily_search 原先 import 了却没挂进 tools，属于"看起来能联网、实际不能"的
# 假能力，会诱发 Agent 编造库外信息。要用时显式挂进 build_agent 的 tools 列表。


def build_agent() -> Agent:
    """构造被测 Agent。评测与生产共用同一份构造逻辑，避免评测跑的是另一个 Agent。"""
    agent = Agent(
        name="产品专家"
        ,instructions=load_product_expert_prompt()
        ,tools=[rag_summarize, web_search, publish_wechat_oa]
        ,llm=llm
    )
    agent._loop_guard = LoopGuard(LoopGuardConfig(enabled=True, max_time_per_turn=1000))
    return agent


productor = build_agent()


def _collect_answer(raw) -> str:
    """把 PraisonAI 可能的 str / Generator / Outcome 统一收敛成纯文本。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, GeneratorType) or (
        hasattr(raw, "__iter__") and not isinstance(raw, (dict, list, tuple))
    ):
        return "".join(str(chunk) for chunk in raw)
    return str(getattr(raw, "raw", None) or getattr(raw, "content", None) or raw)


def run(question: str) -> RunResult:
    """评测唯一入口：一问一 trace，同步返回答案与完整调用链。

    本文件名带连字符、不可直接 import，评测侧通过 eval/runner.py 的
    load_agent_entry() 按路径加载后调用本函数。

    用 productor.run() 而不是 .start()：库源码 execution_mixin.py 明确写了
    .start() 在 TTY 下默认流式、返回 Generator 且带 verbose 富文本；
    .run() 恒为 silent 且返回 str，才是脚本/评测该用的入口。
    """
    trace = begin(question)
    try:
        if hasattr(productor, "run"):
            raw = productor.run(question)
        else:
            raw = productor.start(question, stream=False)
        trace.answer = _collect_answer(raw)
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end(trace)
    return RunResult(answer=trace.answer, trace=trace)


def chat_loop():
    """交互式多轮对话。

    复用同一个 productor 实例：Agent 内部用 self.chat_history 累积每一轮
    的用户/助手消息（见 chat_mixin.chat → _append_to_chat_history），连续提问
    自动带上历史上下文，无需手动拼接消息。

    交互命令：
      exit / quit / q / 退出   结束对话
      clear / 重置 / reset      清空上下文（productor.chat_history = []）
    """
    print("=" * 54)
    print("产品专家 · 多轮对话（exit/quit 退出，clear 清空上下文）")
    print("=" * 54)
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit", "q", "退出"):
            print("再见。")
            break
        if low in ("clear", "重置", "reset"):
            productor.chat_history = []
            print("（上下文已清空）")
            continue
        try:
            # run() 内部 _load_history_context() + chat()，自动累积 chat_history
            answer = productor.run(user_input, stream=False)
            print("\n产品专家:", _collect_answer(answer))
        except Exception as exc:
            print(f"\n[出错] {type(exc).__name__}: {exc}")


if __name__ == '__main__':
   # 单次评测入口（保持兼容，供 eval/runner.py 的 load_agent_entry 调用）
   # result = run("基于 OceanBase 的知识库方案、解决方案和场景描述")
   # print(result.answer)

   # 交互式多轮对话
   chat_loop()
