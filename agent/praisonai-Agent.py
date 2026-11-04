import warnings
# 必须放在所有 import 之前：LeafSpec 的 FutureWarning 在 import praisonaiagents
# （→ litellm/pytree）阶段就触发，过滤器装晚了就拦不住。
# module="typing_extensions" 不可靠（stacklevel 把警告归因到调用方模块），
# 改按消息内容匹配。
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

from praisonaiagents import Agent, Task, AgentTeam
from praisonaiagents.escalation.loop_guard import LoopGuard, LoopGuardConfig
from praisonaiagents.tools import tavily_search as web_search

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.factory import praison_llm as llm
from utils.prompt_loader import load_product_expert_prompts
from agent.tools.agent_tools import rag_summarize



# product_expert = 

productor = Agent(
     name="产品专家"
    ,instructions=load_product_expert_prompts()
    ,tools=[rag_summarize, web_search]
    ,llm=llm
)

_LOOP_GUARD = LoopGuard(LoopGuardConfig(enabled=True, max_time_per_turn=600))
productor._loop_guard = _LOOP_GUARD

msg = "杭州天气怎么样？"
productor.start(msg)


