# -*- coding: utf-8 -*-
"""评测链路追踪：极轻量、零外部依赖，业务代码可安全 import。

设计约束（写在这里而不是写文档）：
- 用模块级单例 sink 而非 contextvar：PraisonAI 内部可能切线程，contextvar
  不会自动向线程池传播，工具调用记录会丢；模块级引用在同一进程内一定拿得到。
- 代价是不能并发。评测 runner 严格串行；生产侧若并发执行，不要开启追踪。
- 业务代码只在 sink 存在时记一笔，sink 为空时开销只有一次 if 判断。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str = ""
    source: str = ""
    page: Any = ""
    preview: str = ""


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    retrieved: list = field(default_factory=list)
    prompt: str = ""
    result_chars: int = 0
    error: str = ""


@dataclass
class RunTrace:
    question: str = ""
    answer: str = ""
    tool_calls: list = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    _t0: float = field(default_factory=perf_counter, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("_t0", None)
        return data


@dataclass
class RunResult:
    """run() 的统一返回类型。评测侧只依赖本模块，不依赖带连字符的入口文件名。"""
    answer: str
    trace: RunTrace

    def to_dict(self) -> dict:
        return {"answer": self.answer, "trace": self.trace.to_dict()}


_ACTIVE: RunTrace | None = None


def begin(question: str) -> RunTrace:
    global _ACTIVE
    _ACTIVE = RunTrace(question=question, _t0=perf_counter())
    return _ACTIVE


def current() -> RunTrace | None:
    return _ACTIVE


def end(trace: RunTrace) -> RunTrace:
    global _ACTIVE
    trace.duration_ms = (perf_counter() - trace._t0) * 1000
    if _ACTIVE is trace:
        _ACTIVE = None
    return trace


def record_tool(
    name: str,
    args: dict | None = None,
    duration_ms: float = 0.0,
    retrieved: list | None = None,
    prompt: str = "",
    result_chars: int = 0,
    error: str = "",
) -> None:
    trace = current()
    if trace is None:
        return
    trace.tool_calls.append(
        ToolCall(
            name=name,
            args=args or {},
            duration_ms=round(duration_ms, 2),
            retrieved=retrieved or [],
            prompt=prompt,
            result_chars=result_chars,
            error=error,
        )
    )
