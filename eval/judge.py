# -*- coding: utf-8 -*-
"""判分器：规则判硬约束 + 可选 LLM judge 判软维度 + 加权聚合 + 硬门控。

两种模式：
- rule（默认，离线）：把 checklist 落成可机器核验的代理指标，0/5/10 三档。
  优点是可复现、零成本、零网络，适合每次改 prompt/参数后的回归。
- llm：把 checklist_by_dim + anchors_by_dim 原样交给 judge 模型打 0/5/10。
  用于和规则分对齐校准；失败自动回退 rule 并在结果里标注。

单独重判（改阈值不用重跑 Agent）：python3 -m eval.judge --input eval/results/xxx.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
RESULTS = EVAL / "results"

STRUCTURE_MARKERS = ("```", '{"', '["', "【参考资料", "参考资料1", "- item", "* ")
QUESTION_MARKERS = ("请先确认", "请问您", "能否先", "请补充", "我需要先问")
ABSTAIN_DEFAULT = ("资料中未", "未提及", "现有资料", "知识库中未", "无法从", "没有相关", "无法回答")


def _norm(text: str) -> str:
    """忽略空格与大小写，兼容 '25 分钟' / '25分钟'、cosine / COSINE。"""
    return re.sub(r"\s+", "", str(text)).lower()


def _must_include(case: dict) -> list:
    return [k for span in case.get("golden_spans", []) for k in span.get("must_include", [])]


def _coverage(keys: list, text: str) -> tuple:
    if not keys:
        return 1.0, [], []
    norm = _norm(text)
    hits = [k for k in keys if _norm(k) in norm]
    return len(hits) / len(keys), hits, [k for k in keys if k not in hits]


def _tier(ratio: float, full: float = 0.999, partial: float = 0.5) -> int:
    if ratio >= full:
        return 10
    if ratio >= partial:
        return 5
    return 0


def _tool_calls(trace: dict) -> list:
    return trace.get("tool_calls", []) or []


def _retrieved(trace: dict) -> list:
    return [c for call in _tool_calls(trace) for c in (call.get("retrieved") or [])]


def _has_chinese(text: str) -> bool:
    if not text:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk / max(1, len(text.strip())) >= 0.25


def _rule_scores(case: dict, answer: str, trace: dict) -> tuple:
    """返回 (各维度 0/5/10, 可解释证据)。只对 checklist_by_dim 声明过的维度打分。"""
    declared = list(case.get("checklist_by_dim", {}))
    scores: dict[str, int] = {}
    evidence: dict = {}

    keys = _must_include(case)
    ratio, hits, missing = _coverage(keys, answer)
    evidence["must_include_hits"] = hits
    evidence["must_include_missing"] = missing
    markers = case.get("abstain_markers", list(ABSTAIN_DEFAULT))
    abstained = any(_norm(m) in _norm(answer) for m in markers)
    evidence["abstained"] = abstained

    calls = _tool_calls(trace)
    retrieved = _retrieved(trace)
    evidence["tool_calls"] = [c.get("name") for c in calls]
    evidence["retrieved_pages"] = sorted({str(c.get("page", "")) for c in retrieved})

    for dim in declared:
        if dim == "groundedness":
            if keys:
                scores[dim] = _tier(ratio)
            else:
                scores[dim] = 10 if abstained else (5 if len(answer.strip()) <= 40 else 0)

        elif dim == "retrieval_quality":
            golden_pages = {str(p) for p in case.get("golden_pages", [])}
            hit_pages = {str(c.get("page", "")) for c in retrieved}
            page_ratio = (
                len(golden_pages & hit_pages) / len(golden_pages) if golden_pages else 0.0
            )
            kw = case.get("retrieval_keywords", [])
            corpus = _norm(" ".join(str(c.get("preview", "")) for c in retrieved))
            kw_ratio = (
                len([k for k in kw if _norm(k) in corpus]) / len(kw) if kw else 0.0
            )
            evidence["retrieval_page_ratio"] = round(page_ratio, 2)
            evidence["retrieval_kw_ratio"] = round(kw_ratio, 2)
            scores[dim] = _tier((page_ratio + kw_ratio) / 2) if (golden_pages or kw) else 10

        elif dim == "tool_decision":
            expected = case.get("expected_tool", "any")
            rag_calls = [c for c in calls if c.get("name") == "rag_summarize"]
            if expected == "required":
                if rag_calls:
                    q = _norm(rag_calls[0].get("args", {}).get("query", ""))
                    scores[dim] = 10 if q == _norm(case.get("user_question", "")) else 5
                else:
                    scores[dim] = 0
            elif expected == "forbidden":
                scores[dim] = 10 if not calls else 0
            else:
                scores[dim] = 10

        elif dim == "abstention":
            is_abstain_case = case.get("case_type") in ("negative", "tool_decision") and "abstain_markers" in case
            if not is_abstain_case:
                scores[dim] = 10 if answer.strip() else 0
            elif abstained:
                scores[dim] = 10
            elif len(answer.strip()) <= 30:
                scores[dim] = 5
            else:
                scores[dim] = 0

        elif dim == "task_completion":
            scores[dim] = _tier(ratio) if keys else (10 if 5 <= len(answer.strip()) <= 800 else 5)

        elif dim == "instruction_follow":
            violations = []
            if not _has_chinese(answer):
                violations.append("非中文")
            if any(m in answer for m in STRUCTURE_MARKERS):
                violations.append("结构化输出")
            if any(m in answer for m in case.get("forbidden_markers", [])):
                violations.append("模板标记残留")
            if any(m in answer for m in QUESTION_MARKERS):
                violations.append("反问")
            max_chars = case.get("max_chars")
            if max_chars and len(answer.strip()) > max_chars:
                violations.append(f"超{max_chars}字")
            evidence["instruction_violations"] = violations
            scores[dim] = 10 if not violations else (5 if len(violations) == 1 else 0)

        else:
            raise ValueError(f"{case['case_id']}: 未实现的维度 {dim}")

    return scores, evidence


def _llm_scores(case: dict, answer: str, trace: dict) -> dict:
    """按范式把 checklist + anchors 原样交给 judge 模型；失败抛错由调用方回退。"""
    from openai import OpenAI
    from settings import chat_api_key, chat_base_url, chat_model

    payload = {
        "question": case.get("user_question"),
        "expected": case.get("expected"),
        "answer": answer,
        "trace_summary": {
            "tool_calls": [c.get("name") for c in _tool_calls(trace)],
            "retrieved_pages": sorted({str(c.get("page", "")) for c in _retrieved(trace)}),
        },
        "checklist_by_dim": case.get("checklist_by_dim", {}),
        "anchors_by_dim": case.get("anchors_by_dim", {}),
    }
    prompt = (
        "你是评测判分器。依据 checklist 逐条核查，并严格按 anchors 的 0/5/10 三档给分。\n"
        "只输出 JSON 对象，key 为维度名，value 只能是 0、5、10 之一，不要解释。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    client = OpenAI(api_key=chat_api_key, base_url=chat_base_url)
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content)
    declared = case.get("checklist_by_dim", {})
    return {k: int(raw[k]) for k in declared if k in raw and int(raw[k]) in (0, 5, 10)}


def _hard_gate(case: dict, answer: str, trace: dict) -> list:
    gates = case.get("hard_gate", {}) or {}
    triggered = []
    if "tool_not_called_when_required" in gates:
        if case.get("expected_tool") == "required" and not any(
            c.get("name") == "rag_summarize" for c in _tool_calls(trace)
        ):
            triggered.append("tool_not_called_when_required")
    if "fabricated_out_of_kb" in gates:
        markers = case.get("abstain_markers", list(ABSTAIN_DEFAULT))
        abstained = any(_norm(m) in _norm(answer) for m in markers)
        if not abstained and len(answer.strip()) > 25:
            triggered.append("fabricated_out_of_kb")
    return triggered


def _aggregate(case: dict, scores: dict, header: dict) -> tuple:
    base = {d["key"]: d["weight"] for d in header["profile"]["dimensions"]}
    override = case.get("dimension_weights") or {}
    weights = {k: override.get(k, base.get(k, 0.0)) for k in scores}
    total = sum(weights.values())
    if total <= 0:
        return 0.0, weights
    return round(sum(scores[k] * weights[k] for k in scores) / total * 10, 1), weights


def score_case(case: dict, answer: str, trace: dict, header: dict, mode: str = "rule") -> dict:
    mode_used = mode
    if mode == "llm":
        try:
            scores = _llm_scores(case, answer, trace)
            if set(scores) != set(case.get("checklist_by_dim", {})):
                raise ValueError("LLM judge 返回的维度不完整")
            evidence = {"source": "llm"}
        except Exception as exc:
            scores, evidence = _rule_scores(case, answer, trace)
            evidence["llm_fallback"] = f"{type(exc).__name__}: {exc}"
            mode_used = "rule(llm_fallback)"
    else:
        scores, evidence = _rule_scores(case, answer, trace)

    score, weights = _aggregate(case, scores, header)
    agg = dict(header["profile"]["aggregation"])
    agg.update(case.get("aggregation", {}))
    gate = _hard_gate(case, answer, trace) if agg.get("hard_gate") else []
    if gate:
        score = 0.0
    verdict = (
        "excellent" if score >= agg["excellent_threshold"]
        else "pass" if score >= agg["pass_threshold"]
        else "fail"
    )
    return {
        "case_id": case["case_id"],
        "case_name": case["case_name"],
        "case_type": case["case_type"],
        "difficulty": case["difficulty"],
        "mode": mode_used,
        "scores_by_dim": scores,
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "score": score,
        "verdict": verdict,
        "gate_triggered": gate,
        "evidence": evidence,
        "answer": answer,
    }


def summarize(results: list, cases: dict, header: dict) -> dict:
    n = len(results)
    by_type: dict[str, list] = {}
    by_dim: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["case_type"], []).append(r["score"])
        for k, v in r["scores_by_dim"].items():
            by_dim.setdefault(k, []).append(v)
    return {
        "dataset": header["dataset_name"],
        "case_count": n,
        "mean_score": round(sum(r["score"] for r in results) / n, 1) if n else 0,
        "pass": sum(1 for r in results if r["verdict"] in ("pass", "excellent")),
        "excellent": sum(1 for r in results if r["verdict"] == "excellent"),
        "fail": sum(1 for r in results if r["verdict"] == "fail"),
        "hard_gate_hits": sum(1 for r in results if r["gate_triggered"]),
        "by_case_type": {
            k: round(sum(v) / len(v), 1) for k, v in sorted(by_type.items())
        },
        "by_dimension_mean": {
            k: round(sum(v) / len(v), 1) for k, v in sorted(by_dim.items())
        },
    }


def print_table(results: list, summary: dict) -> None:
    print("\n{:<10} {:<16} {:<14} {:>6}  {:<10} {}".format(
        "case", "名称", "类型", "得分", "结论", "门控/关键缺口"))
    print("-" * 100)
    for r in results:
        gap = ",".join(r["gate_triggered"]) or ",".join(
            f"{k}:{v}" for k, v in r["scores_by_dim"].items() if v <= 5
        ) or "-"
        print("{:<10} {:<16} {:<14} {:>6}  {:<10} {}".format(
            r["case_id"], r["case_name"][:14], r["case_type"],
            r["score"], r["verdict"], gap[:48]))
    print("-" * 100)
    print(f"均值 {summary['mean_score']} | pass {summary['pass']}/{summary['case_count']} "
          f"| excellent {summary['excellent']} | 硬门控 {summary['hard_gate_hits']}")
    print("分类型:", summary["by_case_type"])
    print("分维度:", summary["by_dimension_mean"])


def _load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="对已有运行结果重判（不重跑 Agent）")
    parser.add_argument("--input", required=True, help="runner 产出的 jsonl")
    parser.add_argument("--judge", choices=["rule", "llm"], default="rule")
    args = parser.parse_args()

    header = json.loads((EVAL / "header.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in _load(EVAL / "cases.jsonl")}
    records = _load(Path(args.input))
    results = [
        score_case(cases[r["case_id"]], r.get("answer", ""), r.get("trace", {}), header, args.judge)
        for r in records if r["case_id"] in cases
    ]
    summary = summarize(results, cases, header)
    print_table(results, summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    (RESULTS / f"{stem}_judged.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {RESULTS / (stem + '_judged.json')}")


if __name__ == "__main__":
    main()
