# -*- coding: utf-8 -*-
"""评测驱动器：逐条提问 → 收集 answer + trace → 交 judge 判分 → 落盘。

    python3 -m eval.runner --dry-run                 # 只校验数据集与知识库快照
    python3 -m eval.runner --mock                    # 离线端到端自检（不调 LLM）
    python3 -m eval.runner --judge rule --limit 3    # 真实跑前 3 条
    python3 -m eval.runner --judge llm               # 用 LLM judge 判软维度

真实模式需要带全套依赖的解释器（本机为 /opt/miniconda3/envs/agentenv/bin/python），
且会真实调用嵌入/对话模型产生费用；先跑 --mock 和 --limit 1 确认链路再全量。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
RESULTS = EVAL / "results"
ENTRY = ROOT / "agent" / "praisonai-Agent.py"


def _load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(header: dict, cases: list) -> list:
    """校验两级配置与冻结基线。返回问题列表（空 = 全通过）。"""
    problems = []
    dims = header["profile"]["dimensions"]
    keys = {d["key"] for d in dims}
    weight_sum = sum(d["weight"] for d in dims)
    if abs(weight_sum - 1.0) > 1e-6:
        problems.append(f"header 维度权重和={weight_sum}，应为 1.0")
    if len(keys) != len(dims):
        problems.append("header 中存在重复维度 key")

    seen = set()
    for c in cases:
        cid = c.get("case_id", "<无 id>")
        if cid in seen:
            problems.append(f"{cid} 重复")
        seen.add(cid)
        declared = set(c.get("checklist_by_dim", {}))
        anchors = set(c.get("anchors_by_dim", {}))
        override = set(c.get("dimension_weights", {}))
        unknown = (declared | anchors | override) - keys
        if unknown:
            problems.append(f"{cid} 未在 header 注册的维度: {sorted(unknown)}")
        if anchors - declared:
            problems.append(f"{cid} anchors 有但 checklist 无: {sorted(anchors - declared)}")
        if declared - anchors:
            problems.append(f"{cid} checklist 有但 anchors 无: {sorted(declared - anchors)}")
        if override and abs(sum(c["dimension_weights"].values()) - 1.0) > 1e-6:
            problems.append(f"{cid} dimension_weights 和不为 1.0")
        for f in c.get("input_files", []):
            if not (EVAL / f["rel"]).exists():
                problems.append(f"{cid} 缺少证据包 {f['rel']}")

    snap = header["profile"].get("kb_snapshot", {})
    if snap.get("chunk_count"):
        db = ROOT / snap.get("persist_directory", "") / "chroma.sqlite3"
        if not db.exists():
            problems.append(f"找不到向量库 {db}")
    for doc in snap.get("documents", []):
        p = ROOT / doc["file"]
        if not p.exists():
            problems.append(f"知识库文件缺失 {doc['file']}")
        elif _md5(p) != doc["md5"]:
            problems.append(f"知识库已变更（md5 不符）: {doc['file']} —— 分数不可比，需重建快照与证据包")
    return problems


def load_agent_entry():
    """入口文件名带连字符不能 import，按路径加载；返回提供 run(question) 的模块。

    直接 `python agent/praisonai-Agent.py` 时，脚本目录（agent/）会自动进 sys.path，
    因此内部 `from tools.xxx import ...` 可解析。这里用 importlib 顶层 exec，该自动行为
    失效，需手动把 agent/ 加入 sys.path（根目录由 agent 自身第 15 行的
    sys.path.append(parent.parent) 负责）。
    """
    agent_dir = str(ENTRY.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("agentic_rag_entry", ENTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Agent 入口 {ENTRY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise RuntimeError(f"{ENTRY} 未暴露 run(question)")
    return module


def execute(case: dict, entry, mock: bool) -> dict:
    if mock:
        from eval import mock_agent
        return mock_agent.run(case)
    result = entry.run(case["user_question"])
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="agentic-rag 评测驱动器")
    parser.add_argument("--mock", action="store_true", help="离线固定响应，验证流水线")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据集与快照")
    parser.add_argument("--judge", choices=["rule", "llm"], default="rule")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    args = parser.parse_args()

    header = json.loads((EVAL / "header.json").read_text(encoding="utf-8"))
    cases = _load_jsonl(EVAL / "cases.jsonl")
    problems = validate_dataset(header, cases)
    print(f"数据集 {header['dataset_name']} | {len(cases)} 条 case | "
          f"{len(header['profile']['dimensions'])} 个维度 | "
          f"校验{'通过' if not problems else '发现 ' + str(len(problems)) + ' 个问题'}")
    for p in problems:
        print("  !", p)
    if args.dry_run:
        raise SystemExit(1 if problems else 0)
    if problems:
        print("  （知识库或配置已变更，本次分数不可与历史对比）")

    if args.limit:
        cases = cases[: args.limit]
    entry = None if args.mock else load_agent_entry()

    from eval.judge import print_table, score_case, summarize

    records, results = [], []
    for case in cases:
        try:
            out = execute(case, entry, args.mock)
            answer = out.get("answer", "") or ""
            trace = out.get("trace", {}) or {}
            error = ""
        except Exception as exc:
            answer, trace = "", {}
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ! {case['case_id']} 执行失败: {error}")
        records.append({
            "case_id": case["case_id"], "answer": answer,
            "trace": trace, "error": error, "mode": "mock" if args.mock else "live",
        })
        results.append(score_case(case, answer, trace, header, args.judge))

    summary = summarize(results, {c["case_id"]: c for c in cases}, header)
    summary["mode"] = "mock" if args.mock else "live"
    summary["judge"] = args.judge
    summary["kb_problems"] = problems
    print_table(results, summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "mock" if args.mock else "live"
    raw = RESULTS / f"run_{tag}_{stamp}.jsonl"
    judged = RESULTS / f"run_{tag}_{stamp}_judged.json"
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    judged.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n原始 trace: {raw.relative_to(ROOT)}")
    print(f"判分结果:   {judged.relative_to(ROOT)}")
    print(f"重判命令:   python3 -m eval.judge --input {raw.relative_to(ROOT)} --judge rule")


if __name__ == "__main__":
    main()
