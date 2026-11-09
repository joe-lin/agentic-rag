# -*- coding: utf-8 -*-
"""从冻结的 Chroma 快照抽取 golden 原文，重建 eval/workspace 证据包。

用法：python3 -m eval.build_workspace

注意：CASE_CHUNKS 里的数字是 embedding_metadata.id（Chroma 内部整数编号），
只在 header.json 的 kb_snapshot 不变时有效。知识库一旦重建（换 embedding、
调 chunk_size、重跑 load_document），这些 id 会漂移，必须按页码 + 关键词
重新定位后再更新本表，然后全量重跑评测。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "chroma_db" / "chroma.sqlite3"
OUT = ROOT / "eval" / "workspace"

CASE_CHUNKS: dict[str, list[int]] = {
    "case_001": [298],
    "case_002": [298, 299],
    "case_003": [357],
    "case_004": [351],
    "case_006": [14],
    "case_010": [299],
}


def _text(conn: sqlite3.Connection, chunk_id: int) -> str:
    row = conn.execute(
        "select string_value from embedding_metadata where id=? and key='chroma:document'",
        (chunk_id,),
    ).fetchone()
    return (row[0] if row else "").strip()


def _meta(conn: sqlite3.Connection, chunk_id: int) -> dict:
    rows = conn.execute(
        "select key, coalesce(string_value, int_value, float_value) "
        "from embedding_metadata where id=? and key in ('source','page')",
        (chunk_id,),
    ).fetchall()
    data = dict(rows)
    data["source"] = str(data.get("source", "")).rsplit("/", 1)[-1]
    return data


def build() -> None:
    if not DB.exists():
        raise SystemExit(f"找不到 Chroma 快照：{DB}")
    conn = sqlite3.connect(str(DB))
    for case_id, chunk_ids in CASE_CHUNKS.items():
        case_dir = OUT / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        parts = [
            f"# {case_id} 固定证据包",
            "# 快照：collection=agentic-rag / k=3 / chunk_size=200 / embedding=BAAI/bge-m3",
            "# 用途：judge 核对 groundedness 与 retrieval_quality；不喂给被测 Agent。",
            "",
        ]
        for cid in chunk_ids:
            meta = _meta(conn, cid)
            parts.append(f"[chunk {cid}] source={meta.get('source','')} page={meta.get('page','')}")
            parts.append(_text(conn, cid))
            parts.append("---")
        target = case_dir / "golden.txt"
        target.write_text("\n".join(parts), encoding="utf-8")
        print(f"{case_id}: {len(chunk_ids)} 块 -> {target.relative_to(ROOT)}")
    conn.close()


if __name__ == "__main__":
    build()
