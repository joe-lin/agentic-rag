# agentic-rag

基于 **praisonaiagents** 编排的 OceanBase 向量数据库技术文档问答智能体（RAG）。
检索侧采用 **Chroma + bge-m3 嵌入 + bge-reranker 重排 + 混合检索（HyDE + 向量 + 词法 bigram）**，
提供 CLI 多轮对话与 Streamlit Web UI，并自带一套可复现的评测套件。

---

## 目录结构

```
agentic-rag/
├── agent/
│   ├── praisonai-agent.py     # Agent 入口（文件名带连字符，运行时按路径加载）
│   └── tools/
│       ├── agent_tools.py     # rag_summarize 工具包装（含 trace 埋点）
│       └── wechat_publisher.py# 微信公众号一键发布工具（适配外部技能 CLI）
├── rag/
│   ├── vector_store.py        # Chroma 封装 + 词法召回（lexical_search）
│   └── rag_service.py         # 检索管线：HyDE → 向量 → 词法 → rerank
├── model/
│   └── factory.py             # LLM / Chat / Embedding 客户端（OpenAI 兼容）
├── settings/
│   └── __init__.py            # 配置唯一读取入口（os.getenv + dotenv）
├── utils/
│   ├── prompt_loader.py       # 提示词加载
│   ├── logger_handler.py      # 日志器
│   ├── file_handler.py        # PDF/TXT 加载 + MD5 去重
│   ├── path_tool.py           # 绝对路径解析
│   └── trace.py               # 评测链路追踪（零外部依赖）
├── app.py                     # Streamlit Web UI
├── eval/                      # 评测驱动器 + judge + 数据集
│   ├── runner.py              # 逐条提问 → trace → judge → 落盘
│   ├── judge.py               # 规则/LLM 判分 + 加权聚合 + 硬门控
│   ├── build_workspace.py     # 从 Chroma 快照重建证据包
│   ├── mock_agent.py          # 离线自检
│   ├── header.json            # 评测维度 / 权重 / 知识库快照
│   └── cases.jsonl            # 评测用例
├── docs/
│   └── rag-optimization-case.md  # RAG 检索优化优秀案例（排查+修复复盘）
├── data/                      # 知识库原始文件（txt/pdf），入库后写 md5
├── prompts/                   # 各角色提示词
├── requirements.txt
└── .env.example
```

---

## 环境要求

- Python 3.12（本机环境：`/opt/miniconda3/envs/agentenv`）
- 依赖见 `requirements.txt`

## 安装

```bash
conda create -n agentenv python=3.12
conda activate agentenv
pip install -r requirements.txt
```

## 配置

复制模板并填入真实密钥：

```bash
cp .env.example .env
```

| 变量 | 作用 |
|---|---|
| `CHAT_MODEL` / `CHAT_API_KEY` / `CHAT_BASE_URL` | RAG 总结链用的 `ChatOpenAI` |
| `PRAISON_CHAT_MODEL` / `PRAISON_API_KEY` / `PRAISON_BASE_URL` | Agent 主模型（praisonaiagents `LLM`，需 `openai/` 前缀） |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | 嵌入模型（OpenAI 兼容，如 SiliconFlow bge-m3） |
| `RERANK_MODEL` | 重排模型（如 `BAAI/bge-reranker-v2-m3`） |
| `HYDE_ENABLED` | 是否启用 HyDE（默认 `true`） |
| `CANDIDATE_K` | 向量召回候选数（rerank 前，默认 20） |
| `TAVILY_API_KEY` | `web_search` 工具所需 |
| `LOG_LEVEL` | 日志级别（默认 `INFO`） |

> 所有键名必须与 `settings/__init__.py` 中 `_get(...)` 读取的完全一致。

---

## 知识库构建

把知识文档（`.txt` / `.pdf`）放进 `data/`，然后执行入库（按文件 MD5 去重，
重复文件跳过）：

```bash
# 方式一
python -m rag.vector_store
# 方式二
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"
```

向量库持久化在 `data/chroma_db/`。改了 `chunk_size` / 换了 embedding / 重跑入库后，
记得重新构建评测证据包（见下）。

---

## 运行

**CLI 交互式多轮对话：**

```bash
python agent/praisonai-agent.py
# 交互命令：exit/quit/q/退出 结束；clear/重置/reset 清空上下文
```

> 文件名带连字符不可直接 `import`，请通过 `python agent/praisonai-agent.py`
> 或 `eval/runner.py`（按路径加载）启动。

**Streamlit Web UI：**

```bash
streamlit run app.py
```

> 本项目附带 `.streamlit/config.toml`，将 `server.fileWatcherType` 设为 `none`，
> 以规避 streamlit 文件监视线程与 transformers 半初始化之间的竞态（详见「环境注意」）。
> 代价：改代码后需手动重启 streamlit。

---

## 评测

自带可复现评测套件，支持规则判分（离线、零成本）与可选 LLM judge。

```bash
python -m eval.runner --dry-run              # 校验数据集与知识库快照，不跑模型
python -m eval.runner --mock                 # 离线端到端自检（不调 LLM）
python -m eval.runner --judge rule --limit 3 # 真实跑前 3 条
python -m eval.runner --judge llm            # 用 LLM judge 判软维度
# 改阈值后重判历史结果，无需重跑 Agent：
python -m eval.judge --input eval/results/run_live_xxx.jsonl --judge rule
# 重建评测证据包（知识库变更后必须重跑）：
python -m eval.build_workspace
```

评测结果落在 `eval/results/`（`run_{mode}_{时间戳}.jsonl` 原始 trace +
`run_{mode}_{时间戳}_judged.json` 判分结果）。

> Agent 侧设置了 loop-guard（`agent/praisonai-agent.py` 中
> `max_time_per_turn=600`）：单轮超过 600s 会被强制终止，避免失控重试。
> 若 web_search 等重链路确实需要更久，调大该值即可。

---

## 检索管线（核心）

`rag/rag_service.py` 的 `retriever_docs` 负责召回，流程：

1. **HyDE**：先让 LLM 基于问题生成一段"假设性答案"（含关键术语），
   用它的嵌入去做向量检索，拉近"抽象问句"与"密集事实清单答案块"的距离；
2. **向量召回**：用嵌入在 Chroma 中取 `candidate_k` 条候选；
3. **词法召回**：用中文字符 bigram 重叠率兜底命中（bi-encoder 对密集事实清单
   常漏召回，词法匹配不依赖 embedding 质量）；
4. **重排**：向量候选 + 词法候选合并去重，交由 bge-reranker 交叉编码器
   用**原始 query** 重排，取前 `k` 条（rerank 失败自动回退向量顺序）。

完整排查与修复复盘见 **[`docs/rag-optimization-case.md`](docs/rag-optimization-case.md)**。

---

## 环境注意 / 踩坑

- **`transformers` 锁定 `4.57.6`**：`langchain_core` 加载时会强制 `import transformers`；
  本项目运行在 **macOS 12** 上，`transformers>=5` 需要 torchvision，而 torchvision 的
  macOS 13+ wheel 在该平台装不出来，会导致导入失败。故锁定 4.x。升级到 macOS 13+ 后可放开。
- **streamlit 文件监视竞态**：默认 `fileWatcherType` 会在独立线程触发 transformers 懒加载，
  与脚本线程的半初始化竞态，偶发 `ImportError: cannot import name 'is_flax_available'`，
  并进一步引发流式中断与 loop-guard 超时。已通过 `.streamlit/config.toml` 关闭监视器规避。
- **`RERANK_MODEL` 键名**：`.env` 中必须用 `RERANK_MODEL`（不是 `RERANKER_MODEL`），
  否则重排模型不会按预期覆盖。

---

## 许可证

见 `LICENSE`。
