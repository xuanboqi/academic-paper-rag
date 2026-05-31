<h1 align="center">Academic Paper RAG</h1>

<p align="center">
  <strong>Qwen3-Embedding + DeepSeek V4 + PostgreSQL/pgvector</strong>
</p>

<p align="center">
  面向学术 PDF 文献的本地优先 RAG 系统，支持文献导入、切片向量化、语义检索和带来源引用的智能问答。
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-后端-009688?logo=fastapi&logoColor=white"></a>
  <a href="https://react.dev/"><img alt="React TypeScript" src="https://img.shields.io/badge/React%20%2B%20TypeScript-前端-61DAFB?logo=react&logoColor=111827"></a>
  <a href="https://github.com/pgvector/pgvector"><img alt="pgvector" src="https://img.shields.io/badge/PostgreSQL%20%2B%20pgvector-向量数据库-4169E1?logo=postgresql&logoColor=white"></a>
  <img alt="License" src="https://img.shields.io/badge/License-No%20license-lightgrey">
</p>

<p align="center">
  <a href="README_EN.md">English</a> | 中文文档
</p>

---

这个项目用于构建一个面向学术 PDF 文献的 RAG 系统。

你的数据集目录就是：

```text
Database/
```

整体流程是：

```text
Database 中的 PDF 文献
  -> 抽取文本
  -> 清洗文本
  -> 重叠切片
  -> 使用本地 Qwen3-Embedding 生成向量
  -> 存入 PostgreSQL + pgvector

用户问题
  -> 使用同一个 Qwen3-Embedding 生成问题向量
  -> 在 pgvector 中检索 top-k 相关文本块
  -> 把检索到的片段交给 DeepSeek V4
  -> 输出带来源引用的中文回答
```

## 模型方案

### Embedding 模型

使用本地 Qwen3-Embedding：

```text
Qwen/Qwen3-Embedding-0.6B
```

建议把模型文件放在：

```text
models/Qwen3-Embedding-0.6B
```

最终目录结构建议是：

```text
academic-paper-rag/
├── Database/
├── models/
│   └── Qwen3-Embedding-0.6B/
├── scripts/
├── .env
├── docker-compose.yml
└── requirements.txt
```

`models/` 已经加入 `.gitignore`，模型文件不会被提交到 Git。

### 问答模型

问答生成使用 DeepSeek V4：

```text
deepseek-v4-flash
```

如果你需要更强的推理和综合能力，可以在 `.env` 中改成：

```text
deepseek-v4-pro
```

## 启动 PostgreSQL

项目已经提供 PostgreSQL + pgvector 的 `docker-compose.yml`：

```powershell
docker compose up -d
```

默认连接地址：

```text
postgresql://postgres:postgres@127.0.0.1:15433/literature_rag
```

## 安装依赖

建议使用项目本地虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

复制环境变量文件：

```powershell
Copy-Item .env.example .env
```

然后修改 `.env`：

```env
POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:15433/literature_rag

EMBEDDING_MODEL=./models/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=cuda
EMBEDDING_TRUST_REMOTE_CODE=true
EMBEDDING_HALF_PRECISION=true
EMBEDDING_MAX_SEQ_LENGTH=1024
EMBEDDING_QUERY_INSTRUCTION=Given a question, retrieve relevant academic paper passages that answer the question.

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

DATA_DIR=Database
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
BATCH_SIZE=4
TOP_K=8
```

## 导入文献

如果你使用的是已经安装 CUDA 版 PyTorch 的 Conda 环境，请先 `conda activate LLM`，然后直接用 `python` 运行脚本，不要再用 `.venv\Scripts\python.exe`。

日常建议使用这个“训练式”脚本。它不是微调 Qwen3 模型，而是把 `Database` 里的 PDF 构建成 RAG 向量库：

```powershell
conda activate LLM
python scripts\train_literature_rag.py --quick-test
```

第一次成功后，后续继续跑或断点续跑，不需要删除已有数据：

```powershell
python scripts\train_literature_rag.py --batch-size 4
```

它会打印当前 Python 路径、PyTorch/CUDA 状态、embedding 设备、向量维度、PDF 进度和最终写入的 chunk 数量。

## 按论文标题重命名 PDF

可以先预览改名方案：

```powershell
conda activate LLM
python scripts\rename_pdfs_by_title.py
```

确认无误后执行真实改名：

```powershell
python scripts\rename_pdfs_by_title.py --apply
```

脚本会优先读取 PDF 元数据里的标题，取不到时再从首页文本推断标题。执行 `--apply` 时，如果 `.env` 中配置了 `POSTGRES_DSN`，会同步更新 `literature_documents` 表里的 `filename` 和 `path`。

先用 3 篇 PDF 做小样本测试：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_literature.py --limit 3 --recreate
```

确认没有问题后，全量导入：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_literature.py --recreate
```

只要你更换了 embedding 模型，或者向量维度发生变化，就应该使用 `--recreate` 重建表并重新入库。

## 测试向量检索

英文问题：

```powershell
.\.venv\Scripts\python.exe scripts\search_literature.py "multimodal object detection under adverse weather"
```

中文问题：

```powershell
.\.venv\Scripts\python.exe scripts\search_literature.py "恶劣天气下的多模态目标检测"
```

脚本会返回最相关的文本块、PDF 文件名、页码范围和相似度分数。

## 分析入库耗时

如果上传或向量化很慢，可以用命令行 profiling 脚本查看各阶段耗时：

```powershell
conda activate LLM
python scripts\profile_literature_ingest.py --limit 1 --batch-size 2
```

默认不会写入数据库，只统计：

```text
hash
extract_text
chunk
embedding
```

如果也想统计 PostgreSQL 写入耗时：

```powershell
python scripts\profile_literature_ingest.py --limit 1 --batch-size 2 --write-db
```

指定某一篇 PDF：

```powershell
python scripts\profile_literature_ingest.py --pdf "Database\your-paper.pdf" --batch-size 2
```

当前项目默认启用了两项 embedding 加速配置：

```env
EMBEDDING_HALF_PRECISION=true
EMBEDDING_MAX_SEQ_LENGTH=1024
BATCH_SIZE=4
```

含义：

- `EMBEDDING_HALF_PRECISION=true`：CUDA 上使用 FP16，通常能明显降低显存占用并提升速度。
- `EMBEDDING_MAX_SEQ_LENGTH=1024`：限制每个 chunk 输入 embedding 模型的最大 token 长度，避免超长文本拖慢推理。
- `BATCH_SIZE=4`：RTX 3060 上比较稳。如果显存充足可以测试 `8`，但不一定更快。

注意：如果你修改了 `EMBEDDING_MAX_SEQ_LENGTH`，建议重新构建全量向量库，保证新旧文档使用一致的 embedding 规则。

## 调用 DeepSeek V4 问答

完成入库后，可以直接提问：

```powershell
.\.venv\Scripts\python.exe scripts\ask_literature.py "恶劣天气下的多模态目标检测有哪些方法？"
```

这个脚本会执行完整 RAG 流程：

1. 用 Qwen3-Embedding 把问题转成向量。
2. 从 PostgreSQL + pgvector 中检索相关论文片段。
3. 把召回片段拼成上下文。
4. 调用 DeepSeek V4 生成中文回答。
5. 输出答案和来源列表。

## Web 可视化界面

项目已经提供一个 TypeScript + React 前端和 FastAPI 后端，界面拆成三个页面：

- `文献管理`：查看 `Database/` 里的 PDF、入库状态、页数、chunk 数和文本块预览。
- `文献导入`：在文献管理页上传本地 PDF，系统会自动提取论文标题重命名，并立即写入向量数据库。
- `向量化管理`：选择 Qwen3-Embedding-0.6B、设置 batch size、是否跳过已入库文献、是否重建向量表，并查看任务日志。
- `智能问答`：输入问题，先用 pgvector 检索相关文献片段，再调用 DeepSeek V4 生成带来源的答案。

启动方式：

```powershell
docker compose up -d
.\scripts\start_web_ui.ps1
```

然后打开：

```text
http://127.0.0.1:5174
```

### 已补充的工程化能力

- 上传 PDF 和单篇重新向量化均采用后台任务执行，前端通过任务接口轮询真实进度。
- 进度阶段包括 `hash`、`extract_text`、`chunk`、`embedding`、`write_db`、`completed`。
- 上传重复 PDF 时会通过 SHA256 文件哈希识别，避免重复生成向量，并复用已有向量记录。
- 如果 PDF 文本抽取结果为 0 页或 0 chunks，界面会提示该文件可能是扫描版 PDF，需要 OCR。
- 智能问答页面会展示召回来源，包括文件名、页码范围和相似度分数，方便核查回答依据。

### 检索效果评估

可以用 JSONL 文件定义一组问题和期望命中的论文/关键词：

```json
{"question":"What method is proposed for multi-modal target detection in dynamic environments?","expected_files":["remotesensing-18-01731"],"expected_terms":["StarRoute-DBNet","multi-modal"]}
```

运行评估脚本：

```powershell
conda activate LLM
python scripts\evaluate_retrieval.py --cases eval_cases.example.jsonl --top-k 8
```

脚本会输出：

- `file_hit_rate`：Top-K 结果是否命中预期文件。
- `term_hit_rate`：Top-K 文本是否包含预期关键词。
- `avg_top_score`：每个问题最高召回分数的平均值。

启动脚本会在后台运行前后端，并把日志写入：

```text
logs/backend.log
logs/backend.err.log
logs/frontend.log
logs/frontend.err.log
```

停止 Web 服务：

```powershell
.\scripts\stop_web_ui.ps1
```

也可以手动分两个终端启动：

```powershell
conda activate LLM
uvicorn backend.main:app --host 127.0.0.1 --port 8002
```

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

## 数据库表

脚本会自动创建两张表：

- `literature_documents`：保存 PDF 文件名、路径、文件哈希、页数等信息。
- `literature_chunks`：保存文本块、页码范围、向量和 pgvector 索引。

注意：同一张向量表里不要混用不同 embedding 模型生成的向量。更换 `EMBEDDING_MODEL` 后请重新入库。

## 说明

- 当前 PDF 文本抽取使用 `pypdf`，适合文本型 PDF。
- 如果某些 PDF 是扫描版，需要额外接 OCR。
- Qwen3-Embedding 在本地运行，完整论文内容不需要发送给在线 embedding API。
- DeepSeek V4 只接收检索出来的少量相关片段，不会接收整个文献库。
