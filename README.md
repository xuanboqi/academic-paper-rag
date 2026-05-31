# Academic Paper RAG: Qwen3-Embedding + DeepSeek V4

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React%20%2B%20TypeScript-Frontend-61DAFB?logo=react&logoColor=111827)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20%2B%20pgvector-Vector%20DB-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-No%20license-lightgrey)](#)

English | [中文文档](README_CN.md)

This project builds a RAG pipeline for academic PDF literature.

The dataset directory is fixed to:

```text
Database/
```

The recommended architecture is:

```text
PDFs in Database
  -> extract text
  -> clean text
  -> split into overlapping chunks
  -> embed chunks with local Qwen3-Embedding
  -> store chunks and vectors in PostgreSQL + pgvector

User question
  -> embed the question with the same Qwen3-Embedding model
  -> retrieve top-k chunks from pgvector
  -> send retrieved chunks to DeepSeek V4
  -> return a Chinese answer with source citations
```

## Models

### Embedding Model

Use Qwen3-Embedding locally:

```text
Qwen/Qwen3-Embedding-0.6B
```

Place the downloaded model here:

```text
models/Qwen3-Embedding-0.6B
```

Final project structure:

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

The `models/` directory is ignored by Git because model files are large.

### Answer Model

DeepSeek V4 is used for answer generation after retrieval:

```text
deepseek-v4-flash
```

You can switch to `deepseek-v4-pro` in `.env` if you need stronger reasoning.

## PostgreSQL

This repo includes a PostgreSQL + pgvector Docker Compose file:

```powershell
docker compose up -d
```

Default connection:

```text
postgresql://postgres:postgres@127.0.0.1:15433/literature_rag
```

## Setup

Create a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Copy the environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

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

## Ingest Literature

If you use a Conda environment with CUDA PyTorch, activate it first and run `python` directly instead of `.venv\Scripts\python.exe`.

For day-to-day corpus building, use the training-style runner. It does not fine-tune Qwen3; it builds the RAG vector database from your PDFs:

```powershell
conda activate LLM
python scripts\train_literature_rag.py --quick-test
```

After the first successful run, continue or resume without deleting existing rows:

```powershell
python scripts\train_literature_rag.py --batch-size 4
```

It prints the Python executable, PyTorch/CUDA status, embedding device, model dimension, PDF progress, and final chunk count.

## Rename PDFs By Paper Title

Preview the rename plan first:

```powershell
conda activate LLM
python scripts\rename_pdfs_by_title.py
```

Apply the rename after checking the plan:

```powershell
python scripts\rename_pdfs_by_title.py --apply
```

The script reads the PDF metadata title first, then falls back to first-page title extraction. When `--apply` is used and `POSTGRES_DSN` is configured in `.env`, it also updates `literature_documents.filename` and `literature_documents.path`.

Start with a small test:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_literature.py --limit 3 --recreate
```

Then ingest all PDFs:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_literature.py --recreate
```

Use `--recreate` when you change the embedding model or vector dimension.

## Test Vector Retrieval

```powershell
.\.venv\Scripts\python.exe scripts\search_literature.py "multimodal object detection under adverse weather"
```

Chinese queries also work:

```powershell
.\.venv\Scripts\python.exe scripts\search_literature.py "恶劣天气下的多模态目标检测"
```

## Profile Ingestion Time

When upload or vectorization feels slow, use the command-line profiling script:

```powershell
conda activate LLM
python scripts\profile_literature_ingest.py --limit 1 --batch-size 2
```

By default, it does not write to the database. It measures:

```text
hash
extract_text
chunk
embedding
```

To include PostgreSQL write time:

```powershell
python scripts\profile_literature_ingest.py --limit 1 --batch-size 2 --write-db
```

To profile one specific PDF:

```powershell
python scripts\profile_literature_ingest.py --pdf "Database\your-paper.pdf" --batch-size 2
```

The project now enables these embedding speed settings by default:

```env
EMBEDDING_HALF_PRECISION=true
EMBEDDING_MAX_SEQ_LENGTH=1024
BATCH_SIZE=4
```

- `EMBEDDING_HALF_PRECISION=true`: use FP16 on CUDA to reduce VRAM and improve throughput.
- `EMBEDDING_MAX_SEQ_LENGTH=1024`: cap the token length sent into the embedding model.
- `BATCH_SIZE=4`: a stable default for RTX 3060. You can test `8`, but it is not always faster.

If you change `EMBEDDING_MAX_SEQ_LENGTH`, rebuild the full vector database so old and new documents use the same embedding rule.

## Ask With DeepSeek V4

After ingestion, ask a question:

```powershell
.\.venv\Scripts\python.exe scripts\ask_literature.py "恶劣天气下的多模态目标检测有哪些方法？"
```

The script will:

1. Embed the question with Qwen3-Embedding.
2. Retrieve the most relevant chunks from PostgreSQL.
3. Send only those chunks to DeepSeek V4.
4. Print the answer and source list.

## Web UI

The project includes a TypeScript + React frontend and a FastAPI backend. The UI is split into three pages:

- `Library`: inspect PDFs in `Database/`, ingestion status, page counts, chunk counts, and chunk previews.
- `PDF upload`: upload a local PDF from the Library page. The backend renames it by paper title and immediately writes it into the vector database.
- `Vectorization`: use Qwen3-Embedding-0.6B, set batch size, skip existing documents, rebuild the vector table, and watch task logs.
- `Q&A`: ask questions, retrieve relevant chunks from pgvector, then call DeepSeek V4 for sourced answers.

Start it with:

```powershell
docker compose up -d
.\scripts\start_web_ui.ps1
```

Then open:

```text
http://127.0.0.1:5174
```

The startup script runs backend and frontend in the background and writes logs to:

```text
logs/backend.log
logs/backend.err.log
logs/frontend.log
logs/frontend.err.log
```

Stop the web services with:

```powershell
.\scripts\stop_web_ui.ps1
```

You can also start the two services manually:

```powershell
conda activate LLM
uvicorn backend.main:app --host 127.0.0.1 --port 8002
```

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

## Database Tables

The ingestion script creates:

- `literature_documents`: PDF metadata, file hash, path, page count.
- `literature_chunks`: chunk text, page range, embedding vector, pgvector index.

Do not mix vectors from different embedding models in the same table. If you change `EMBEDDING_MODEL`, run ingestion with `--recreate`.

## Notes

- `pypdf` works well for text-based PDFs. Scanned PDFs need OCR.
- Qwen3-Embedding runs locally, so the full PDF text does not need to be sent to an online embedding API.
- DeepSeek V4 receives only the retrieved passages, not the whole literature library.
