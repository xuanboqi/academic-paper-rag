# academic-paper-rag 本地使用说明

这份文档用于说明如何在本地启动并打开 `academic-paper-rag`。

## 1. 进入项目目录

打开 PowerShell，进入项目根目录：

```powershell
cd D:\LLM\3.RAG_Practice\1.Literature-RAG
```

如果你后面已经把文件夹改名成 `academic-paper-rag`，就进入改名后的目录。

## 2. 激活 Python 环境

如果你使用的是 Conda 环境：

```powershell
conda activate LLM
```

如果你使用的是项目里的 `.venv`，可以执行：

```powershell
.\.venv\Scripts\activate
```

二选一即可，不要同时混用。

## 3. 安装依赖

第一次运行，或者我更新了 `requirements.txt` 后，需要安装依赖：

```powershell
pip install -r requirements.txt
```

如果你刚加入 LangGraph 功能，必须确保依赖里已经安装：

```powershell
pip install langgraph
```

## 4. 配置 `.env`

如果还没有 `.env`，先复制一份模板：

```powershell
Copy-Item .env.example .env
```

然后打开 `.env`，至少确认这些配置：

```env
POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:15433/literature_rag
DATA_DIR=Database
EMBEDDING_MODEL=./models/Qwen3-Embedding-0.6B
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash
```

注意：`.env` 里有 API Key，不要上传到 GitHub。

## 5. 启动 PostgreSQL + pgvector

项目使用 Docker 里的 PostgreSQL + pgvector 作为向量数据库。

启动数据库：

```powershell
docker compose up -d
```

查看数据库容器是否运行：

```powershell
docker ps
```

正常情况下能看到类似：

```text
literature-postgres   Up ...   0.0.0.0:15433->5432/tcp
```

## 6. 启动网页服务

项目已经提供启动脚本：

```powershell
.\scripts\start_web_ui.ps1
```

这个脚本会启动两个服务：

```text
FastAPI 后端：http://127.0.0.1:8002
React 前端：http://127.0.0.1:5174
```

启动后日志会写入：

```text
logs/backend.log
logs/frontend.log
```

## 7. 打开网页

浏览器打开：

```text
http://127.0.0.1:5174
```

如果打开成功，你会看到三个主要页面：

- `文献管理`：上传 PDF、查看入库状态、查看 chunk 内容。
- `向量化管理`：批量构建向量库。
- `智能问答`：基础 RAG 问答、仅检索、Agentic RAG 问答。

## 8. 停止网页服务

如果要关闭前后端服务：

```powershell
.\scripts\stop_web_ui.ps1
```

如果还想停止数据库：

```powershell
docker compose down
```

只停止网页服务不会删除数据库数据。执行 `docker compose down` 会关闭容器，但默认不会删除 volume 数据。

## 9. 常用功能

### 上传 PDF 并自动入库

打开：

```text
http://127.0.0.1:5174
```

进入 `文献管理`，点击：

```text
选择 PDF 并导入向量库
```

系统会自动执行：

```text
PDF 上传
自动提取论文标题并重命名
文本抽取
切片
Qwen3-Embedding 向量化
写入 PostgreSQL + pgvector
```

### 使用基础 RAG 问答

进入 `智能问答`，输入问题，然后点击：

```text
问答
```

系统流程：

```text
问题 -> Qwen3-Embedding -> pgvector 检索 -> DeepSeek 生成回答
```

### 使用 Agentic RAG

进入 `智能问答`，输入问题，然后点击：

```text
Agentic RAG
```

系统流程：

```text
任务识别
查询改写
证据检索
答案生成
```

页面会额外展示：

```text
任务类型
改写后的检索查询
LangGraph 工作流步骤
```

### 本地视觉大模型解析图表公式

项目支持接入本地 `Qwen2.5-VL-3B-Instruct`，用于解析 PDF 页面中的图、表、公式和复杂版面。

第一次使用先下载模型：

```powershell
python scripts\download_qwen25_vl.py
```

模型默认保存到：

```text
models/Qwen2.5-VL-3B-Instruct
```

然后解析某一篇 PDF：

```powershell
python scripts\ingest_multimodal.py --pdf "Database\你的论文.pdf" --max-pages 5
```

它会执行：

```text
PDF 候选页筛选
页面渲染成图片
Qwen2.5-VL 本地解析图表公式
解析文本向量化
写入 literature_multimodal_chunks
```

解析后的图片缓存会放在：

```text
artifacts/multimodal
```

该目录已经被 `.gitignore` 忽略，不会上传到 GitHub。

完成多模态入库后，普通问答和 Agentic RAG 会自动合并检索：

```text
正文文本 chunks + 图表公式解析 chunks
```

为了避免本地视觉模型耗时过长，脚本默认不会解析每一页，而是：

```text
先扫描每篇论文前 10 页
按 Figure / Table / Equation / architecture / ablation / mAP 等关键词打分
只把每篇论文得分最高的 3 个候选页送入 Qwen2.5-VL
已经解析过的页面会自动跳过
```

如果你想进一步加速，可以改成每篇只解析 1 页：

```powershell
python scripts\ingest_multimodal.py --max-candidate-pages 1
```

如果你想提高覆盖率，可以改成每篇解析 5 页：

```powershell
python scripts\ingest_multimodal.py --max-candidate-pages 5
```

## 10. 常见问题

### 1. 打不开 `http://127.0.0.1:5174`

先确认服务是否启动：

```powershell
.\scripts\start_web_ui.ps1
```

再查看日志：

```powershell
Get-Content logs\frontend.log -Tail 50
Get-Content logs\backend.log -Tail 50
```

### 2. 页面显示 PostgreSQL unavailable

说明数据库没连上。先启动 Docker：

```powershell
docker compose up -d
```

再检查 `.env`：

```env
POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:15433/literature_rag
```

### 3. Agentic RAG 报 LangGraph 没安装

执行：

```powershell
pip install -r requirements.txt
```

或者：

```powershell
pip install langgraph
```

然后重启服务：

```powershell
.\scripts\stop_web_ui.ps1
.\scripts\start_web_ui.ps1
```

### 4. 向量化很慢

可以先确认是否使用 GPU：

```powershell
python scripts\profile_literature_ingest.py --pdf "Database\你的论文.pdf" --batch-size 2
```

如果输出里显示：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

说明 GPU 正常。

### 5. 上传 PDF 后是 pending 或 0 chunks

可能原因：

- PDF 是扫描版，普通文本抽取不到内容。
- PDF 版式复杂，需要 OCR 或多模态解析。
- 向量化任务还没完成。

可以点文献详情里的：

```text
重新向量化当前文献
```

## 11. 最短启动命令

日常使用时，一般只需要三步：

```powershell
cd D:\LLM\3.RAG_Practice\1.Literature-RAG
conda activate LLM
docker compose up -d
.\scripts\start_web_ui.ps1
```

然后打开：

```text
http://127.0.0.1:5174
```
