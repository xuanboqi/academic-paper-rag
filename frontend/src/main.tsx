import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  Database,
  FileText,
  Gauge,
  Layers,
  Library,
  Play,
  RefreshCw,
  Search,
  Settings2,
  Square,
  Upload,
} from "lucide-react";
import "./styles.css";

type Page = "library" | "vectorize" | "qa";
type SortKey = "filename" | "page_count" | "chunk_count" | "status" | "size_mb" | "created_at";
type SortDirection = "asc" | "desc";

// 这些类型对应 FastAPI 返回的数据结构，前端靠它们约束接口字段。
type Health = {
  database: string;
  documents: number;
  chunks: number;
  multimodal_chunks: number;
  embedding_model: string;
  embedding_device: string;
  embedding_dimension: number;
  cuda_available: boolean;
  gpu: string | null;
  deepseek_model: string;
};

type DocumentRow = {
  id: string | null;
  filename: string;
  path: string;
  page_count: number | null;
  chunk_count: number;
  multimodal_chunk_count: number;
  created_at: string | null;
  status: string;
  size_mb: number | null;
};

type Hit = {
  filename: string;
  page_start: number;
  page_end: number;
  text: string;
  score: number;
  source_type?: string;
};

type AgentAskResult = {
  answer: string;
  hits: Hit[];
  model: string;
  task_type: string;
  rewritten_query: string;
  steps: string[];
};

type VectorStatus = {
  status: string;
  running: boolean;
  started_at: number | null;
  logs: string[];
};

type UploadResult = {
  original_filename: string;
  filename: string;
  title: string | null;
  page_count: number | null;
  chunk_count: number | null;
  task_id: string | null;
  status: string;
  duplicate: boolean;
  existing_filename: string | null;
  warning: string | null;
};

type RevectorizeResult = {
  filename: string;
  path: string;
  task_id: string;
  status: string;
};

type IngestTaskStatus = {
  id: string;
  kind: string;
  filename: string;
  path: string;
  status: string;
  stage: string;
  percent: number;
  message: string;
  duplicate: boolean;
  existing_filename: string | null;
  warning: string | null;
  error: string | null;
  page_count: number | null;
  chunk_count: number | null;
};

// 轻量 API 封装：统一处理 JSON 请求和错误抛出。
const api = {
  async get<T>(url: string): Promise<T> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async post<T>(url: string, body?: unknown): Promise<T> {
    const res = await fetch(url, {
      method: "POST",
      headers: body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

function App() {
  // 三页式应用外壳：文献管理、向量化管理、智能问答。
  const [page, setPage] = useState<Page>("qa");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  const refreshHealth = async () => {
    try {
      setHealth(await api.get<Health>("/api/health"));
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    refreshHealth();
    const id = window.setInterval(refreshHealth, 10000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Layers size={20} /></div>
          <div>
            <strong>academic-paper-rag</strong>
            <span>Qwen3 + DeepSeek</span>
          </div>
        </div>
        <nav>
          <NavButton active={page === "library"} icon={<Library size={18} />} label="文献管理" onClick={() => setPage("library")} />
          <NavButton active={page === "vectorize"} icon={<Database size={18} />} label="向量化管理" onClick={() => setPage("vectorize")} />
          <NavButton active={page === "qa"} icon={<Bot size={18} />} label="智能问答" onClick={() => setPage("qa")} />
        </nav>
      </aside>
      <main className="main">
        <StatusBar health={health} onRefresh={refreshHealth} />
        {error && <div className="error-banner">{error}</div>}
        {page === "library" && <LibraryPage onRefreshHealth={refreshHealth} />}
        {page === "vectorize" && <VectorizePage onRefreshHealth={refreshHealth} />}
        {page === "qa" && <QAPage />}
      </main>
    </div>
  );
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`nav-button ${active ? "active" : ""}`} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function StatusBar({ health, onRefresh }: { health: Health | null; onRefresh: () => void }) {
  // 顶部状态栏展示后端、数据库、模型和向量库的当前状态。
  return (
    <header className="status-bar">
      <div className="status-pill ok"><Database size={16} /> PostgreSQL {health?.database ?? "checking"}</div>
      <div className="status-pill"><Gauge size={16} /> {health?.embedding_device ?? "-"} {health?.embedding_dimension ?? "-"}d</div>
      <div className="status-pill"><Bot size={16} /> {health?.deepseek_model ?? "-"}</div>
      <div className="status-pill"><FileText size={16} /> {health?.documents ?? 0} docs / {health?.chunks ?? 0} chunks</div>
      <div className="status-pill"><Layers size={16} /> {health?.multimodal_chunks ?? 0} multimodal</div>
      <button className="icon-button" onClick={onRefresh} title="刷新状态"><RefreshCw size={16} /></button>
    </header>
  );
}

function LibraryPage({ onRefreshHealth }: { onRefreshHealth: () => void }) {
  // 文献管理页：把 Database/ 中的 PDF 和数据库入库记录合并展示。
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [selected, setSelected] = useState<DocumentRow | null>(null);
  const [chunks, setChunks] = useState<{ chunk_index: number; page_start: number; page_end: number; text: string }[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [revectorizing, setRevectorizing] = useState(false);
  const [revectorizeMessage, setRevectorizeMessage] = useState("");
  const [revectorizeError, setRevectorizeError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [ingestTask, setIngestTask] = useState<IngestTaskStatus | null>(null);

  const load = async () => {
    const data = await api.get<{ documents: DocumentRow[] }>("/api/library/documents");
    setDocuments(data.documents);
    if (!selected && data.documents.length) {
      setSelected(data.documents[0]);
    } else if (selected) {
      const fresh = data.documents.find((doc) => doc.path === selected.path || doc.filename === selected.filename);
      if (fresh) setSelected(fresh);
    }
    return data.documents;
  };

  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!selected?.id) {
      setChunks([]);
      return;
    }
    api.get<{ chunks: typeof chunks }>(`/api/library/documents/${selected.id}/chunks`).then((data) => setChunks(data.chunks));
  }, [selected?.id]);

  const stats = useMemo(() => {
    const indexed = documents.filter((doc) => doc.status === "indexed").length;
    const abnormal = documents.filter((doc) => doc.status === "indexed" && (!doc.page_count || !doc.chunk_count)).length;
    const seenIds = new Set<string>();
    const seenMmIds = new Set<string>();
    const chunks = documents.reduce((sum, doc) => {
      if (!doc.id || seenIds.has(doc.id)) return sum;
      seenIds.add(doc.id);
      return sum + doc.chunk_count;
    }, 0);
    const multimodalChunks = documents.reduce((sum, doc) => {
      if (!doc.id || seenMmIds.has(doc.id)) return sum;
      seenMmIds.add(doc.id);
      return sum + (doc.multimodal_chunk_count ?? 0);
    }, 0);
    return { indexed, abnormal, total: documents.length, chunks, multimodalChunks };
  }, [documents]);

  const sortedDocuments = useMemo(() => {
    const direction = sortDirection === "asc" ? 1 : -1;
    return [...documents].sort((left, right) => {
      const leftValue = sortValue(left, sortKey);
      const rightValue = sortValue(right, sortKey);
      if (leftValue < rightValue) return -1 * direction;
      if (leftValue > rightValue) return 1 * direction;
      return left.filename.localeCompare(right.filename, "zh-Hans-CN");
    });
  }, [documents, sortDirection, sortKey]);

  const setSort = (nextKey: SortKey) => {
    if (sortKey === nextKey) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(nextKey);
    setSortDirection(nextKey === "filename" || nextKey === "status" ? "asc" : "desc");
  };

  const waitForIngestTask = async (taskId: string): Promise<IngestTaskStatus> => {
    let latest: IngestTaskStatus | null = null;
    while (true) {
      latest = await api.get<IngestTaskStatus>(`/api/ingest/tasks/${taskId}`);
      setIngestTask(latest);
      if (latest.status === "completed" || latest.status === "failed") return latest;
      await new Promise((resolve) => window.setTimeout(resolve, 900));
    }
  };

  const uploadPdf = async (file: File | null) => {
    if (!file) return;
    setUploading(true);
    setUploadError("");
    setUploadResult(null);
    setIngestTask(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await api.post<UploadResult>("/api/documents/upload", formData);
      setUploadResult(result);
      if (result.task_id) {
        const task = await waitForIngestTask(result.task_id);
        if (task.status === "failed") throw new Error(task.error ?? "Upload ingest failed");
        setUploadResult({
          ...result,
          page_count: task.page_count,
          chunk_count: task.chunk_count,
          status: task.status,
          warning: task.warning,
        });
      } else if (result.warning) {
        setIngestTask({
          id: "duplicate",
          kind: "upload",
          filename: result.filename,
          path: "",
          status: result.status,
          stage: "duplicate",
          percent: 100,
          message: result.warning,
          duplicate: result.duplicate,
          existing_filename: result.existing_filename,
          warning: result.warning,
          error: null,
          page_count: result.page_count,
          chunk_count: result.chunk_count,
        });
      }
      await load();
      await onRefreshHealth();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const refreshLibrary = async () => {
    setRefreshing(true);
    try {
      await load();
      await onRefreshHealth();
    } finally {
      setRefreshing(false);
    }
  };

  const revectorizeSelected = async () => {
    if (!selected) return;
    setRevectorizing(true);
    setRevectorizeError("");
    setRevectorizeMessage("");
    setIngestTask(null);
    try {
      const result = await api.post<RevectorizeResult>("/api/library/revectorize", { path: selected.path });
      const task = await waitForIngestTask(result.task_id);
      if (task.status === "failed") throw new Error(task.error ?? "Revectorize failed");
      setRevectorizeMessage(`${task.filename}: ${task.page_count ?? 0} pages / ${task.chunk_count ?? 0} chunks`);
      const freshDocuments = await load();
      await onRefreshHealth();
      const freshSelected = freshDocuments.find((doc) => doc.path === result.path || doc.filename === result.filename);
      if (freshSelected) setSelected(freshSelected);
    } catch (err) {
      setRevectorizeError(err instanceof Error ? err.message : String(err));
    } finally {
      setRevectorizing(false);
    }
  };

  return (
    <section className="page-grid library-grid">
      <div className="section full">
        <div className="section-title">
          <h1>文献管理</h1>
          <button className="secondary-button" onClick={refreshLibrary} disabled={refreshing}>
            <RefreshCw size={16} />
            {refreshing ? "刷新中..." : "刷新"}
          </button>
        </div>
        <div className="metrics">
          <Metric label="PDF 总数" value={stats.total} />
          <Metric label="已入库" value={stats.indexed} />
          <Metric label="文本块" value={stats.chunks} />
          <Metric label="多模态块" value={stats.multimodalChunks} />
          <Metric label="异常文献" value={stats.abnormal} tone={stats.abnormal ? "warn" : "ok"} />
        </div>
      </div>
      <div className="section full upload-section">
        <div>
          <h2>导入本地 PDF</h2>
          <p className="muted">上传后会自动提取论文标题重命名，并立即切片、向量化、写入 PostgreSQL + pgvector。</p>
        </div>
        <label className={`upload-dropzone ${uploading ? "busy" : ""}`}>
          <Upload size={18} />
          <span>{uploading ? "正在上传并向量化..." : "选择 PDF 并导入向量库"}</span>
          <input
            type="file"
            accept="application/pdf,.pdf"
            disabled={uploading}
            onChange={(event) => {
              uploadPdf(event.target.files?.[0] ?? null);
              event.currentTarget.value = "";
            }}
          />
        </label>
        {uploadResult && (
          <div className="upload-result">
            <strong>{uploadResult.filename}</strong>
            <span>{uploadResult.page_count ?? 0} pages / {uploadResult.chunk_count ?? 0} chunks</span>
            {uploadResult.warning && <span>{uploadResult.warning}</span>}
          </div>
        )}
        {uploading && ingestTask && <TaskProgress task={ingestTask} />}
        {uploadError && <div className="error-banner">{uploadError}</div>}
      </div>
      <div className="section table-section">
        <table>
          <thead>
            <tr>
              <SortableTh label="文件名" sortKey="filename" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
              <SortableTh label="页数" sortKey="page_count" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
              <SortableTh label="Chunks" sortKey="chunk_count" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
              <th>多模态</th>
              <SortableTh label="状态" sortKey="status" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
              <SortableTh label="大小" sortKey="size_mb" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
              <SortableTh label="导入时间" sortKey="created_at" activeKey={sortKey} direction={sortDirection} onSort={setSort} />
            </tr>
          </thead>
          <tbody>
            {sortedDocuments.map((doc) => (
              <tr key={doc.filename} className={selected?.filename === doc.filename ? "selected" : ""} onClick={() => setSelected(doc)}>
                <td>{doc.filename}</td>
                <td>{doc.page_count ?? "-"}</td>
                <td>{doc.chunk_count}</td>
                <td>{doc.multimodal_chunk_count ?? 0}</td>
                <td><span className={`tag ${doc.status}`}>{doc.status}</span></td>
                <td>{doc.size_mb ?? "-"} MB</td>
                <td>{formatDateTime(doc.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="section detail-section">
        <h2>文献详情</h2>
        {selected ? (
          <>
            <p className="file-title">{selected.filename}</p>
            <p className="muted">{selected.path}</p>
            <div className="detail-actions">
              <button className="primary-button" onClick={revectorizeSelected} disabled={revectorizing || selected.status === "missing-file"}>
                <RefreshCw size={16} />
                {revectorizing ? "正在重新向量化..." : "重新向量化当前文献"}
              </button>
              <p className="muted">
                适合处理 pending、0 chunks 或修改过切片/embedding 参数后的单篇文献。
              </p>
              {revectorizing && (
                ingestTask ? <TaskProgress task={ingestTask} compact /> : (
                  <div className="inline-progress" aria-label="正在重新向量化">
                    <div className="inline-progress-bar" />
                  </div>
                )
              )}
            </div>
            {revectorizeMessage && <div className="success-banner">{revectorizeMessage}</div>}
            {ingestTask?.warning && !revectorizing && <div className="warning-banner">{ingestTask.warning}</div>}
            {revectorizeError && <div className="error-banner">{revectorizeError}</div>}
            <div className="chunk-list">
              {chunks.slice(0, 8).map((chunk) => (
                <article className="chunk-card" key={chunk.chunk_index}>
                  <strong>Chunk {chunk.chunk_index} · pp.{chunk.page_start}-{chunk.page_end}</strong>
                  <p>{chunk.text.slice(0, 420)}</p>
                </article>
              ))}
            </div>
          </>
        ) : <p className="muted">选择一篇文献查看详情。</p>}
      </div>
    </section>
  );
}

function sortValue(doc: DocumentRow, key: SortKey): string | number {
  if (key === "filename") return doc.filename.toLocaleLowerCase();
  if (key === "page_count") return doc.page_count ?? -1;
  if (key === "chunk_count") return doc.chunk_count;
  if (key === "status") return doc.status;
  if (key === "size_mb") return doc.size_mb ?? -1;
  if (key === "created_at") return doc.created_at ? Date.parse(doc.created_at) : 0;
  return "";
}

function formatDateTime(value: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SortableTh({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  direction: SortDirection;
  onSort: (key: SortKey) => void;
}) {
  const active = sortKey === activeKey;
  return (
    <th>
      <button className={`sort-button ${active ? "active" : ""}`} onClick={() => onSort(sortKey)}>
        <span>{label}</span>
        <span className="sort-indicator">{active ? (direction === "asc" ? "↑" : "↓") : "↕"}</span>
      </button>
    </th>
  );
}

function VectorizePage({ onRefreshHealth }: { onRefreshHealth: () => void }) {
  // 向量化页：通过后端启动 Python 入库脚本，并轮询任务日志。
  const [batchSize, setBatchSize] = useState(4);
  const [recreate, setRecreate] = useState(false);
  const [skipExisting, setSkipExisting] = useState(true);
  const [quickTest, setQuickTest] = useState(false);
  const [status, setStatus] = useState<VectorStatus>({ status: "idle", running: false, started_at: null, logs: [] });

  const loadStatus = async () => {
    const data = await api.get<VectorStatus>("/api/vectorize/status");
    setStatus(data);
    onRefreshHealth();
  };

  useEffect(() => {
    loadStatus();
    const id = window.setInterval(loadStatus, 2500);
    return () => window.clearInterval(id);
  }, []);

  const start = async () => {
    await api.post("/api/vectorize/start", { batch_size: batchSize, recreate, skip_existing: skipExisting, quick_test: quickTest });
    await loadStatus();
  };

  const stop = async () => {
    await api.post("/api/vectorize/stop");
    await loadStatus();
  };

  return (
    <section className="page-grid vector-grid">
      <div className="section">
        <h1>向量化管理</h1>
        <label>Embedding 模型</label>
        <div className="readonly-input">Qwen3-Embedding-0.6B</div>
        <label>运行设备</label>
        <div className="segmented"><button className="active">CUDA</button><button>CPU</button></div>
        <label>Batch size: {batchSize}</label>
        <input type="range" min={1} max={16} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
        <label className="check-row"><input type="checkbox" checked={skipExisting} onChange={(e) => setSkipExisting(e.target.checked)} />跳过已入库文档</label>
        <label className="check-row"><input type="checkbox" checked={recreate} onChange={(e) => setRecreate(e.target.checked)} />重建向量表</label>
        <label className="check-row"><input type="checkbox" checked={quickTest} onChange={(e) => setQuickTest(e.target.checked)} />快速测试模式</label>
        <div className="button-row">
          <button className="primary-button" onClick={start} disabled={status.running}><Play size={16} />开始</button>
          <button className="secondary-button" onClick={stop} disabled={!status.running}><Square size={16} />停止</button>
        </div>
      </div>
      <div className="section wide">
        <div className="section-title">
          <h2>任务状态</h2>
          <span className={`tag ${status.running ? "running" : "indexed"}`}>{status.status}</span>
        </div>
        <div className="log-panel">
          {status.logs.length ? status.logs.map((line, index) => <pre key={index}>{line}</pre>) : <p className="muted">暂无任务日志。</p>}
        </div>
      </div>
    </section>
  );
}

function QAPage() {
  // 智能问答页：用户手动输入问题后，可选择只检索证据或调用 DeepSeek 生成答案。
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(8);
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [agentResult, setAgentResult] = useState<AgentAskResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"search" | "ask" | "agent">("ask");

  const run = async (nextMode: "search" | "ask" | "agent") => {
    // “仅检索”只返回证据片段；“问答”会进一步调用 DeepSeek 生成答案。
    setMode(nextMode);
    setLoading(true);
    try {
      if (nextMode === "search") {
        const data = await api.post<{ hits: Hit[] }>("/api/retrieval/search", { query, top_k: topK });
        setHits(data.hits);
        setAnswer("");
        setAgentResult(null);
      } else if (nextMode === "agent") {
        const data = await api.post<AgentAskResult>("/api/agent/ask", { query, top_k: topK });
        setAnswer(data.answer);
        setHits(data.hits);
        setAgentResult(data);
      } else {
        const data = await api.post<{ answer: string; hits: Hit[] }>("/api/qa/ask", { query, top_k: topK });
        setAnswer(data.answer);
        setHits(data.hits);
        setAgentResult(null);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="page-grid qa-grid">
      <div className="section qa-controls">
        <h1>智能问答</h1>
        <label>问题</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={7}
        />
        <label>Top-K: {topK}</label>
        <input type="range" min={3} max={15} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
        <div className="button-row">
          <button className="secondary-button" onClick={() => run("search")} disabled={loading}><Search size={16} />仅检索</button>
          <button className="primary-button" onClick={() => run("ask")} disabled={loading}><Bot size={16} />问答</button>
          <button className="primary-button agent-button" onClick={() => run("agent")} disabled={loading}><Settings2 size={16} />Agentic RAG</button>
        </div>
      </div>
      <div className="section answer-section">
        <div className="section-title">
          <h2>{mode === "search" ? "检索结果" : mode === "agent" ? "Agentic RAG 回答" : "回答"}</h2>
          {loading && <span className="tag running">running</span>}
        </div>
        {agentResult && (
          <div className="agent-trace">
            <div><strong>任务类型</strong><span>{agentResult.task_type}</span></div>
            <div><strong>改写查询</strong><span>{agentResult.rewritten_query}</span></div>
            <div><strong>工作流</strong><span>{agentResult.steps.join(" -> ")}</span></div>
          </div>
        )}
        {answer ? <div className="answer">{answer}</div> : <p className="muted">输入问题后点击问答，答案会显示在这里。</p>}
      </div>
      <div className="section full">
        <h2>召回证据</h2>
        <div className="hit-grid">
          {hits.map((hit, index) => (
            <article className="hit-card" key={`${hit.filename}-${index}`}>
              <div className="hit-head">
                <strong>[{index + 1}] {hit.filename}</strong>
                <span>{hit.source_type ?? "text"} · {Number(hit.score).toFixed(4)}</span>
              </div>
              <p className="muted">pp.{hit.page_start}-{hit.page_end}</p>
              <p>{hit.text.slice(0, 680)}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function TaskProgress({ task, compact = false }: { task: IngestTaskStatus; compact?: boolean }) {
  return (
    <div className={`task-progress ${compact ? "compact" : ""}`}>
      <div className="task-progress-head">
        <strong>{task.stage}</strong>
        <span>{task.percent}%</span>
      </div>
      <div className="determinate-progress">
        <div style={{ width: `${Math.min(Math.max(task.percent, 0), 100)}%` }} />
      </div>
      <p>{task.message}</p>
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone?: "ok" | "warn" }) {
  return (
    <div className={`metric ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
