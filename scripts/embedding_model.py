from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "./models/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question, retrieve relevant academic paper passages that answer the question."
)
DEFAULT_EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值，方便 .env 中用 true/false 控制开关。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


@dataclass
class LocalEmbeddingModel:
    """本地 embedding 模型封装。

    这里把 SentenceTransformer 的加载、文档向量化、问题向量化统一封装起来。
    文档和问题必须使用同一个 embedding 模型，否则 pgvector 中的相似度没有意义。
    """

    model_name_or_path: str
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    trust_remote_code: bool = True
    device: str = DEFAULT_EMBEDDING_DEVICE
    half_precision: bool = True
    max_seq_length: Optional[int] = None

    def __post_init__(self) -> None:
        # Qwen3-Embedding 放在本地 models/ 下，device 由 .env 控制，优先使用 cuda。
        model_kwargs = {}
        if self.half_precision and self.device.startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self.model = SentenceTransformer(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
            model_kwargs=model_kwargs or None,
        )
        if self.max_seq_length is not None:
            self.model.max_seq_length = self.max_seq_length
        if self.half_precision and self.device.startswith("cuda"):
            self.model.half()
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for {self.model_name_or_path}")
        self.dimension = dimension

    def encode_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        # 文档向量会入库保存，normalize 后可以直接用 cosine 距离做近邻检索。
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        # 查询向量不入库，只用于和 literature_chunks.embedding 做相似度排序。
        text = self.format_query(query)
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def format_query(self, query: str) -> str:
        # Qwen3-Embedding 支持 instruction-style query，能让检索更贴近“找答案片段”这个任务。
        query = query.strip()
        if not self.query_instruction:
            return query
        return f"Instruct: {self.query_instruction}\nQuery: {query}"


def load_embedding_model(model_name_or_path: str | None = None) -> LocalEmbeddingModel:
    """按 .env 加载 embedding 模型，供入库、检索、问答、Web API 复用。"""
    return LocalEmbeddingModel(
        model_name_or_path=model_name_or_path or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        query_instruction=os.getenv("EMBEDDING_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
        trust_remote_code=env_bool("EMBEDDING_TRUST_REMOTE_CODE", True),
        device=os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
        half_precision=env_bool("EMBEDDING_HALF_PRECISION", True),
        max_seq_length=env_int("EMBEDDING_MAX_SEQ_LENGTH"),
    )
