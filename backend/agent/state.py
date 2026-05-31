from __future__ import annotations

from typing import Any, Literal, TypedDict


TaskType = Literal["qa", "literature_review", "paper_comparison", "figure_or_formula"]


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    query: str
    top_k: int
    task_type: TaskType
    rewritten_query: str
    hits: list[dict[str, Any]]
    answer: str
    model: str
    steps: list[str]
