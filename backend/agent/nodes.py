from __future__ import annotations

from typing import Any, Callable

from .prompts import ANSWER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT, build_answer_prompt, build_rewrite_prompt
from .state import AgentState, TaskType


ChatFn = Callable[[list[dict[str, str]], float], str]
RetrieveFn = Callable[[str, int], list[dict[str, Any]]]
ContextFn = Callable[[list[dict[str, Any]]], str]


def classify_task(state: AgentState) -> AgentState:
    """Route common academic tasks without spending an LLM call."""
    query = state["query"].lower()
    task_type: TaskType = "qa"
    if any(keyword in query for keyword in ["综述", "review", "survey", "研究现状"]):
        task_type = "literature_review"
    elif any(keyword in query for keyword in ["对比", "比较", "compare", "difference", "区别"]):
        task_type = "paper_comparison"
    elif any(keyword in query for keyword in ["图", "表", "公式", "figure", "table", "equation", "formula"]):
        task_type = "figure_or_formula"
    steps = [*state.get("steps", []), f"classify_task: {task_type}"]
    return {"task_type": task_type, "steps": steps}


def make_rewrite_node(chat: ChatFn) -> Callable[[AgentState], AgentState]:
    def rewrite_query(state: AgentState) -> AgentState:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_rewrite_prompt(state["query"], state.get("task_type", "qa"))},
        ]
        rewritten = chat(messages, 0.1).strip()
        if not rewritten:
            rewritten = state["query"]
        steps = [*state.get("steps", []), "rewrite_query: completed"]
        return {"rewritten_query": rewritten, "steps": steps}

    return rewrite_query


def make_retrieve_node(retrieve: RetrieveFn) -> Callable[[AgentState], AgentState]:
    def retrieve_context(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        hits = retrieve(query, state.get("top_k", 8))
        steps = [*state.get("steps", []), f"retrieve_context: {len(hits)} hits"]
        return {"hits": hits, "steps": steps}

    return retrieve_context


def make_generate_node(chat: ChatFn, build_context: ContextFn) -> Callable[[AgentState], AgentState]:
    def generate_answer(state: AgentState) -> AgentState:
        context = build_context(state.get("hits", []))
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_answer_prompt(
                    query=state["query"],
                    rewritten_query=state.get("rewritten_query") or state["query"],
                    task_type=state.get("task_type", "qa"),
                    context=context,
                ),
            },
        ]
        answer = chat(messages, 0.2)
        steps = [*state.get("steps", []), "generate_answer: completed"]
        return {"answer": answer, "steps": steps}

    return generate_answer
