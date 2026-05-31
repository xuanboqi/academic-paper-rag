from __future__ import annotations

from typing import Any, Callable

from .nodes import make_generate_node, make_retrieve_node, make_rewrite_node, classify_task
from .state import AgentState


ChatFn = Callable[[list[dict[str, str]], float], str]
RetrieveFn = Callable[[str, int], list[dict[str, Any]]]
ContextFn = Callable[[list[dict[str, Any]]], str]


def run_agentic_rag(
    *,
    query: str,
    top_k: int,
    chat: ChatFn,
    retrieve: RetrieveFn,
    build_context: ContextFn,
    model: str,
) -> AgentState:
    """Run the Agentic RAG workflow.

    LangGraph is imported inside this function so the backend can still start with a clear
    error message if dependencies have not been installed yet.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Run `pip install -r requirements.txt` first.") from exc

    graph = StateGraph(AgentState)
    graph.add_node("classify_task", classify_task)
    graph.add_node("rewrite_query", make_rewrite_node(chat))
    graph.add_node("retrieve_context", make_retrieve_node(retrieve))
    graph.add_node("generate_answer", make_generate_node(chat, build_context))

    graph.add_edge(START, "classify_task")
    graph.add_edge("classify_task", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    app = graph.compile()
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "model": model,
        "steps": ["start"],
    }
    return app.invoke(initial_state)
