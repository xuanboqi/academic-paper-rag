from __future__ import annotations


REWRITE_SYSTEM_PROMPT = (
    "你是学术文献检索专家。你的任务是把用户问题改写成适合检索英文论文片段的查询。"
    "保留模型名、数据集名、指标名、缩写和关键术语。只输出改写后的查询，不要解释。"
)

ANSWER_SYSTEM_PROMPT = (
    "你是严谨的学术论文分析智能体。只能基于给定证据回答。"
    "如果证据不足，要明确说明不足。回答用中文，关键结论后标注来源编号。"
)


def build_rewrite_prompt(query: str, task_type: str) -> str:
    return (
        f"任务类型：{task_type}\n"
        f"用户问题：{query}\n\n"
        "请改写成适合学术论文 RAG 检索的查询。"
    )


def build_answer_prompt(query: str, rewritten_query: str, task_type: str, context: str) -> str:
    return (
        f"原始问题：{query}\n"
        f"检索查询：{rewritten_query}\n"
        f"任务类型：{task_type}\n\n"
        f"证据片段：\n{context}\n\n"
        "请给出结构化回答。若是文献综述或对比分析，请主动组织为小标题或表格。"
    )
