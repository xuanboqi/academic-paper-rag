from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


CANDIDATE_PATTERN = re.compile(
    r"\b(fig\.?|figure|table|tab\.?|equation|eq\.?|formula)\b|图|表|公式",
    re.IGNORECASE,
)

HIGH_VALUE_PATTERN = re.compile(
    r"\b(fig\.?\s*\d+|figure\s*\d+|table\s*\d+|tab\.?\s*\d+|equation\s*\d+|eq\.?\s*\(?\d+|"
    r"architecture|framework|overview|pipeline|comparison|ablation|experiment|result|mAP|accuracy|"
    r"precision|recall|f1|loss|module|network)\b|图\s*\d+|表\s*\d+|公式\s*\d+|结构|框架|流程|对比|消融|实验|结果",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path
    text: str
    is_candidate: bool
    score: int


def score_page(text: str, page_index: int) -> int:
    """Rank pages likely to contain useful figures, tables, formulas, or experiments."""
    normalized = text or ""
    score = 0
    score += len(CANDIDATE_PATTERN.findall(normalized)) * 2
    score += len(HIGH_VALUE_PATTERN.findall(normalized)) * 4
    # Early pages often contain architecture figures and contribution summaries.
    if page_index < 4:
        score += 3
    # Reference-only pages frequently contain many "Fig." mentions but little visual content.
    if "references" in normalized.lower()[:1200]:
        score -= 12
    if len(normalized) > 6000:
        score -= 2
    return score


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 120,
    max_pages: int | None = None,
    only_candidates: bool = True,
    max_candidate_pages: int | None = 3,
) -> list[RenderedPage]:
    """Render selected PDF pages to PNG images for local vision-language parsing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedPage] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        limit = min(page_count, max_pages) if max_pages else page_count
        candidates: list[tuple[int, int, str, bool]] = []
        for page_index in range(limit):
            page = doc[page_index]
            text = page.get_text("text") or ""
            is_candidate = bool(CANDIDATE_PATTERN.search(text))
            if only_candidates and not is_candidate:
                continue
            score = score_page(text, page_index)
            if only_candidates and score <= 0:
                continue
            candidates.append((score, page_index, text, is_candidate))

        if max_candidate_pages and max_candidate_pages > 0:
            candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_candidate_pages]
        candidates = sorted(candidates, key=lambda item: item[1])

        for score, page_index, text, is_candidate in candidates:
            page = doc[page_index]
            image_path = output_dir / f"page_{page_index + 1:04d}.png"
            if not image_path.exists():
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(image_path)
            rendered.append(
                RenderedPage(
                    page_number=page_index + 1,
                    image_path=image_path,
                    text=text,
                    is_candidate=is_candidate,
                    score=score,
                )
            )
    return rendered
