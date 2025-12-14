# pdf_reader.py
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@dataclass
class VisualPageSummary:
    page_number: int  # 1-based
    summary_md: str


def _to_data_url_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _render_page_png(doc: fitz.Document, page_index: int, zoom: float = 2.0) -> bytes:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _score_page_for_visuals(doc: fitz.Document, page_index: int) -> int:
    """
    Heuristic scoring:
    - More embedded images => higher score
    - Mentions of Figure/Table in extracted text => higher score
    """
    page = doc.load_page(page_index)
    images = page.get_images(full=True)
    img_score = len(images) * 10

    text = (page.get_text("text") or "").lower()
    kw_score = 0
    if "figure" in text:
        kw_score += 6
    if "table" in text:
        kw_score += 6
    if "kaplan" in text or "cumulative incidence" in text:
        kw_score += 4

    # If text is very short but has images, likely a figure page
    short_text_bonus = 2 if len(text.strip()) < 200 and len(images) > 0 else 0

    return img_score + kw_score + short_text_bonus


def pick_visual_pages(pdf_path: str, max_pages: int = 8) -> List[int]:
    """
    Returns 0-based page indexes to analyze with vision.
    """
    doc = fitz.open(pdf_path)
    scored: List[Tuple[int, int]] = []
    for i in range(doc.page_count):
        score = _score_page_for_visuals(doc, i)
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    picked = [i for _, i in scored[:max_pages]]
    doc.close()
    return sorted(set(picked))


def summarize_pdf_visuals(
    pdf_path: str,
    llm_vision: ChatOpenAI,
    max_pages: int = 8,
    zoom: float = 2.0,
) -> Tuple[str, List[VisualPageSummary]]:
    """
    Renders selected pages as images and asks the model to summarize ONLY what is visible.
    Returns:
      - markdown block with all summaries
      - list of per-page summaries
    """
    doc = fitz.open(pdf_path)
    page_indexes = pick_visual_pages(pdf_path, max_pages=max_pages)

    summaries: List[VisualPageSummary] = []

    system_prompt = (
        "You are reading a single PDF page image from a clinical manuscript.\n"
        "Your job:\n"
        "- Identify any FIGURES/TABLES/plots on the page.\n"
        "- Extract what is visibly present (axes labels, units, legends, CI bands, numbers-at-risk tables, p-values shown).\n"
        "- If something is not readable/visible, say 'Not readable' / 'Not visible'.\n"
        "- Do NOT guess.\n"
        "- Return concise markdown bullets.\n"
    )

    for idx in page_indexes:
        try:
            png_bytes = _render_page_png(doc, idx, zoom=zoom)
        except Exception as e:
            summaries.append(
                VisualPageSummary(page_number=idx + 1, summary_md=f"- Rendering failed: {e}")
            )
            continue

        data_url = _to_data_url_png(png_bytes)

        msg = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        try:
            resp = llm_vision.invoke([msg])
            text = (resp.content or "").strip()
        except Exception as e:
            text = f"- Vision model call failed: {e}"

        summaries.append(VisualPageSummary(page_number=idx + 1, summary_md=text))

    doc.close()

    combined_md = "### Visual extracts (from PDF page images)\n\n"
    if not summaries:
        combined_md += "_No visual pages detected or analyzed._\n"
        return combined_md, summaries

    for s in summaries:
        combined_md += f"**Page {s.page_number}**\n\n{s.summary_md}\n\n---\n\n"

    return combined_md, summaries
