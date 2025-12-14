# agent_graph.py

import operator
import os
from typing import Annotated, List, TypedDict, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from indexer import retrieve_guidelines_by_type  # our RAG helper

load_dotenv()

ALLOWED_PAPER_TYPES = {
    "Randomized Clinical Trial",
    "Observational Study",
    "Systematic Review",
    "Meta-analysis",
    "Other",
}

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()  # e.g., low/medium/high

MAX_CHUNK_CHARS = 14000
CHUNK_OVERLAP_CHARS = 700


class AgentState(TypedDict):
    paper_content: str
    paper_type: str
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    kwargs = {
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_retries": DEFAULT_MAX_RETRIES,
        "openai_api_key": api_key,
    }

    # Only pass reasoning_effort if user set it (keeps compatibility across models).
    if DEFAULT_REASONING_EFFORT:
        kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT

    return ChatOpenAI(**kwargs)


try:
    llm = _get_llm()
except Exception as e:
    print(f"Initialization Warning (LLM): {e}")
    llm = None


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _fmt_rule_chunks(docs) -> str:
    lines: List[str] = []
    for d in docs:
        src = d.metadata.get("source_doc", "unknown.pdf")
        page = d.metadata.get("page", None)
        page_display = f"{page + 1}" if isinstance(page, int) else "?"
        prefix = f"[{src} p.{page_display}]"
        lines.append(f"{prefix}\n{d.page_content}".strip())
    return "\n\n".join(lines)


def _run_chunked_audit(
    *,
    title: str,
    rules: str,
    paper_text: str,
    per_chunk_template: str,
    combine_template: str,
) -> str:
    """
    For long manuscripts: run a quick scan per chunk, then a single combine pass.
    Improves coverage + avoids context overflow.
    """
    if llm is None:
        return f"### {title}\n*Error: LLM not initialized.*"

    chunks = _chunk_text(paper_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)

    per_chunk_prompt = ChatPromptTemplate.from_template(per_chunk_template)
    combine_prompt = ChatPromptTemplate.from_template(combine_template)

    chunk_notes: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        resp = (per_chunk_prompt | llm).invoke(
            {"rules": rules, "paper_chunk": chunk, "chunk_no": i, "chunk_total": len(chunks)}
        )
        chunk_notes.append(resp.content.strip())

    combined = (combine_prompt | llm).invoke(
        {"rules": rules, "all_chunk_notes": "\n\n---\n\n".join(chunk_notes)}
    )
    return combined.content.strip()


# --- NODES ---

def classifier_node(state: AgentState):
    """Classify manuscript type from abstract/intro."""
    if llm is None:
        return {"paper_type": "Unknown", "audit_logs": ["LLM not initialized."]}

    content_snippet = state["paper_content"][:6000]

    prompt = ChatPromptTemplate.from_template(
        """
You are an experienced statistical editor for *European Urology*.

Given the text below (mostly abstract/introduction), classify the manuscript
STRICTLY as one of the following categories:

- "Randomized Clinical Trial"
- "Observational Study"
- "Systematic Review"
- "Meta-analysis"
- "Other"

Return ONLY the category name, nothing else.

TEXT:
{text}
        """
    )

    resp = (prompt | llm).invoke({"text": content_snippet})
    category = resp.content.strip()

    if category not in ALLOWED_PAPER_TYPES:
        category = "Other"

    return {
        "paper_type": category,
        "audit_logs": [f"**Paper classified as:** {category}"],
    }


def stats_auditor_node(state: AgentState):
    """Check general statistical reporting against stats guidelines."""
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    try:
        guideline_docs = retrieve_guidelines_by_type(
            "statistics",
            "p-values, confidence intervals, precision, effect sizes, primary endpoint, sample size",
            k=6,
        )
    except Exception as e:
        return {"audit_logs": [f"### Statistical Reporting Check\nCould not load statistics guidelines: {e}"]}

    rules = _fmt_rule_chunks(guideline_docs)
    paper_text = state["paper_content"]

    per_chunk_template = """
You are a Statistical Editor for *European Urology*.

You have guideline extracts (with citations):
---------------- GUIDELINES (STATISTICS) ----------------
{rules}
---------------------------------------------------------

Review ONLY this manuscript chunk ({chunk_no}/{chunk_total}). Do NOT assume anything not in this chunk.
List potential issues as bullets; be conservative.

MANUSCRIPT CHUNK:
----------------
{paper_chunk}
----------------

Return markdown with EXACTLY:
#### Chunk {chunk_no}/{chunk_total} notes
Blocking:
- ...
Important:
- ...
Minor:
- ...
If none for a category, write "- None."
"""

    combine_template = """
You are a Statistical Editor for *European Urology*.

Below are chunk notes from scanning a manuscript. Use ONLY these notes (do not invent details).
Write a single consolidated section titled EXACTLY:

### Statistical Reporting Check

Within it, list:
- 1–3 **blocking issues** (if any) that would prevent acceptance.
- 2–4 **important but fixable** issues (if any).
- Any **minor suggestions**.

If there are essentially no problems, say explicitly that statistical reporting appears compliant.

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

    section = _run_chunked_audit(
        title="Statistical Reporting Check",
        rules=rules,
        paper_text=paper_text,
        per_chunk_template=per_chunk_template,
        combine_template=combine_template,
    )
    return {"audit_logs": [section]}


def figtab_auditor_node(state: AgentState):
    """Check figures and tables against figure/table guidelines."""
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    try:
        guideline_docs = retrieve_guidelines_by_type(
            "figures_tables",
            "figures tables graphs labels legends units precision Kaplan-Meier number at risk forest plot",
            k=6,
        )
    except Exception as e:
        return {"audit_logs": [f"### Figures and Tables Check\nCould not load figures/tables guidelines: {e}"]}

    rules = _fmt_rule_chunks(guideline_docs)
    paper_text = state["paper_content"]

    per_chunk_template = """
You are a Statistical Editor for *European Urology*.

You have guideline extracts (with citations):
---------------- FIGURES/TABLES GUIDELINES ----------------
{rules}
----------------------------------------------------------

Review ONLY this manuscript chunk ({chunk_no}/{chunk_total}).
You cannot see images; only infer from text mentions (e.g., "Figure 2", "Kaplan–Meier", "Table 1").

MANUSCRIPT CHUNK:
----------------
{paper_chunk}
----------------

Return markdown with EXACTLY:
#### Chunk {chunk_no}/{chunk_total} notes
Blocking:
- ...
Important:
- ...
Minor:
- ...
If none for a category, write "- None."
"""

    combine_template = """
You are a Statistical Editor for *European Urology*.

Use ONLY the chunk notes below.
Write a section titled EXACTLY:

### Figures and Tables Check

Summarize:
- Any major violations of the guidelines (blocking if serious).
- Other important issues.
- Minor suggestions (clarity/aesthetics/small formatting).

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

    section = _run_chunked_audit(
        title="Figures and Tables Check",
        rules=rules,
        paper_text=paper_text,
        per_chunk_template=per_chunk_template,
        combine_template=combine_template,
    )
    return {"audit_logs": [section]}


def type_specific_auditor_node(state: AgentState):
    """Type-specific checks for observational causality or SR/MA."""
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    paper_type = (state.get("paper_type") or "").lower()
    paper_text = state["paper_content"]

    if "observational" in paper_type:
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "causality",
                "causal language confounding causal pathways introduction methods discussion associated vs causes",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Causality / Observational Study Check\nCould not load causality guidelines: {e}"]}

        rules = _fmt_rule_chunks(guideline_docs)

        per_chunk_template = """
You are a Statistical Editor for *European Urology*.

The manuscript is an Observational Study.
You have guideline extracts (with citations):
---------------- CAUSALITY GUIDELINES ----------------
{rules}
------------------------------------------------------

Review ONLY this manuscript chunk ({chunk_no}/{chunk_total}).
Focus on causal intent, confounding discussion, and causal vs associational language.

MANUSCRIPT CHUNK:
----------------
{paper_chunk}
----------------

Return markdown with EXACTLY:
#### Chunk {chunk_no}/{chunk_total} notes
Blocking:
- ...
Important:
- ...
Minor:
- ...
If none for a category, write "- None."
"""

        combine_template = """
You are a Statistical Editor for *European Urology*.

Use ONLY the chunk notes below.
Write a section titled EXACTLY:

### Causality / Observational Study Check

Under that heading, summarize:
- Causal clarity in aims and introduction.
- Adequacy of confounding control and discussion.
- Appropriateness of causal vs associational language.
- Blocking issues vs smaller wording/interpretation fixes.

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

        section = _run_chunked_audit(
            title="Causality / Observational Study Check",
            rules=rules,
            paper_text=paper_text,
            per_chunk_template=per_chunk_template,
            combine_template=combine_template,
        )
        return {"audit_logs": [section]}

    if "systematic review" in paper_type or "meta-analysis" in paper_type or "meta analysis" in paper_type:
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "systematic_meta",
                "PRISMA MOOSE heterogeneity protocol reproducible methods risk of bias SUCRA rankings",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Systematic Review / Meta-analysis Check\nCould not load SR/MA guidelines: {e}"]}

        rules = _fmt_rule_chunks(guideline_docs)

        per_chunk_template = """
You are a Statistical Editor for *European Urology*.

The manuscript is a {ptype}.
You have guideline extracts (with citations):
---------------- SR/MA GUIDELINES ----------------
{rules}
--------------------------------------------------

Review ONLY this manuscript chunk ({chunk_no}/{chunk_total}).
Focus on PRISMA/MOOSE reporting, reproducibility, heterogeneity, and interpretation.

MANUSCRIPT CHUNK:
----------------
{paper_chunk}
----------------

Return markdown with EXACTLY:
#### Chunk {chunk_no}/{chunk_total} notes
Blocking:
- ...
Important:
- .
