# agent_graph.py

import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from indexer import retrieve_guidelines_by_type

load_dotenv()

# --- QUALITY / SAFETY DEFAULTS ---
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # set OPENAI_MODEL to your best available
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# Chunking avoids context overflow on long manuscripts
MAX_CHUNK_CHARS = int(os.getenv("PAPER_CHUNK_CHARS", "14000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("PAPER_CHUNK_OVERLAP_CHARS", "700"))

ALLOWED_PAPER_TYPES = {
    "Randomized Clinical Trial",
    "Observational Study",
    "Systematic Review",
    "Meta-analysis",
    "Other",
}


# --- 1. STATE ---

class AgentState(TypedDict):
    paper_content: str
    paper_type: str
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


# --- Helpers: robust text extraction (fixes list/str/content issues) ---

def _content_to_text(content) -> str:
    """
    LangChain responses can be:
      - AIMessage (has .content)
      - str
      - list of content blocks (dicts) or strings
    Convert everything to a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # common: {"type": "text", "text": "..."}
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def _as_text(resp) -> str:
    """Accept AIMessage OR raw string/list and return plain text."""
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return _content_to_text(getattr(resp, "content"))
    return _content_to_text(resp)


def _normalize_paper_type(raw: str) -> str:
    s = (raw or "").strip().strip('"').strip("'")
    # Take first line if model outputs extra junk
    s = s.splitlines()[0].strip()
    # Normalize common variants
    low = s.lower()
    if "random" in low and "trial" in low:
        return "Randomized Clinical Trial"
    if "observ" in low:
        return "Observational Study"
    if "systematic" in low and "review" in low:
        return "Systematic Review"
    if "meta" in low:
        return "Meta-analysis"
    if s in ALLOWED_PAPER_TYPES:
        return s
    return "Other"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _fmt_rule_chunks(docs) -> str:
    """
    Render guideline chunks with a simple citation prefix like:
    [Guideline.pdf p.3]
    """
    lines: List[str] = []
    for d in docs:
        src = d.metadata.get("source_doc", "unknown.pdf")
        page = d.metadata.get("page", None)
        page_display = f"{page + 1}" if isinstance(page, int) else "?"
        prefix = f"[{src} p.{page_display}]"
        lines.append(f"{prefix}\n{d.page_content}".strip())
    return "\n\n".join(lines)


# --- 2. LLM SETUP ---

def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    # Some versions of langchain_openai accept different init args.
    # We try the common one; if it errors, fallback to minimal.
    try:
        return ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_retries=DEFAULT_MAX_RETRIES,
            openai_api_key=api_key,
        )
    except TypeError:
        return ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            openai_api_key=api_key,
        )


try:
    llm = _get_llm()
except Exception as e:
    print(f"Initialization Warning (LLM): {e}")
    llm = None


def _run_chunked_audit(
    *,
    rules: str,
    paper_text: str,
    per_chunk_template: str,
    combine_template: str,
) -> str:
    """
    Scan manuscript in chunks -> create chunk notes -> combine into a single section.
    More coverage + avoids context overflow.
    """
    if llm is None:
        return "*Error: LLM not initialized.*"

    chunks = _chunk_text(paper_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)

    per_chunk_prompt = ChatPromptTemplate.from_template(per_chunk_template)
    combine_prompt = ChatPromptTemplate.from_template(combine_template)

    chunk_notes: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        resp = (per_chunk_prompt | llm).invoke(
            {"rules": rules, "paper_chunk": chunk, "chunk_no": i, "chunk_total": len(chunks)}
        )
        chunk_notes.append(_as_text(resp).strip())

    combined = (combine_prompt | llm).invoke(
        {"all_chunk_notes": "\n\n---\n\n".join(chunk_notes)}
    )
    return _as_text(combined).strip()


# --- 3. NODES ---

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
    category = _normalize_paper_type(_as_text(resp))

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
            "p-values confidence intervals precision effect sizes primary endpoint sample size multiplicity",
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

Review ONLY this manuscript chunk ({chunk_no}/{chunk_total}).
Do NOT assume anything not in this chunk. Be conservative.

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

Using ONLY the chunk notes below (do not invent details),
write a single consolidated section titled EXACTLY:

### Statistical Reporting Check

Within it, list:
- 1–3 **blocking issues** (if any).
- 2–4 **important but fixable** issues (if any).
- **minor suggestions**.

If essentially no problems, say explicitly that statistical reporting appears compliant.

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

    section = _run_chunked_audit(
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
            "figures tables labels legends units precision Kaplan-Meier number at risk forest plot",
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
You cannot see images; infer only from text mentions (e.g., "Figure 2", "Kaplan–Meier", "Table 1").

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

Using ONLY the chunk notes below,
write a section titled EXACTLY:

### Figures and Tables Check

Summarize:
- blocking issues (if any),
- important issues,
- minor suggestions (clarity/aesthetics/small formatting).

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

    section = _run_chunked_audit(
        rules=rules,
        paper_text=paper_text,
        per_chunk_template=per_chunk_template,
        combine_template=combine_template,
    )
    return {"audit_logs": [section]}


def type_specific_auditor_node(state: AgentState):
    """Causality checks for observational; SR/MA checks for reviews/meta-analyses."""
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

This is an Observational Study.
Guideline extracts (with citations):
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

Using ONLY the chunk notes below,
write a section titled EXACTLY:

### Causality / Observational Study Check

Summarize:
- causal clarity in aims/introduction,
- confounding control description,
- appropriateness of causal vs associational language,
- blocking vs fixable issues.

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

        section = _run_chunked_audit(
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
                "PRISMA MOOSE protocol reproducible methods risk of bias heterogeneity SUCRA",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Systematic Review / Meta-analysis Check\nCould not load SR/MA guidelines: {e}"]}

        rules = _fmt_rule_chunks(guideline_docs)

        per_chunk_template = """
You are a Statistical Editor for *European Urology*.

Guideline extracts (with citations):
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
- ...
Minor:
- ...
If none for a category, write "- None."
"""

        combine_template = """
You are a Statistical Editor for *European Urology*.

Using ONLY the chunk notes below,
write a section titled EXACTLY:

### Systematic Review / Meta-analysis Check

Summarize:
- blocking issues,
- important but fixable issues,
- minor suggestions.

CHUNK NOTES:
----------------
{all_chunk_notes}
----------------
"""

        section = _run_chunked_audit(
            rules=rules,
            paper_text=paper_text,
            per_chunk_template=per_chunk_template,
            combine_template=combine_template,
        )
        return {"audit_logs": [section]}

    return {
        "audit_logs": [
            "### Type-Specific Check\nStudy type does not trigger additional causality or SR/MA checks "
            "beyond general statistics and figures/tables."
        ]
    }


def reporter_node(state: AgentState):
    """Combine logs into a single editorial-style report."""
    logs_text = "\n\n---\n\n".join(state.get("audit_logs", []))

    if llm is None:
        fallback = (
            "🇪🇺 European Urology Statistical Report\n"
            f"Detected Type: {state.get('paper_type', 'Unknown')}\n\n"
            "LLM not initialized – showing raw logs:\n\n"
            f"{logs_text}"
        )
        return {"final_report": fallback}

    prompt = ChatPromptTemplate.from_template(
        """
You are a Statistical Editor for *European Urology*.

A manuscript has been analyzed by several automated checkers.
The manuscript type (as classified) is:

> {paper_type}

Below are their raw notes (with some overlap):

---------------- ANALYSIS NOTES ----------------
{logs}
------------------------------------------------

Using ONLY these notes (do not invent details you do not see),
draft a concise report in markdown with EXACTLY the following structure:

🇪🇺 European Urology Statistical Report
Detected Type: {paper_type}

Overall summary
- 2–4 bullet points describing the overall quality & main themes.

Blocking issues
- Bullet list of issues that **must** be fixed before acceptance.
- If none, write: "None."

Important but fixable issues
- Bullet list of non-fatal but important issues.
- If none, write: "None."

Minor issues / suggestions
- Bullet list of minor style/clarity/presentation suggestions.
- If none, write: "None."

Provisional recommendation
- One line like:
  "Acceptable with minor revisions", or
  "Major revisions required", or
  "Not acceptable in current form."
        """
    )

    resp = (prompt | llm).invoke({"paper_type": state.get("paper_type", "Unknown"), "logs": logs_text})
    return {"final_report": _as_text(resp).strip()}


# --- 4. GRAPH ---

workflow = StateGraph(AgentState)

workflow.add_node("classify", classifier_node)
workflow.add_node("check_stats", stats_auditor_node)
workflow.add_node("check_type_specific", type_specific_auditor_node)
workflow.add_node("check_figtab", figtab_auditor_node)
workflow.add_node("report", reporter_node)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "check_stats")
workflow.add_edge("check_stats", "check_type_specific")
workflow.add_edge("check_type_specific", "check_figtab")
workflow.add_edge("check_figtab", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()
