# agent_graph.py
import operator
import os
from functools import lru_cache
from typing import Annotated, List, TypedDict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from indexer import retrieve_guidelines_by_type

load_dotenv()


class AgentState(TypedDict):
    paper_content: str
    paper_visuals: str
    paper_type: str
    kb_ready: bool
    kb_details: str
    visuals_used: bool
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing env var: {name}")
    return val


def _get_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        if not resp:
            return ""
        return _get_text(resp[-1])
    if hasattr(resp, "content"):
        return (resp.content or "").strip()
    if isinstance(resp, dict):
        for k in ("content", "output_text", "text"):
            if k in resp and isinstance(resp[k], str):
                return resp[k].strip()
    return str(resp).strip()


def _coerce_category(raw: str) -> str:
    allowed = {
        "Randomized Clinical Trial",
        "Observational Study",
        "Systematic Review",
        "Meta-analysis",
        "Other",
    }
    s = (raw or "").strip().strip('"').strip("'")
    lower = s.lower()
    if "random" in lower and ("trial" in lower or "rct" in lower):
        return "Randomized Clinical Trial"
    if "observ" in lower or "cohort" in lower or "case" in lower:
        return "Observational Study"
    if "systematic" in lower:
        return "Systematic Review"
    if "meta" in lower:
        return "Meta-analysis"
    if s in allowed:
        return s
    return "Other"


def _make_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    api_key = _require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key, "model": model_name, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=8)
def get_app_graph(model_name: str, temperature: float = 0.0):
    llm = _make_llm(model_name=model_name, temperature=temperature)

    def kb_guard_node(state: AgentState):
        if not state.get("kb_ready", False):
            return {
                "paper_type": "Other",
                "audit_logs": [
                    "### Knowledge base check\n"
                    f"Vector DB ready: **False**\n\n"
                    f"Details: {state.get('kb_details','(no details)')}\n\n"
                    "⚠️ Blocking run: guidelines are not available in the vector DB.\n"
                    "Rebuild/validate the KB from the sidebar and rerun."
                ],
            }
        return {
            "audit_logs": [
                "### Knowledge base check\n"
                f"Vector DB ready: **True**\n\n"
                f"Details: {state.get('kb_details','')}"
            ]
        }

    def classifier_node(state: AgentState):
        content_snippet = state["paper_content"][:4000]

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
            """.strip()
        )

        resp = (prompt | llm).invoke({"text": content_snippet})
        category = _coerce_category(_get_text(resp))

        return {
            "paper_type": category,
            "audit_logs": [f"**Paper classified as:** {category}"],
        }

    def stats_auditor_node(state: AgentState):
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "statistics",
                "p-values, confidence intervals, precision, effect sizes, primary endpoint, sample size",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Statistical Reporting Check\nCould not load statistics guidelines: {e}"]}

        if not guideline_docs:
            return {"audit_logs": ["### Statistical Reporting Check\nNo statistics guideline chunks retrieved (KB issue)."]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Statistics for Clinical Research in Urology**.
Here are relevant extracts:

---------------- GUIDELINES (STATISTICS) ----------------
{rules}
---------------------------------------------------------

MANUSCRIPT TEXT:
----------------
{paper}
----------------

VISUAL EXTRACTS (FIGURES/TABLES FROM PDF IMAGES):
----------------
{visuals}
----------------

Check the manuscript for statistical reporting issues using the guidelines.
Only flag figure/table problems if supported by VISUAL EXTRACTS or the manuscript text.
Do NOT guess.

Write:

### Statistical Reporting Check

- 1–3 **blocking issues**
- 2–4 **important but fixable** issues
- Any **minor suggestions**
            """.strip()
        )

        resp = (prompt | llm).invoke(
            {"rules": rules, "paper": state["paper_content"], "visuals": state.get("paper_visuals", "")}
        )
        return {"audit_logs": [_get_text(resp)]}

    def figtab_auditor_node(state: AgentState):
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "figures_tables",
                "figures tables graphs dos and don'ts precision labels legends Kaplan-Meier numbers-at-risk CI bands",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Figures and Tables Check\nCould not load figures/tables guidelines: {e}"]}

        if not guideline_docs:
            return {"audit_logs": ["### Figures and Tables Check\nNo figures/tables guideline chunks retrieved (KB issue)."]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Figures and Tables**.

---------------- FIGURES/TABLES GUIDELINES ----------------
{rules}
-----------------------------------------------------------

Below are extracted summaries from the ACTUAL PDF page images.
Treat them as evidence. If a detail is not in these extracts, you must NOT claim it.

VISUAL EXTRACTS (FROM PDF IMAGES):
----------------
{visuals}
----------------

MANUSCRIPT TEXT (for captions/cross-reference only):
----------------
{paper}
----------------

Write:

### Figures and Tables Check

- Any **blocking** violations (only if supported by the visual extracts)
- Other important issues
- Minor suggestions
            """.strip()
        )

        resp = (prompt | llm).invoke(
            {"rules": rules, "visuals": state.get("paper_visuals", ""), "paper": state["paper_content"]}
        )
        return {"audit_logs": [_get_text(resp)]}

    def type_specific_auditor_node(state: AgentState):
        paper_type = (state.get("paper_type") or "").lower()
        if "observational" not in paper_type:
            return {
                "audit_logs": [
                    "### Type-Specific Check\nStudy type does not trigger additional causality or SR/MA checks."
                ]
            }

        try:
            guideline_docs = retrieve_guidelines_by_type(
                "causality",
                "causal language, confounding, causal pathways, introduction, methods, discussion",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Causality / Observational Study Check\nCould not load causality guidelines: {e}"]}

        if not guideline_docs:
            return {"audit_logs": ["### Causality / Observational Study Check\nNo causality guideline chunks retrieved (KB issue)."]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

The manuscript is an **Observational Study**.

---------------- CAUSALITY GUIDELINES ----------------
{rules}
------------------------------------------------------

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write:

### Causality / Observational Study Check

- Are aims framed as causal vs prognostic/associational?
- Confounding/detection bias clarity
- Wording fixes and any blocking issues
            """.strip()
        )

        resp = (prompt | llm).invoke({"rules": rules, "paper": state["paper_content"]})
        return {"audit_logs": [_get_text(resp)]}

    def reporter_node(state: AgentState):
        logs_text = "\n\n---\n\n".join(state.get("audit_logs", []))

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

Below are raw notes from automated checkers:

---------------- NOTES ----------------
{logs}
-------------------------------------

Draft a concise report in markdown with this structure:

🇪🇺 European Urology Statistical Report
Detected Type: {paper_type}

Overall summary
- 2–4 bullets

Blocking issues
- Bullets, or "None."

Important but fixable issues
- Bullets, or "None."

Minor issues / suggestions
- Bullets, or "None."

Provisional recommendation
- One line (e.g., Major revisions required.)

Then append:

Knowledge base status
- Vector DB ready: {kb_ready}
- Details: {kb_details}

Visual extraction status
- Figures/tables analyzed from PDF images: {visuals_used}

Important: do not invent manuscript specifics not supported by the notes.
            """.strip()
        )

        resp = (prompt | llm).invoke(
            {
                "paper_type": state.get("paper_type", "Unknown"),
                "logs": logs_text,
                "kb_ready": state.get("kb_ready", False),
                "kb_details": state.get("kb_details", ""),
                "visuals_used": state.get("visuals_used", False),
            }
        )
        return {"final_report": _get_text(resp)}

    workflow = StateGraph(AgentState)
    workflow.add_node("guard_kb", kb_guard_node)
    workflow.add_node("classify", classifier_node)
    workflow.add_node("check_stats", stats_auditor_node)
    workflow.add_node("check_type_specific", type_specific_auditor_node)
    workflow.add_node("check_figtab", figtab_auditor_node)
    workflow.add_node("report", reporter_node)

    workflow.set_entry_point("guard_kb")

    workflow.add_conditional_edges(
        "guard_kb",
        lambda s: "classify" if s.get("kb_ready", False) else "report",
        {"classify": "classify", "report": "report"},
    )

    workflow.add_edge("classify", "check_stats")
    workflow.add_edge("check_stats", "check_type_specific")
    workflow.add_edge("check_type_specific", "check_figtab")
    workflow.add_edge("check_figtab", "report")
    workflow.add_edge("report", END)

    return workflow.compile()
