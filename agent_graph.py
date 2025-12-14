# agent_graph.py

import operator
import os
from functools import lru_cache
from typing import Annotated, List, TypedDict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from langchain_core.messages import HumanMessage, SystemMessage

from indexer import retrieve_guidelines_by_type  # RAG helper

load_dotenv()


class AgentState(TypedDict):
    paper_content: str
    paper_type: str
    paper_images: List[str]  # base64 data URLs ("data:image/jpeg;base64,...") for PDF pages
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing env var: {name}")
    return val


def _get_text(resp: Any) -> str:
    """
    LangChain return types can vary by version:
    - AIMessage (has .content)
    - str
    - list of messages
    - dict-like
    This normalizes to a string safely.
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        if not resp:
            return ""
        return _get_text(resp[-1])
    # AIMessage / BaseMessage
    if hasattr(resp, "content"):
        return (resp.content or "").strip()
    # dict-ish
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
    if "observ" in lower or "cohort" in lower or "case" in lower or "retrospective" in lower:
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
    """
    Returns a compiled LangGraph app for the selected model.
    Cached so switching models doesn't recompile every time.
    """
    llm = _make_llm(model_name=model_name, temperature=temperature)

    def classifier_node(state: AgentState):
        """Classify manuscript type from abstract/intro."""
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
        """Check general statistical reporting against stats guidelines."""
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "statistics",
                "p-values, confidence intervals, precision, effect sizes, primary endpoint, sample size",
                k=6,
            )
        except Exception as e:
            return {"audit_logs": [f"### Statistical Reporting Check\nCould not load statistics guidelines: {e}"]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)
        paper_snip = state["paper_content"]

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Statistics for Clinical Research in Urology**.
Here are relevant extracts:

---------------- GUIDELINES (STATISTICS) ----------------
{rules}
---------------------------------------------------------

Now check the following manuscript text for **statistical reporting** issues,
using those guidelines as your reference. Focus on:

- Whether effect estimates have confidence intervals and appropriate precision.
- Whether p-values are used and interpreted according to the guidelines.
- Whether primary/secondary endpoints and analysis methods are clearly reported.
- Any obviously misleading or non-guideline-concordant statistical reporting.

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a short section titled:

### Statistical Reporting Check

Within it, list:
- 1–3 **blocking issues** (if any) that would prevent acceptance.
- 2–4 **important but fixable** issues (if any).
- Any **minor suggestions**.

If there are essentially no problems, say explicitly that statistical reporting appears compliant.
            """.strip()
        )

        resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
        return {"audit_logs": [_get_text(resp)]}

    def figtab_auditor_node(state: AgentState):
        """
        Check figures and tables against figure/table guidelines, using:
        - PDF page images (vision) when available
        - Fallback to text-only if vision call fails
        """
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "figures_tables",
                "figures tables graphs labels legends Kaplan-Meier numbers at risk confidence interval shading",
                k=4,  # keep shorter because we also send images
            )
        except Exception as e:
            return {"audit_logs": [f"### Figures and Tables Check\nCould not load figures/tables guidelines: {e}"]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        images = (state.get("paper_images") or [])[:8]
        if not images:
            # No images were provided, do a transparent text-only check.
            paper_snip = state["paper_content"]
            prompt = ChatPromptTemplate.from_template(
                """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Figures and Tables for Clinical Research in Urology**.

Relevant extracts:
---------------- FIGURES/TABLES GUIDELINES ----------------
{rules}
----------------------------------------------------------

You do NOT have access to the figure images. Review the manuscript text
for figure/table-related issues ONLY where the text explicitly supports it.
If something cannot be verified from the text, write "Manual check needed".

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a short section titled:

### Figures and Tables Check

Summarize:
- Any major violations of the guidelines (blocking if serious).
- Other important issues.
- Minor suggestions.
                """.strip()
            )
            resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
            return {"audit_logs": [_get_text(resp)]}

        # Vision path: actually look at the page images.
        system = SystemMessage(content="You are a Statistical Editor for *European Urology*.")

        content_parts = [
            {
                "type": "text",
                "text": (
                    "You have the official EU Guidelines for Figures and Tables.\n\n"
                    "---------------- FIGURES/TABLES GUIDELINES ----------------\n"
                    f"{rules}\n"
                    "----------------------------------------------------------\n\n"
                    "You are given PDF page images. Review figures/tables directly from the images.\n"
                    "Be evidence-driven: only claim something is missing/incorrect if you can see it.\n\n"
                    "If Kaplan–Meier / cumulative incidence plots are present, explicitly state:\n"
                    "- Numbers-at-risk shown? (yes/no)\n"
                    "- 95% CI bands/shading shown? (yes/no/unclear)\n"
                    "- Axes labeled with endpoint and time origin? (yes/no/unclear)\n\n"
                    "Write:\n\n### Figures and Tables Check\n"
                    "- Blocking issues\n"
                    "- Important but fixable issues\n"
                    "- Minor suggestions\n"
                ),
            }
        ]
        for img in images:
            content_parts.append({"type": "image_url", "image_url": {"url": img}})

        user = HumanMessage(content=content_parts)

        try:
            resp = llm.invoke([system, user])
            return {"audit_logs": [_get_text(resp)]}
        except Exception as e:
            # Robust fallback: never silently hallucinate; report vision failure and do text-only.
            paper_snip = state["paper_content"]
            prompt = ChatPromptTemplate.from_template(
                """
### Figures and Tables Check

Vision-based review failed with error: {err}

Proceeding with TEXT-ONLY review (manual image verification still required).

You have the official **Guidelines for Reporting of Figures and Tables for Clinical Research in Urology**.

Relevant extracts:
---------------- FIGURES/TABLES GUIDELINES ----------------
{rules}
----------------------------------------------------------

Review the manuscript text for figure/table issues ONLY where the text explicitly supports it.
If something cannot be verified from the text, label it "Manual check needed".

MANUSCRIPT TEXT:
----------------
{paper}
----------------
                """.strip()
            )
            resp2 = (prompt | llm).invoke({"rules": rules, "paper": paper_snip, "err": str(e)})
            return {"audit_logs": [_get_text(resp2)]}

    def type_specific_auditor_node(state: AgentState):
        """Causality for observational; SR/MA for systematic/meta."""
        paper_type = (state.get("paper_type") or "").lower()
        paper_snip = state["paper_content"]

        if "observational" in paper_type:
            try:
                guideline_docs = retrieve_guidelines_by_type(
                    "causality",
                    "causal language confounding causal pathways introduction methods discussion",
                    k=6,
                )
            except Exception as e:
                return {"audit_logs": [f"### Causality / Observational Study Check\nCould not load causality guidelines: {e}"]}

            rules = "\n\n".join(d.page_content for d in guideline_docs)

            prompt = ChatPromptTemplate.from_template(
                """
You are a Statistical Editor for *European Urology*.

The manuscript has been classified as an **Observational Study**.

You have the official **Guidelines for Reporting Observational Research in Urology: The Importance of Clear Reference to Causality**.

Relevant extracts:
---------------- CAUSALITY GUIDELINES ----------------
{rules}
------------------------------------------------------

Using those guidelines, review the manuscript for **causal language and causal thinking**.

Focus on:
- Does the Introduction clearly state whether the question is causal vs descriptive/predictive?
- Are causal mechanisms/pathways described when appropriate?
- Are confounders and their treatment described in the Methods (not just “we adjusted for…”)?
- Is the Results/Discussion language (“reduces risk”, “improves outcomes”, “associated with”) appropriate?

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a section titled:

### Causality / Observational Study Check

Under that heading, summarize:
- Causal clarity in aims and introduction.
- Adequacy of confounding control and discussion.
- Appropriateness of causal vs associational language.
- Blocking issues vs smaller wording/interpretation fixes.
                """.strip()
            )

            resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
            return {"audit_logs": [_get_text(resp)]}

        if "systematic review" in paper_type or "meta-analysis" in paper_type or "meta analysis" in paper_type:
            try:
                guideline_docs = retrieve_guidelines_by_type(
                    "systematic_meta",
                    "PRISMA MOOSE heterogeneity SUCRA protocol reproducible methods",
                    k=6,
                )
            except Exception as e:
                return {"audit_logs": [f"### Systematic Review / Meta-analysis Check\nCould not load SR/MA guidelines: {e}"]}

            rules = "\n\n".join(d.page_content for d in guideline_docs)

            prompt = ChatPromptTemplate.from_template(
                """
You are a Statistical Editor for *European Urology*.

The manuscript has been classified as a **{ptype}**.

You have the official **Guidelines for Meta-analyses and Systematic Reviews in Urology**.

Relevant extracts:
---------------- SR/MA GUIDELINES ----------------
{rules}
--------------------------------------------------

Using those guidelines, review the manuscript for:

- PRISMA/MOOSE-style reporting (flow, inclusion/exclusion, risk of bias).
- Whether the methodology is reported in enough detail for exact replication.
- Interpretation of heterogeneity, rankings (e.g., SUCRA), and precision.

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a section titled:

### Systematic Review / Meta-analysis Check

Summarize:
- Any blocking issues.
- Important but fixable issues.
- Minor suggestions.
                """.strip()
            )

            resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip, "ptype": state.get("paper_type", "")})
            return {"audit_logs": [_get_text(resp)]}

        return {
            "audit_logs": [
                "### Type-Specific Check\nStudy type does not trigger additional causality or SR/MA checks "
                "beyond general statistics and figures/tables."
            ]
        }

    def reporter_node(state: AgentState):
        """Combine logs into a single editorial-style report."""
        logs_text = "\n\n---\n\n".join(state.get("audit_logs", []))

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

A manuscript has been analyzed by several automated checkers.
The manuscript type (as classified) is:

> {paper_type}

Below are their raw notes (unordered, with some overlap):

---------------- ANALYSIS NOTES ----------------
{logs}
------------------------------------------------

Using ONLY these notes (do not invent details you do not see),
draft a concise report in markdown with EXACTLY the following structure:

🇪🇺 European Urology Statistical Report
Detected Type: {paper_type}

Overall summary
- 2–4 bullet points describing the overall quality of reporting & main themes.

Blocking issues
- Bullet list of issues that **must** be fixed before acceptance.
- If none, write: "None."

Important but fixable issues
- Bullet list of non-fatal but important issues.
- If none, write: "None."

Minor issues / suggestions
- Bullet list of minor style, clarity, or presentation suggestions.
- If none, write: "None."

Provisional recommendation
- One line with something like:
  "Acceptable with minor revisions", or
  "Major revisions required", or
  "Not acceptable in current form."

Be concrete but not aggressive in tone, and keep the length similar to an internal editorial note.
            """.strip()
        )

        resp = (prompt | llm).invoke({"paper_type": state.get("paper_type", "Unknown"), "logs": logs_text})
        return {"final_report": _get_text(resp)}

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

    return workflow.compile()
