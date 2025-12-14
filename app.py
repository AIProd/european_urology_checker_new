# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD (Chroma uses sqlite) ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3", sys.modules.get("sqlite3"))
# ----------------------------------------------------------

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback

import indexer
from agent_graph import get_app_graph

load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")
st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"

# Pricing references:
# GPT-5.2: $1.75 in / $14 out per 1M tokens :contentReference[oaicite:2]{index=2}
# GPT-4.1: $2 in / $8 out per 1M tokens :contentReference[oaicite:3]{index=3}
# GPT-5 mini: $0.25 in / $2 out per 1M tokens :contentReference[oaicite:4]{index=4}
# Embeddings: text-embedding-3-large $0.13 / 1M tokens :contentReference[oaicite:5]{index=5}

MODEL_PRICING_USD_PER_1M = {
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.20, "output": 0.80},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
    # If you use other models (e.g., gpt-4o / gpt-4o-mini), add them here.
}


def _guidelines_present() -> bool:
    return os.path.exists(GUIDELINES_DIR) and any(f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR))


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        return None
    return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]


# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY. Set it in .env (local) or in Streamlit Secrets (cloud).")
    st.stop()


# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1) System Setup")

    if _guidelines_present():
        st.success("✅ Guidelines present in ./guidelines")
    else:
        st.warning("⚠️ Guidelines missing. Upload the EU guideline PDFs below.")

    st.divider()
    st.subheader("Model selection")

    recommended_models = [
        "gpt-4.1",      # strong quality/cost
        "gpt-5.2",      # max quality (more expensive)
        "gpt-5-mini",   # fast/cheap
        "gpt-4.1-mini", # fast/cheap-ish
    ]

    selected_model = st.selectbox("Choose model for this run", recommended_models, index=0)

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.caption(
        "Note: for **reasoning models**, hidden *reasoning tokens* are billed as output tokens. "
        "Your billing dashboard is the source of truth."
    )

    st.divider()
    st.subheader("Upload / Update Guidelines")
    st.info(
        "Upload the guideline PDFs:\n"
        "- Causality\n"
        "- Figures and Tables\n"
        "- Stat Reporting Guidelines\n"
        "- Systematic review and MA guidelines"
    )

    uploaded_guidelines = st.file_uploader(
        "Upload Guideline PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Build / Validate Knowledge Base"):
        if not uploaded_guidelines:
            st.error("Please upload the guideline PDFs first.")
        else:
            with st.spinner("Saving guideline PDFs and building Chroma knowledge base..."):
                os.makedirs(GUIDELINES_DIR, exist_ok=True)

                for pdf in uploaded_guidelines:
                    save_path = os.path.join(GUIDELINES_DIR, pdf.name)
                    with open(save_path, "wb") as f:
                        f.write(pdf.getbuffer())

                try:
                    indexer.build_knowledge_base(force_rebuild=True)
                    st.success("✅ Knowledge base built and validated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while building/validating knowledge base: {e}")


# --- MAIN: MANUSCRIPT CHECKER ---
st.header("2) Run Compliance Check")

uploaded_paper = st.file_uploader("Upload Manuscript (PDF)", type="pdf")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        if not _guidelines_present():
            st.error("Upload and build the guideline knowledge base first (left sidebar).")
        else:
            with st.spinner("Agent is analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_paper.read())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    full_text = "\n".join([p.page_content for p in pages])

                    initial_state = {
                        "paper_content": full_text,
                        "paper_type": "",
                        "audit_logs": [],
                        "final_report": "",
                    }

                    app_graph = get_app_graph(model_name=selected_model, temperature=temperature)

                    # Capture usage across all LLM calls in the graph
                    with get_openai_callback() as cb:
                        result = app_graph.invoke(initial_state)

                    review_md = result["final_report"]

                    # Cost estimate
                    est_cost = _estimate_cost_usd(
                        selected_model,
                        prompt_tokens=cb.prompt_tokens,
                        completion_tokens=cb.completion_tokens,
                    )

                    cost_block = (
                        "\n\n---\n\n"
                        "### Run usage & cost estimate\n"
                        f"- Model: `{selected_model}`\n"
                        f"- Prompt tokens: **{cb.prompt_tokens:,}**\n"
                        f"- Output tokens: **{cb.completion_tokens:,}**\n"
                        f"- Total tokens: **{cb.total_tokens:,}**\n"
                    )
                    if est_cost is None:
                        cost_block += (
                            "- Estimated cost: **N/A (model not in local pricing table)**\n"
                            "- Tip: add this model to `MODEL_PRICING_USD_PER_1M` to estimate cost.\n"
                        )
                    else:
                        cost_block += f"- Estimated cost: **${est_cost:.6f}**\n"

                    review_md_with_cost = review_md + cost_block

                    st.success("Analysis Complete")
                    st.markdown(review_md_with_cost)

                    base_name = uploaded_paper.name.rsplit(".", 1)[0]
                    st.download_button(
                        "💾 Download review (Markdown)",
                        data=review_md_with_cost,
                        file_name=f"{base_name}_eu_stats_review.md",
                        mime="text/markdown",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(tmp_path)
