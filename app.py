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
from langchain_openai import ChatOpenAI

import indexer
from agent_graph import get_app_graph
from pdf_reader import summarize_pdf_visuals

load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")
st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"

MODEL_PRICING_USD_PER_1M = {
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.20, "output": 0.80},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
}


def _guidelines_present() -> bool:
    return os.path.exists(GUIDELINES_DIR) and any(
        f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR)
    )


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        return None
    return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]


def _make_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key, "model": model_name, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY. Set it in .env (local) or in Streamlit Secrets (cloud).")
    st.stop()

# Ensure KB exists (extract from chroma_guidelines.zip if needed)
kb_status = indexer.ensure_knowledge_base_present()

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1) System Setup")

    # Show KB status
    if kb_status.get("ready"):
        st.success(f"✅ Knowledge base ready ({kb_status['total_chunks']} chunks)")
    else:
        st.error("❌ Knowledge base NOT ready")
        st.caption(kb_status.get("details", ""))

    st.caption(
        "Normal users do NOT need guideline PDFs. PDFs are only needed if an admin wants to rebuild the KB."
    )

    st.divider()
    st.subheader("Model selection (text analysis)")
    recommended_models = ["gpt-5.2", "gpt-4.1", "gpt-5-mini", "gpt-4.1-mini"]
    selected_model = st.selectbox("Choose model for this run", recommended_models, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.divider()
    st.subheader("Figure/Table reading (Vision)")
    use_vision = st.checkbox("Analyze figures/tables from PDF images", value=True)

    vision_model = st.selectbox(
        "Vision model",
        ["gpt-5.2", "gpt-4.1"],
        index=0,
        disabled=not use_vision,
    )
    max_vision_pages = st.slider("Max pages to analyze with vision", 2, 20, 14, disabled=not use_vision)

    st.divider()

    # --- Admin gate for rebuilding KB ---
    admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
    is_admin = False
    if admin_token:
        entered = st.text_input("Admin token (only needed to rebuild KB)", type="password")
        is_admin = bool(entered) and (entered == admin_token)

    # Only show rebuild UI if:
    # - KB is not ready (emergency), OR
    # - user is admin
    if (not kb_status.get("ready")) or is_admin:
        st.subheader("Upload / Update Guidelines (Admin)")

        if _guidelines_present():
            st.success("✅ Guidelines present in ./guidelines")
        else:
            st.warning("⚠️ Guidelines missing in ./guidelines (only needed to rebuild KB).")

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
    else:
        st.info("KB is ready. Rebuild is disabled for non-admin users.")

    st.divider()
    st.subheader("Knowledge base status")
    kb_status = indexer.ensure_knowledge_base_present()
    if kb_status["ready"]:
        st.success(f"✅ Vector DB ready ({kb_status['total_chunks']} chunks)")
    else:
        st.error("❌ Vector DB NOT ready")
    st.caption(kb_status["details"])


# --- MAIN: MANUSCRIPT CHECKER ---
st.header("2) Run Compliance Check")
uploaded_paper = st.file_uploader("Upload Manuscript (PDF)", type="pdf")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        kb_status = indexer.ensure_knowledge_base_present()
        if not kb_status["ready"]:
            st.error(
                "Vector DB is not ready. Make sure `chroma_guidelines.zip` exists in the repo root "
                "or ask the admin to rebuild.\n\n"
                f"Details: {kb_status['details']}"
            )
            st.stop()

        with st.spinner("Analyzing manuscript (including figures/tables if enabled)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_paper.read())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])

                paper_visuals = ""
                visuals_used = False
                visuals_error = ""

                with get_openai_callback() as cb:
                    if use_vision:
                        try:
                            llm_vision = _make_llm(model_name=vision_model, temperature=0.0)
                            paper_visuals = summarize_pdf_visuals(
                                pdf_path=tmp_path,
                                llm=llm_vision,
                                max_pages=max_vision_pages,
                                zoom=2.0,
                            )
                            visuals_used = True
                        except Exception as e:
                            visuals_error = str(e)
                            paper_visuals = (
                                "### Visual extracts (from PDF page images)\n\n"
                                f"_Vision extraction failed: {visuals_error}_\n"
                            )

                    initial_state = {
                        "paper_content": full_text,
                        "paper_visuals": paper_visuals,
                        "paper_type": "",
                        "kb_ready": kb_status["ready"],
                        "kb_details": kb_status["details"],
                        "visuals_used": visuals_used,
                        "audit_logs": [],
                        "final_report": "",
                    }

                    app_graph = get_app_graph(model_name=selected_model, temperature=temperature)
                    result = app_graph.invoke(initial_state)

                    review_md = result["final_report"]

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
                        cost_block += "- Estimated cost: **N/A (model not in local pricing table)**\n"
                    else:
                        cost_block += f"- Estimated cost: **${est_cost:.6f}**\n"

                    if visuals_used:
                        cost_block += "- Vision: **enabled**\n"
                    else:
                        cost_block += "- Vision: **disabled**\n"
                        if visuals_error:
                            cost_block += f"- Vision error: `{visuals_error}`\n"

                st.subheader("📄 Review Output")
                st.markdown(review_md)

                st.download_button(
                    "💾 Download review (Markdown)",
                    data=review_md + cost_block,
                    file_name=f"{uploaded_paper.name.rsplit('.', 1)[0]}_eu_stats_review.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
