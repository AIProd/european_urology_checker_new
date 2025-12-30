# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD (Chroma uses sqlite) ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3", sys.modules.get("sqlite3"))
# ----------------------------------------------------------

import os
import tempfile
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

import indexer
from agent_graph import run_graph

load_dotenv()

GUIDELINES_DIR = "./guidelines"

MODEL_PRICING_USD_PER_1M = {
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.50, "output": 2.00},
    "gpt-5-nano": {"input": 0.20, "output": 0.80},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
}


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
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


st.set_page_config(page_title="European Urology Checker", layout="wide")
st.title("European Urology: Manuscript Compliance Checker (Stats + Methods)")

# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY. Set it in .env (local) or in Streamlit Secrets (cloud).")
    st.stop()

# Ensure KB is present (from folder or from ./chroma_guidelines.zip)
kb_status = indexer.ensure_knowledge_base_present()

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1) System Setup")

    # Knowledge base (shared) status
    if kb_status.get("ready"):
        st.success(f"✅ Knowledge base ready ({kb_status['total_chunks']} chunks)")
    else:
        st.error("❌ Knowledge base NOT ready")
        st.caption(kb_status.get("details", ""))

    st.caption(
        "Guideline PDFs are only needed to (re)build the KB. "
        "Normal users can run checks if chroma_guidelines/ or chroma_guidelines.zip is present."
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

    # --- KB build/rebuild (admin-only optional) ---
    # If you don't set ADMIN_TOKEN, rebuild UI will be shown only when KB is missing AND zip is missing.
    admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
    is_admin = False
    if admin_token:
        entered_token = st.text_input("Admin token (only needed to rebuild KB)", type="password")
        is_admin = bool(entered_token) and (entered_token == admin_token)

    # Allow build UI only if:
    # - KB is not ready AND there is no committed zip
    # OR admin token is correct
    zip_exists = os.path.exists(indexer.KB_ZIP_PATH)
    allow_build_ui = (not kb_status.get("ready") and not zip_exists) or is_admin

    if allow_build_ui:
        st.subheader("Upload / Update Guidelines (Build KB)")
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
                        # admin can force refresh; otherwise build once
                        force_rebuild = is_admin
                        indexer.build_knowledge_base(force_rebuild=force_rebuild)
                        st.success("✅ Knowledge base built and validated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while building/validating knowledge base: {e}")
    else:
        if kb_status.get("ready"):
            st.info("Knowledge base available. Upload/rebuild disabled for non-admin users.")
        else:
            st.info("KB zip exists; it will be extracted automatically on start. Rebuild disabled for non-admin users.")

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
                "Vector DB is not ready. Add chroma_guidelines.zip (repo root) or build KB as admin.\n\n"
                f"Details: {kb_status['details']}"
            )
            st.stop()

        with st.spinner("Analyzing manuscript (including figures/tables if enabled)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_paper.read())
                tmp_path = tmp.name

            try:
                # Extract text
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])

                # Extract visuals (optional)
                paper_visuals = ""
                visuals_used = False
                visuals_error = ""

                with get_openai_callback() as cb:
                    if use_vision:
                        try:
                            from pdf_reader import extract_pdf_visuals

                            paper_visuals = extract_pdf_visuals(
                                tmp_path,
                                model=vision_model,
                                max_pages=max_vision_pages,
                            )
                            visuals_used = True
                        except Exception as ve:
                            visuals_error = str(ve)

                    # Run graph
                    review_md = run_graph(
                        paper_text=full_text,
                        paper_visuals=paper_visuals,
                        model_name=selected_model,
                        temperature=temperature,
                    )

                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens
                    total_tokens = cb.total_tokens

                # Cost estimation
                cost = _estimate_cost_usd(selected_model, prompt_tokens, completion_tokens)
                cost_block = "\n\n---\n"
                cost_block += f"**Token usage:** prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}\n"
                if cost is not None:
                    cost_block += f"**Estimated cost:** ${cost:.4f}\n"
                else:
                    cost_block += "**Estimated cost:** unknown (pricing not configured for this model)\n"

                if visuals_used:
                    cost_block += "**Vision:** enabled\n"
                else:
                    cost_block += "**Vision:** disabled\n"
                    if visuals_error:
                        cost_block += f"**Vision error:** {visuals_error}\n"

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
