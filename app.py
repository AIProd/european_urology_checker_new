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


st.set_page_config(page_title="European Urology Checker", layout="wide")
st.title("European Urology: Manuscript Compliance Checker (Stats + Methods)")

# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY. Set it in .env (local) or in Streamlit Secrets (cloud).")
    st.stop()

# Try to ensure KB exists (download if KB_ZIP_URL configured)
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

    st.caption("Guideline PDFs are only needed to (re)build the KB. Normal users can run checks without uploading PDFs.")

    st.divider()
    st.subheader("Model selection (text analysis)")
    recommended_models = ["gpt-5.2", "gpt-4.1", "gpt-5-mini", "gpt-4.1-mini"]
    selected_model = st.selectbox("Choose model for this run", recommended_models, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.divider()
    st.subheader("Figure/Table reading (Vision)")
    use_vision = st.checkbox("Analyze figures/tables from PDF images", value=True)

    # Safer default: use a known multimodal model for vision
    vision_model = st.selectbox(
        "Vision model",
        ["gpt-5.2", "gpt-4.1"],
        index=0,
        disabled=not use_vision,
    )
    max_vision_pages = st.slider("Max pages to analyze with vision", 2, 20, 14, disabled=not use_vision)

    st.divider()
    # --- KB build/rebuild (admin) ---
    kb_url = (os.getenv("KB_ZIP_URL") or "").strip()
    admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()

    is_admin = False
    if admin_token:
        entered_token = st.text_input("Admin token (only needed to rebuild KB)", type="password")
        is_admin = bool(entered_token) and (entered_token == admin_token)

    # If KB_ZIP_URL is configured, normal users should not build locally.
    allow_build_ui = ((not kb_status.get("ready")) and (not kb_url)) or is_admin

    if allow_build_ui:
        st.subheader("Upload / Update Guidelines")
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
                        # If KB isn't ready, it's safe to rebuild. Admin token allows forced refresh anytime.
                        force_rebuild = is_admin or (not kb_status.get("ready"))
                        indexer.build_knowledge_base(force_rebuild=force_rebuild)
                        st.success("✅ Knowledge base built and validated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while building/validating knowledge base: {e}")
    else:
        st.info("Knowledge base already available. Upload/rebuild is disabled for non-admin users.")
        if kb_url:
            st.caption("KB is expected to be auto-downloaded from KB_ZIP_URL when missing.")

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
                "Vector DB is not ready. Ask the admin to provide/build the KB.\n\n"
                f"Details: {kb_status['details']}"
            )
            st.stop()

        with st.spinner("Analyzing manuscript (including figures/tables if enabled)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_paper.read())
                tmp_path = tmp.name

            try:
                # Extract text (fast)
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])

                # Extract visuals (true figure reading)
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
