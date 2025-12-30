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
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key, "model": model_name, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY. Set it in .env (local) or in Streamlit Secrets (cloud).")
    st.stop()

# Try to ensure the shared KB exists (auto-download if KB_ZIP_URL is configured)
kb_status = indexer.ensure_knowledge_base_present()


# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1) System Setup")

    st.caption("This app uses a shared Chroma knowledge base (KB). Normal users don’t need to upload guideline PDFs.")
    st.caption("If KB_ZIP_URL is set, the app will auto-download the KB when missing.")

    st.divider()
    st.subheader("Model selection")

    model_name = st.selectbox(
        "Choose model",
        options=["gpt-5-mini", "gpt-5.2", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.subheader("Optional checks")
    include_figtab = st.checkbox("Include figures/tables check (slower)", value=True)
    include_type_specific = st.checkbox("Include study-type specific checks", value=True)

    st.divider()
    st.subheader("Admin: Build / Refresh Knowledge Base")

    admin_token = os.getenv("ADMIN_TOKEN", "")
    if admin_token:
        entered = st.text_input(
            "Admin token",
            type="password",
            help="Required to build/refresh the shared KB for all users.",
        )
        is_admin = entered == admin_token
    else:
        is_admin = True
        st.caption("ADMIN_TOKEN is not set; build controls are unlocked.")

    if not is_admin:
        st.info("KB build/refresh is disabled for non-admin users.")
    else:
        # Source PDFs are only needed for (re)building the KB
        uploaded_guidelines = st.file_uploader(
            "Upload Guideline PDFs (admin only)",
            type="pdf",
            accept_multiple_files=True,
        )

        force_rebuild = st.checkbox("Force rebuild (wipe existing KB)", value=True)

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
                        indexer.build_knowledge_base(force_rebuild=force_rebuild)
                        st.success("✅ Knowledge base built and validated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while building/validating knowledge base: {e}")

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

st.divider()
st.subheader("Run")
st.caption("This will extract the paper text and optionally summarize visuals (figures/tables) before running the agents.")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        kb_status = indexer.ensure_knowledge_base_present()
        if not kb_status["ready"]:
            st.error(
                "Knowledge base is not ready. Ask the admin to configure KB_ZIP_URL or rebuild the KB in the sidebar."
            )
            st.caption(kb_status["details"])
            st.stop()

        with st.spinner("Analyzing manuscript (including figures/tables if enabled)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_paper.getbuffer())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                paper_text = "\n\n".join(d.page_content for d in docs)

                paper_visuals = ""
                if include_figtab:
                    paper_visuals = summarize_pdf_visuals(tmp_path)

                llm = _make_llm(model_name=model_name, temperature=temperature)

                app = get_app_graph(
                    include_type_specific=include_type_specific,
                    include_figtab=include_figtab,
                )

                state = {
                    "paper_content": paper_text,
                    "paper_visuals": paper_visuals,
                    "audit_logs": [],
                }

                with get_openai_callback() as cb:
                    result = app.invoke(state)

                    review_md = "\n\n".join(result.get("audit_logs", []))

                    cost_usd = _estimate_cost_usd(
                        model=model_name,
                        prompt_tokens=cb.prompt_tokens,
                        completion_tokens=cb.completion_tokens,
                    )
                    cost_block = (
                        f"\n\n---\n\n**Tokens**: prompt={cb.prompt_tokens}, completion={cb.completion_tokens}, total={cb.total_tokens}"
                    )
                    if cost_usd is not None:
                        cost_block += f"\n\n**Estimated cost**: ${cost_usd:.4f} (USD)"
                    else:
                        cost_block += "\n\n**Estimated cost**: (unknown model pricing)"

                st.subheader("✅ Review output")
                st.markdown(review_md + cost_block)

                base_name = os.path.splitext(uploaded_paper.name)[0]
                st.download_button(
                    "💾 Download review (Markdown)",
                    data=review_md + cost_block,
                    file_name=f"{base_name}_eu_stats_review.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
else:
    st.info("Upload a manuscript PDF to begin.")
