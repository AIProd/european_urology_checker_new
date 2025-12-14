# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD (helps chroma/sqlite) ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3", sys.modules.get("sqlite3"))
# ------------------------------------------------------------

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

import indexer
from agent_graph import app_graph

load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")
st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"


def _guidelines_present() -> bool:
    return (
        os.path.exists(GUIDELINES_DIR)
        and any(f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR))
    )


# --- ENV CHECK ---
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "⚠️ OpenAI API key missing (OPENAI_API_KEY). "
        "If running locally, check your .env. If on Streamlit Cloud, check Secrets."
    )
    st.stop()


with st.sidebar:
    st.header("1. System Setup")

    st.caption(f"Chat model: `{os.getenv('OPENAI_MODEL', 'gpt-5.2')}`")
    st.caption(f"Embedding model: `{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')}`")

    if _guidelines_present():
        st.success("✅ Guidelines present in ./guidelines")
    else:
        st.warning("⚠️ Guidelines missing. Upload the 4 EU guideline PDFs below.")

    st.divider()

    st.subheader("Upload / Update Guidelines")
    st.info(
        "Upload the four guideline PDFs:\n"
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
                    st.success("✅ Knowledge base build/validation complete.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while building/validating knowledge base: {e}")


st.header("2. Run Compliance Check")

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

                    result = app_graph.invoke(initial_state)

                    st.success("Analysis Complete")

                    review_md = result["final_report"]
                    st.markdown(review_md)

                    base_name = uploaded_paper.name.rsplit(".", 1)[0]
                    st.download_button(
                        "💾 Download review (Markdown)",
                        data=review_md,
                        file_name=f"{base_name}_eu_stats_review.txt",
                        mime="text/markdown",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
