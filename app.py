# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD (Chroma uses sqlite) ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3", sys.modules.get("sqlite3"))
# ----------------------------------------------------------

import base64
import io
import os
import tempfile
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

import indexer
from agent_graph import get_app_graph

# PyMuPDF + PIL for PDF -> images
import fitz  # pymupdf
from PIL import Image

load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")
st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"

# -------- Vision extraction tuning ----------
VISION_SCALE = 2.0
VISION_MAX_PAGES = 12
VISION_SEND_PAGES = 8
VISION_JPEG_QUALITY = 85
VISION_MAX_SIDE = 1800
# --------------------------------------------

MODEL_PRICING_USD_PER_1M = {
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.20, "output": 0.80},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
}


def _guidelines_present() -> bool:
    return os.path.exists(GUIDELINES_DIR) and any(f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR))


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        return None
    return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]


def _img_to_data_url(pil_img: Image.Image) -> str:
    w, h = pil_img.size
    mx = max(w, h)
    if mx > VISION_MAX_SIDE:
        s = VISION_MAX_SIDE / mx
        pil_img = pil_img.resize((int(w * s), int(h * s)))

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=VISION_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _find_candidate_visual_pages(doc: fitz.Document) -> List[int]:
    keywords = [
        "Figure", "Fig.", "Table", "Kaplan", "numbers at risk", "at risk",
        "cumulative incidence", "confidence interval", "95% CI", "forest plot"
    ]
    hits: List[int] = []
    for i in range(len(doc)):
        t = (doc[i].get_text("text") or "")
        if any(k.lower() in t.lower() for k in keywords):
            hits.append(i)

    if not hits:
        return list(range(min(VISION_MAX_PAGES, len(doc))))

    expanded = []
    for i in hits:
        expanded.extend([i - 1, i, i + 1])

    seen = set()
    ordered: List[int] = []
    for i in expanded:
        if 0 <= i < len(doc) and i not in seen:
            seen.add(i)
            ordered.append(i)

    return ordered[:VISION_MAX_PAGES]


def extract_text_and_images(pdf_path: str) -> Tuple[str, List[str]]:
    doc = fitz.open(pdf_path)

    texts = []
    for i in range(len(doc)):
        texts.append(doc[i].get_text("text") or "")
    full_text = "\n".join(texts)

    page_indices = _find_candidate_visual_pages(doc)

    images: List[str] = []
    for i in page_indices:
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(VISION_SCALE, VISION_SCALE), alpha=False)
        pil = Image.open(io.BytesIO(pix.tobytes("jpeg")))
        images.append(_img_to_data_url(pil))

    return full_text, images


def _kb_gate_or_stop() -> None:
    """
    Hard gate: prevent agent run if vector DB isn't present/loaded.
    """
    ok, msg, details = indexer.validate_knowledge_base()
    if ok:
        return

    st.error("⚠️ Knowledge base not ready — analysis is blocked.")
    st.markdown(
        f"""
**Reason:** {msg}

**What to do:**
1) Upload the EU guideline PDFs in the sidebar  
2) Click **Build / Validate Knowledge Base**  
3) Re-run the manuscript analysis

**Debug:**  
- Vector DB exists: `{details.get("exists")}`  
- Total chunks: `{details.get("total_chunks")}`  
- Missing types: `{details.get("missing_types")}`
"""
    )
    st.stop()


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

    # Show KB status (even before analysis)
    ok, msg, details = indexer.validate_knowledge_base()
    if ok:
        st.success(f"✅ Vector DB ready ({details.get('total_chunks')} chunks)")
    else:
        st.warning("⚠️ Vector DB not ready (analysis will be blocked)")
        st.caption(msg)

    st.divider()
    st.subheader("Model selection")

    recommended_models = [ 
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5-pro",
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]

    selected_model = st.selectbox("Choose model for this run", recommended_models, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

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
        # HARD gate right before run (covers the “sometimes runs without db” case)
        _kb_gate_or_stop()

        if not _guidelines_present():
            st.error("Upload the guideline PDFs and rebuild the knowledge base first (left sidebar).")
            st.stop()

        with st.spinner("Agent is analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_paper.read())
                tmp_path = tmp.name

            try:
                full_text, page_images = extract_text_and_images(tmp_path)

                initial_state = {
                    "paper_content": full_text,
                    "paper_type": "",
                    "paper_images": page_images[:VISION_SEND_PAGES],
                    "audit_logs": [],
                    "final_report": "",
                }

                app_graph = get_app_graph(model_name=selected_model, temperature=temperature)

                with get_openai_callback() as cb:
                    result = app_graph.invoke(initial_state)

                review_md = result["final_report"]

                # Add an explicit KB status footer so if someone screenshots output,
                # it always records whether KB was loaded.
                ok, msg, details = indexer.validate_knowledge_base()
                kb_stamp = (
                    "\n\n---\n\n"
                    "### Knowledge base status\n"
                    f"- Vector DB ready: **{ok}**\n"
                    f"- Details: {msg}\n"
                )

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

                review_md_with_meta = review_md + kb_stamp + cost_block

                st.success("Analysis Complete")
                st.markdown(review_md_with_meta)

                base_name = uploaded_paper.name.rsplit(".", 1)[0]
                st.download_button(
                    "💾 Download review (Markdown)",
                    data=review_md_with_meta,
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
