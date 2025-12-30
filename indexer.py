# indexer.py
import os
import shutil
import tempfile
import zipfile
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

GUIDELINES_DIR = "./guidelines"
CHROMA_DIR = "./chroma_guidelines"
CHROMA_COLLECTION = "eu_guidelines"

# If you commit the zip into the repo root, keep it here:
KB_ZIP_PATH = "./chroma_guidelines.zip"


def infer_guideline_type(filename: str) -> str:
    """Heuristic mapping based on filename."""
    n = filename.lower()
    if "figure" in n or "table" in n or "fig" in n:
        return "figures_tables"
    if "causal" in n or "causality" in n:
        return "causality"
    if "systematic" in n or "meta" in n:
        return "systematic_meta"
    if "stat" in n or "statistics" in n or "p-value" in n:
        return "statistics"
    return "other"


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing env var: {name}")
    return val


def _get_embedding_function() -> OpenAIEmbeddings:
    _ = _require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    kwargs = {"model": embedding_model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def _load_guideline_docs() -> List[Document]:
    if not os.path.exists(GUIDELINES_DIR):
        raise FileNotFoundError(
            f"Guidelines folder not found: {GUIDELINES_DIR}. Create it and add PDFs."
        )

    pdfs = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {GUIDELINES_DIR}.")

    docs: List[Document] = []
    for pdf in pdfs:
        path = os.path.join(GUIDELINES_DIR, pdf)
        loader = PyPDFLoader(path)
        loaded = loader.load()
        gtype = infer_guideline_type(pdf)
        for d in loaded:
            d.metadata["guideline_type"] = gtype
            d.metadata["source_pdf"] = pdf
        docs.extend(loaded)

    return docs


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)


def _get_chroma(embedding: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
    )


def build_knowledge_base(force_rebuild: bool = False) -> None:
    """
    Builds the Chroma persisted vector DB in CHROMA_DIR.

    - If CHROMA_DIR already exists and looks valid, this does nothing unless force_rebuild=True.
    - If force_rebuild=True, CHROMA_DIR is wiped and rebuilt from PDFs in GUIDELINES_DIR.
    """
    docs = _load_guideline_docs()
    chunks = _split_docs(docs)
    embedding = _get_embedding_function()

    if (not force_rebuild) and os.path.exists(CHROMA_DIR):
        status = get_knowledge_base_status()
        if status.get("ready"):
            print(f"ℹ️ KB already ready at {CHROMA_DIR}. Skipping rebuild.")
            return

    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vs = _get_chroma(embedding)
    vs.add_documents(chunks)
    vs.persist()

    status = get_knowledge_base_status()
    if not status["ready"]:
        raise RuntimeError(
            f"Vector DB build completed but validation failed: {status.get('details')}"
        )

    print(f"✅ Built Chroma KB at {CHROMA_DIR} with {len(chunks)} chunks.")


def retrieve_guidelines_by_type(guideline_type: str, query: str, k: int = 6) -> List[str]:
    embedding = _get_embedding_function()
    vs = _get_chroma(embedding)

    results = vs.similarity_search(
        query,
        k=k,
        filter={"guideline_type": guideline_type},
    )
    return [r.page_content for r in results]


def get_knowledge_base_status(
    required_types: Optional[List[str]] = None,
    min_total_chunks: int = 20,
) -> Dict[str, object]:
    """
    Readiness check for the persisted KB.
    """
    if required_types is None:
        required_types = ["statistics", "figures_tables", "causality", "systematic_meta"]

    if not os.path.exists(CHROMA_DIR):
        return {
            "ready": False,
            "total_chunks": 0,
            "type_counts": {},
            "details": f"Chroma directory not found at {CHROMA_DIR}.",
        }

    try:
        embedding = _get_embedding_function()
        vs = _get_chroma(embedding)

        type_counts: Dict[str, int] = {}
        total = 0

        # total chunk count (best-effort)
        try:
            total = int(vs._collection.count())  # type: ignore[attr-defined]
        except Exception:
            total = 0

        # required types presence probe
        for t in required_types:
            try:
                hits = vs.similarity_search("probe", k=1, filter={"guideline_type": t})
                type_counts[t] = 1 if hits else 0
            except Exception:
                type_counts[t] = 0

        missing = [t for t in required_types if type_counts.get(t, 0) <= 0]
        ready = (total >= min_total_chunks) and (len(missing) == 0)

        details = f"Vector DB {'OK' if ready else 'NOT ready'}: {total} chunks available."
        if missing:
            details += f" Missing types: {missing}"

        return {"ready": ready, "total_chunks": total, "type_counts": type_counts, "details": details}

    except Exception as e:
        return {
            "ready": False,
            "total_chunks": 0,
            "type_counts": {},
            "details": f"KB status check failed: {e}",
        }


def _safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """
    Safe-ish zip extraction: prevents Zip Slip by validating paths.
    Extracts to current project (dest_dir is usually ".").
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member_path = member.filename

            # Skip weird entries
            if not member_path or member_path.endswith("/") and member.file_size == 0:
                continue

            # Resolve final path
            final_path = os.path.normpath(os.path.join(dest_dir, member_path))
            dest_abs = os.path.abspath(dest_dir)
            final_abs = os.path.abspath(final_path)

            if not final_abs.startswith(dest_abs + os.sep) and final_abs != dest_abs:
                raise RuntimeError(f"Unsafe zip path detected: {member_path}")

        z.extractall(dest_dir)


def ensure_knowledge_base_present() -> Dict[str, object]:
    """
    Ensures a usable persisted Chroma KB exists at CHROMA_DIR.

    Priority:
    1) If CHROMA_DIR already exists & validates -> use it
    2) Else if KB_ZIP_PATH exists -> extract it to create CHROMA_DIR
    3) Else -> not ready
    """
    status = get_knowledge_base_status()
    if status.get("ready"):
        return status

    if os.path.exists(KB_ZIP_PATH):
        try:
            # Extract zip into repo root; it should contain chroma_guidelines/...
            _safe_extract_zip(KB_ZIP_PATH, ".")
            return get_knowledge_base_status()
        except Exception as e:
            return {
                "ready": False,
                "total_chunks": 0,
                "type_counts": {},
                "details": f"KB zip found at {KB_ZIP_PATH} but extraction/validation failed: {e}",
            }

    # Nothing available
    return {
        "ready": False,
        "total_chunks": 0,
        "type_counts": {},
        "details": f"KB missing: {CHROMA_DIR} not found and {KB_ZIP_PATH} not found.",
    }
