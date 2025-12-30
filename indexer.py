# indexer.py
import os
import shutil
import tempfile
import zipfile
import urllib.request
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


def infer_guideline_type(filename: str) -> str:
    """
    Heuristic to tag each PDF into one of the guideline types used by the agents.
    Adjust mappings as your file naming conventions evolve.
    """
    fn = filename.lower()
    if "stat" in fn:
        return "statistics"
    if "figure" in fn or "table" in fn or "figtab" in fn:
        return "figures_tables"
    if "causal" in fn:
        return "causality"
    if "systematic" in fn or "meta" in fn:
        return "systematic_meta"
    # default bucket
    return "statistics"


def _require_env(var_name: str) -> str:
    v = os.getenv(var_name)
    if not v:
        raise RuntimeError(
            f"Missing environment variable: {var_name}. "
            "Set it in .env (local) or in Streamlit Secrets (cloud)."
        )
    return v


def _get_embedding_function() -> OpenAIEmbeddings:
    _require_env("OPENAI_API_KEY")
    # You can pin a specific embeddings model if you want; OpenAIEmbeddings will use defaults otherwise.
    return OpenAIEmbeddings()


def _load_guideline_docs() -> List[Document]:
    if not os.path.exists(GUIDELINES_DIR):
        raise RuntimeError(
            f"Guidelines directory '{GUIDELINES_DIR}' does not exist. "
            "Admins: upload PDFs in the sidebar (or create the folder locally)."
        )

    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in '{GUIDELINES_DIR}'. Admins: upload guideline PDFs in the sidebar."
        )

    all_docs: List[Document] = []
    for fname in pdf_files:
        path = os.path.join(GUIDELINES_DIR, fname)
        loader = PyPDFLoader(path)
        docs = loader.load()
        gtype = infer_guideline_type(fname)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = fname
            d.metadata["guideline_type"] = gtype
        all_docs.extend(docs)
    return all_docs


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Ensure each chunk has the expected metadata
    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.setdefault("guideline_type", "statistics")
        c.metadata.setdefault("source", "unknown")
    return chunks


def _get_chroma(embedding: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
        collection_name="eu_guidelines",
    )


def build_knowledge_base(force_rebuild: bool = False) -> None:
    """
    Builds/refreshes the knowledge base from PDFs in GUIDELINES_DIR.

    - If force_rebuild=True: wipes CHROMA_DIR before rebuilding.
    - Otherwise: appends/updates (Chroma behavior depends on ids/collection).
    """
    docs = _load_guideline_docs()
    chunks = _split_docs(docs)
    embedding = _get_embedding_function()

    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vs = _get_chroma(embedding)
    vs.add_documents(chunks)
    vs.persist()

    # Validation
    status = get_knowledge_base_status()
    if not status["ready"]:
        raise RuntimeError(f"Vector DB build completed but validation failed: {status['details']}")


def retrieve_guidelines_by_type(guideline_type: str, query: str, k: int = 5) -> List[Document]:
    # If KB isn't on disk, try to fetch it (e.g., from GitHub) before failing.
    ensure_knowledge_base_present()

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Chroma KB not found at '{CHROMA_DIR}'. "
            "Admin: build/refresh the KB in the sidebar, or configure KB_ZIP_URL."
        )

    embedding = _get_embedding_function()
    vs = _get_chroma(embedding)

    try:
        return vs.similarity_search(query, k=k, filter={"guideline_type": guideline_type})
    except TypeError:
        # Some Chroma versions use `where=` instead of `filter=`
        return vs.similarity_search(query, k=k, where={"guideline_type": guideline_type})


def get_knowledge_base_status(
    required_types: Optional[List[str]] = None,
    min_total_chunks: int = 20,
) -> Dict[str, object]:
    """
    Hard readiness check so you can BLOCK runs if the DB is empty/missing key types.
    """
    if required_types is None:
        required_types = ["statistics", "figures_tables", "causality", "systematic_meta"]

    if not os.path.exists(CHROMA_DIR):
        return {
            "ready": False,
            "total_chunks": 0,
            "type_counts": {},
            "details": f"Chroma directory not found at {CHROMA_DIR}. Build the KB first.",
        }

    try:
        embedding = _get_embedding_function()
        vs = _get_chroma(embedding)

        type_counts: Dict[str, int] = {}
        total: Optional[int] = None

        try:
            total = int(vs._collection.count())
            for t in required_types:
                try:
                    type_counts[t] = int(vs._collection.count(where={"guideline_type": t}))
                except Exception:
                    # fallback: query-based existence check
                    type_counts[t] = 1 if len(retrieve_guidelines_by_type(t, "test", k=1)) > 0 else 0
        except Exception:
            # fallback if _collection not available
            total = 0
            for t in required_types:
                type_counts[t] = 1 if len(retrieve_guidelines_by_type(t, "test", k=1)) > 0 else 0
            # if any type exists, we still treat as >0
            total = sum(type_counts.values())

        missing = [t for t in required_types if type_counts.get(t, 0) <= 0]

        ready = (total is not None and total >= min_total_chunks and len(missing) == 0)
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


def ensure_knowledge_base_present(
    required_types: Optional[List[str]] = None,
    min_total_chunks: int = 20,
) -> Dict[str, object]:
    """
    Ensures a persisted Chroma knowledge base is present locally.

    Behavior:
    - If CHROMA_DIR exists and is non-empty, uses it.
    - Otherwise, if KB_ZIP_URL is set (e.g., a GitHub Release asset URL),
      downloads a zip and extracts it into the project root (expects it to contain
      a `chroma_guidelines/` folder).
    - Returns `get_knowledge_base_status(...)` either way.

    This lets you build the KB once (admin) and reuse it for all users.
    """
    # Fast-path: local KB already exists
    if os.path.exists(CHROMA_DIR):
        try:
            if any(True for _ in os.scandir(CHROMA_DIR)):
                return get_knowledge_base_status(
                    required_types=required_types,
                    min_total_chunks=min_total_chunks,
                )
        except Exception:
            # If scandir fails, fall back to status check
            return get_knowledge_base_status(
                required_types=required_types,
                min_total_chunks=min_total_chunks,
            )

    kb_url = os.getenv("KB_ZIP_URL")
    if not kb_url:
        return get_knowledge_base_status(
            required_types=required_types,
            min_total_chunks=min_total_chunks,
        )

    # Attempt download + extract
    try:
        tmpdir = tempfile.mkdtemp(prefix="kb_dl_")
        zpath = os.path.join(tmpdir, "kb.zip")

        with urllib.request.urlopen(kb_url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)

        # Extract into current working directory (project root when run via Streamlit)
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(".")

    except Exception as e:
        return {
            "ready": False,
            "total_chunks": 0,
            "type_counts": {},
            "details": f"KB missing and auto-download failed: {e}",
        }

    return get_knowledge_base_status(
        required_types=required_types,
        min_total_chunks=min_total_chunks,
    )
