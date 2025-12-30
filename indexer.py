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
CHROMA_COLLECTION = "eu_guidelines"


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
    docs = _load_guideline_docs()
    chunks = _split_docs(docs)
    embedding = _get_embedding_function()

    # Avoid accidental duplicate-ingestion:
    # If the KB is already ready and force_rebuild is False, do nothing.
    if (not force_rebuild) and os.path.exists(CHROMA_DIR):
        try:
            status = get_knowledge_base_status()
            if status.get("ready"):
                print(f"ℹ️ KB already ready at {CHROMA_DIR}. Skipping rebuild.")
                return
        except Exception:
            # If status check fails, continue with build attempt.
            pass

    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vs = _get_chroma(embedding)
    vs.add_documents(chunks)
    vs.persist()

    # Validation
    status = get_knowledge_base_status()
    if not status["ready"]:
        raise RuntimeError(f"Vector DB build completed but validation failed: {status['details']}")

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

        total = None
        type_counts: Dict[str, int] = {}

        # Best effort count
        try:
            total = int(vs._collection.count())
            for t in required_types:
                try:
                    # count per type isn't directly supported, so do a cheap existence probe
                    hits = vs.similarity_search("test", k=1, filter={"guideline_type": t})
                    type_counts[t] = 1 if len(hits) > 0 else 0
                except Exception:
                    type_counts[t] = 0
        except Exception:
            # fallback: query-based existence check
            total = 0
            for t in required_types:
                type_counts[t] = 1 if len(retrieve_guidelines_by_type(t, "test", k=1)) > 0 else 0
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


def ensure_knowledge_base_present() -> Dict[str, object]:
    """
    Ensures a usable persisted Chroma KB exists at CHROMA_DIR.

    - If CHROMA_DIR exists and passes validation: returns status.
    - If missing and KB_ZIP_URL env var is set: downloads a zip and extracts it.
      The zip should contain a top-level 'chroma_guidelines/' folder.
    - Otherwise returns the current status (not ready).
    """
    status = get_knowledge_base_status()
    if status.get("ready"):
        return status

    kb_url = (os.getenv("KB_ZIP_URL") or "").strip()
    if not kb_url:
        return status

    try:
        tmpdir = tempfile.mkdtemp(prefix="kb_dl_")
        zpath = os.path.join(tmpdir, "kb.zip")

        with urllib.request.urlopen(kb_url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)

        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(".")

        return get_knowledge_base_status()

    except Exception as e:
        return {
            "ready": False,
            "total_chunks": 0,
            "type_counts": {},
            "details": f"KB missing and auto-download failed: {e}",
        }
