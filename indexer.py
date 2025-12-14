# indexer.py

import os
import shutil
from typing import List, Optional, Dict, Tuple

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

REQUIRED_GUIDELINE_TYPES = ["statistics", "figures_tables", "causality", "systematic_meta"]


def infer_guideline_type(filename: str) -> str:
    fn = filename.lower()
    if "causality" in fn:
        return "causality"
    if "figure" in fn or "table" in fn:
        return "figures_tables"
    if "systematic" in fn or "meta" in fn:
        return "systematic_meta"
    if "stat" in fn:
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

    kwargs = {"model": embedding_model, "api_key": os.getenv("OPENAI_API_KEY")}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def _load_guideline_docs() -> List[Document]:
    if not os.path.exists(GUIDELINES_DIR):
        raise FileNotFoundError(
            f"Guidelines folder '{GUIDELINES_DIR}' not found. "
            "Upload the EU guideline PDFs via the Streamlit sidebar first."
        )

    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {GUIDELINES_DIR}. Upload the guideline PDFs there.")

    documents: List[Document] = []
    for file in pdf_files:
        path = os.path.join(GUIDELINES_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        gtype = infer_guideline_type(file)
        for d in docs:
            d.metadata["source_doc"] = file
            d.metadata["guideline_type"] = gtype
        documents.extend(docs)

    return documents


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=120)
    return splitter.split_documents(docs)


def _get_chroma(embedding: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
    )


def _safe_collection_count(vs: Chroma) -> int:
    """
    Chroma / LangChain versions differ; try a couple ways.
    """
    # Newer LC/Chroma typically exposes _collection.count()
    try:
        return int(vs._collection.count())  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fallback: vs.get() and count ids
    try:
        data = vs.get(include=[])
        ids = data.get("ids") or []
        return len(ids)
    except Exception:
        return 0


def validate_knowledge_base(
    required_types: Optional[List[str]] = None,
) -> Tuple[bool, str, Dict[str, object]]:
    """
    Hard validation that the vector DB exists and is populated.
    Also checks that each required guideline_type has at least 1 retrievable chunk.

    Returns: (ok, message, details)
    """
    required_types = required_types or REQUIRED_GUIDELINE_TYPES

    if not os.path.exists(CHROMA_DIR):
        return (
            False,
            f"Vector DB not found at '{CHROMA_DIR}'. Build/Validate the knowledge base first.",
            {"exists": False, "total_chunks": 0, "missing_types": required_types},
        )

    embedding = _get_embedding_function()
    vs = _get_chroma(embedding)

    total = _safe_collection_count(vs)
    if total <= 0:
        return (
            False,
            f"Vector DB exists at '{CHROMA_DIR}' but appears EMPTY (0 chunks). Rebuild the knowledge base.",
            {"exists": True, "total_chunks": total, "missing_types": required_types},
        )

    missing: List[str] = []
    for gtype in required_types:
        try:
            docs = retrieve_guidelines_by_type(gtype, "guideline", k=1)
        except Exception:
            docs = []
        if not docs:
            missing.append(gtype)

    if missing:
        return (
            False,
            "Vector DB is present, but some required guideline types have 0 retrievable chunks: "
            + ", ".join(missing)
            + ". Re-upload the missing guideline PDFs and rebuild.",
            {"exists": True, "total_chunks": total, "missing_types": missing},
        )

    return (
        True,
        f"Vector DB OK: {total} chunks available, required types present.",
        {"exists": True, "total_chunks": total, "missing_types": []},
    )


def build_knowledge_base(force_rebuild: bool = True) -> None:
    """
    Builds a persistent Chroma index on disk (./chroma_guidelines).
    This happens when you click 'Build / Validate Knowledge Base'.
    """
    docs = _load_guideline_docs()
    chunks = _split_docs(docs)

    embedding = _get_embedding_function()

    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vs = _get_chroma(embedding)
    vs.add_documents(chunks)
    vs.persist()

    # HARD validate after build; fail fast if something went wrong.
    ok, msg, _ = validate_knowledge_base()
    if not ok:
        raise RuntimeError(f"Knowledge base build completed but validation failed: {msg}")

    print(f"✅ Built Chroma KB at {CHROMA_DIR} with {len(chunks)} chunks.")


def retrieve_guidelines_by_type(guideline_type: str, query: str, k: int = 5) -> List[Document]:
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Chroma KB not found at '{CHROMA_DIR}'. "
            "Click 'Build / Validate Knowledge Base' in the sidebar first."
        )

    embedding = _get_embedding_function()
    vs = _get_chroma(embedding)

    try:
        return vs.similarity_search(query, k=k, filter={"guideline_type": guideline_type})
    except TypeError:
        # older versions may use "where"
        return vs.similarity_search(query, k=k, where={"guideline_type": guideline_type})
