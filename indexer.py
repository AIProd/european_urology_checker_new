# indexer.py

import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

GUIDELINES_DIR = "./guidelines"
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "eu_guidelines"

DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def infer_guideline_type(filename: str) -> str:
    """Map guideline filename to a high-level type."""
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


def _get_embedding_function() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    return OpenAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        openai_api_key=api_key,
    )


def _load_guideline_docs() -> List[Document]:
    """Load all guideline PDFs as LangChain Documents with guideline_type metadata."""
    if not os.path.exists(GUIDELINES_DIR):
        raise FileNotFoundError(
            f"Guidelines folder '{GUIDELINES_DIR}' not found. "
            "Upload the EU guideline PDFs via the Streamlit sidebar first."
        )

    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {GUIDELINES_DIR}. "
            "Upload the 4 EU guideline PDFs there."
        )

    documents: List[Document] = []
    for file in pdf_files:
        path = os.path.join(GUIDELINES_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        gtype = infer_guideline_type(file)
        for d in docs:
            d.metadata["source_doc"] = file
            d.metadata["guideline_type"] = gtype
            # PyPDFLoader typically sets `page` (0-indexed). Keep as-is, we’ll display as +1 later.
        documents.extend(docs)

    return documents


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


def _open_vectorstore() -> Chroma:
    embeddings = _get_embedding_function()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def build_knowledge_base(force_rebuild: bool = True) -> None:
    """
    Build a persistent Chroma DB from the guideline PDFs.

    - On rebuild, clears CHROMA_DIR to avoid duplicates.
    - Stores all guidelines in one collection with metadata filters.
    """
    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    docs = _load_guideline_docs()
    chunks = _split_docs(docs)

    # Stable IDs help avoid accidental duplication if you ever switch force_rebuild=False.
    ids: List[str] = []
    for i, d in enumerate(chunks):
        src = d.metadata.get("source_doc", "unknown.pdf")
        page = d.metadata.get("page", "na")
        gtype = d.metadata.get("guideline_type", "other")
        ids.append(f"{gtype}:{src}:p{page}:c{i}")

    vs = _open_vectorstore()
    vs.add_documents(chunks, ids=ids)

    # Quick validation retrievals
    tested_types = ["statistics", "figures_tables", "causality", "systematic_meta"]
    for gtype in tested_types:
        _ = retrieve_guidelines_by_type(gtype, "test", k=1)

    print(f"✅ Knowledge base built. Stored {len(chunks)} chunks in Chroma ({CHROMA_DIR}).")


def retrieve_guidelines_by_type(
    guideline_type: str,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve top-k guideline chunks for a given guideline_type using Chroma metadata filtering.
    Uses MMR for better diversity/coverage.
    """
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            "Chroma DB not found. Build/validate knowledge base from the sidebar first."
        )

    vs = _open_vectorstore()

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(20, k * 4),
            "filter": {"guideline_type": guideline_type},
        },
    )
    return retriever.invoke(query)
