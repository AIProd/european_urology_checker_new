# indexer.py
import os
import shutil
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


def build_knowledge_base(force_rebuild: bool = True) -> None:
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

        total = None
        type_counts: Dict[str, int] = {}

        # Best effort count
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
