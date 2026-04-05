"""
rag.py — RAG Knowledge Base Module
=====================================

Embeds defect-standard documents from ``data/knowledge_base/`` into a FAISS
index and provides ``retrieve_context(query)`` for the LLM agent.

The FAISS index is cached to ``data/faiss_index/`` so subsequent app restarts
skip the (slow) embedding step.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

KB_DIR    = Path("data/knowledge_base")
INDEX_DIR = Path("data/faiss_index")

# ── Module-level index cache ───────────────────────────────────────────────────

_retriever = None


def _build_retriever():
    """Load or build the FAISS retriever."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load from saved index if available
    if INDEX_DIR.exists():
        print("[rag] Loading FAISS index from cache …")
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        ).as_retriever(search_kwargs={"k": 3})

    # Build index from knowledge base documents
    print("[rag] Building FAISS index from knowledge base …")
    loader = DirectoryLoader(
        str(KB_DIR),
        glob="**/*.{md,txt}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"[rag] Index saved → {INDEX_DIR}")

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant knowledge-base excerpts for a query string.

    Returns a plain-text string of the top-k chunks, ready to inject into
    the LLM prompt. Returns a fallback string if the knowledge base is missing.
    """
    global _retriever

    if not KB_DIR.exists() or not any(KB_DIR.iterdir()):
        return "No knowledge base available. Use standard defect inspection protocols."

    try:
        if _retriever is None:
            _retriever = _build_retriever()
        docs = _retriever.invoke(query)
        return "\n\n---\n\n".join(d.page_content for d in docs)
    except Exception as exc:
        print(f"[rag] Retrieval failed: {exc}")
        return "Knowledge base retrieval unavailable. Apply standard inspection criteria."
