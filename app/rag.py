"""
rag.py — RAG Knowledge Base Module
=====================================

This module manages the Retrieval-Augmented Generation (RAG) pipeline that
grounds the LLM agent's recommendations in documented defect standards.

Pipeline position:
    detect.py → agent.py ↔ [rag.py]

What is RAG and why use it here?
    Without RAG, the LLM generates recommended actions from its training data
    alone. For industrial inspection this is a problem: the model may produce
    plausible-sounding but incorrect or context-free recommendations.

    With RAG, the agent retrieves the most relevant excerpts from
    ``data/knowledge_base/`` — which contains our curated defect classification
    standards — and injects them directly into the prompt. This means:
      - Recommended actions reference real documented standards
      - Adding new defect types only requires updating the knowledge base files
      - No LLM retraining needed when standards change
      - The provenance of each recommendation is auditable

How it works:
    1. ``build_index()`` loads all ``.txt`` and ``.md`` files from the
       knowledge base directory, splits them into overlapping text chunks,
       and embeds each chunk using a local sentence-transformer model.
       The resulting FAISS index is saved to disk so subsequent restarts
       load from cache rather than re-embedding from scratch.
    2. ``retrieve_context()`` embeds the query string using the same
       sentence-transformer, performs a cosine similarity search against
       the FAISS index, and returns the top-k most relevant chunks as a
       concatenated string for injection into the LLM prompt.

Embedding model:
    ``sentence-transformers/all-MiniLM-L6-v2`` (~80 MB) runs entirely on CPU
    with no GPU required. It produces 384-dimensional embeddings and achieves
    a strong balance between retrieval quality and inference speed.

FAISS:
    Facebook AI Similarity Search (FAISS) provides efficient approximate
    nearest-neighbour search for dense vector embeddings. The ``faiss-cpu``
    build requires no GPU and supports indexes with thousands of documents
    without performance issues on standard hardware.

Persistence:
    The FAISS index is saved to ``models/faiss_index/`` after first build.
    Both the index files and the embedding model weights are in ``.gitignore``
    — they are derived artefacts, not source code.

Dependencies:
    langchain-community        — DirectoryLoader, FAISS wrapper, HuggingFaceEmbeddings
    langchain-text-splitters   — RecursiveCharacterTextSplitter
    faiss-cpu                  — Vector similarity search
    sentence-transformers      — Local embedding model
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────────────────────────

# Directory containing knowledge base documents (defect standards, etc.)
# Add any .txt or .md file here to extend the RAG knowledge base.
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")

# Where the serialised FAISS index is persisted between runs.
# Avoids re-embedding on every startup (embedding takes ~10s on first run).
FAISS_INDEX_PATH = Path("models/faiss_index")

# Sentence-transformer model for embedding both documents and queries.
# Must be the same model for both — different models produce incompatible
# embedding spaces and will cause poor retrieval quality.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitter configuration.
# CHUNK_SIZE: maximum characters per chunk. 512 chars ≈ 100–120 words, which
#   is large enough to contain a full defect description with context.
# CHUNK_OVERLAP: characters shared between adjacent chunks. 64-char overlap
#   prevents a relevant sentence being split across two chunks and missing
#   retrieval.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Number of chunks to retrieve per query.
# 3 chunks provides enough context for the LLM without overloading the prompt.
TOP_K = 3


# ── Internals ─────────────────────────────────────────────────────────────────

# Module-level FAISS vectorstore cache.
# None until first call to build_index(). Prevents repeated disk I/O.
_vectorstore: FAISS | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialise the local sentence-transformer embedding model.

    The model is downloaded from Hugging Face Hub on first use (~80 MB).
    Subsequent calls reuse the cached model weights from disk.

    ``device: "cpu"`` ensures the embedding model runs on CPU regardless of
    GPU availability — we want GPU resources free for YOLOv8 inference.

    ``normalize_embeddings: True`` L2-normalises the output vectors, which
    makes cosine similarity equivalent to dot product and slightly improves
    retrieval quality with FAISS.

    Returns:
        Configured ``HuggingFaceEmbeddings`` instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> FAISS:
    """
    Build (or load from disk) the FAISS vector index over knowledge base docs.

    Call order and caching:
        1. If ``_vectorstore`` is already loaded in memory, return it.
           (Normal path during a running session — zero overhead.)
        2. If ``FAISS_INDEX_PATH`` exists on disk, load from there.
           (Normal path on app restart — fast, ~1 second.)
        3. Otherwise, load all ``.txt``/``.md`` files from
           ``KNOWLEDGE_BASE_DIR``, split, embed, and build from scratch.
           Save the result to ``FAISS_INDEX_PATH`` for next time.
           (First-run path — takes ~10–30 seconds depending on hardware.)

    Args:
        force_rebuild: If ``True``, skip both caches and rebuild from source
                       documents. Use this after adding new files to the
                       knowledge base directory.

    Returns:
        Ready-to-query ``FAISS`` vectorstore instance.

    Raises:
        FileNotFoundError: If the knowledge base directory is empty and no
                           cached index exists. Add ``.txt`` or ``.md`` files
                           to ``data/knowledge_base/`` before calling.

    Example::

        # Force a rebuild after adding new defect standard documents
        from app.rag import build_index
        vs = build_index(force_rebuild=True)
        print(f"Index contains {vs.index.ntotal} vectors")
    """
    global _vectorstore

    # Level 1 cache: already loaded in memory.
    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    # Level 2 cache: serialised FAISS index on disk.
    if FAISS_INDEX_PATH.exists() and not force_rebuild:
        print("[rag] Loading FAISS index from disk (use force_rebuild=True to refresh)...")
        _vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            _get_embeddings(),
            # allow_dangerous_deserialization is required by FAISS.load_local
            # when loading a pickle-based index. Safe here because we wrote it.
            allow_dangerous_deserialization=True,
        )
        return _vectorstore

    # Level 3: build from scratch.
    print("[rag] Building FAISS index from knowledge base documents...")

    # Load all supported document types from the knowledge base directory.
    # Python's glob does NOT support bash-style brace expansion like {txt,md},
    # so we run two separate DirectoryLoader passes and merge the results.
    docs = []
    for pattern in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            str(KNOWLEDGE_BASE_DIR),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(loader.load())

    if not docs:
        raise FileNotFoundError(
            f"No .md or .txt files found in: {KNOWLEDGE_BASE_DIR}\n"
            "Check that data/knowledge_base/defect_standards.md exists."
        )

    # Split documents into overlapping chunks.
    # RecursiveCharacterTextSplitter splits on paragraph breaks first, then
    # sentences, then words — always trying to keep semantically coherent units.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[rag] Loaded {len(docs)} documents → {len(chunks)} chunks after splitting.")

    # Embed all chunks and build FAISS index.
    embeddings = _get_embeddings()
    _vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk for fast reload on next startup.
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"[rag] Index saved to '{FAISS_INDEX_PATH}' ({_vectorstore.index.ntotal} vectors).")

    return _vectorstore


def retrieve_context(query: str, k: int = TOP_K) -> str:
    """
    Retrieve the most relevant knowledge base excerpts for a defect query.

    Embeds ``query`` using the same sentence-transformer used at index-build
    time, performs cosine similarity search against the FAISS index, and
    returns the top-k chunks as a single string separated by horizontal rules.

    This string is injected verbatim into the LLM agent's prompt as the
    "Relevant Standards & Context" section.

    Args:
        query: Natural language description of what to look up.
               Typically constructed by ``agent.py`` as:
               ``"defect types: scratch, pit"``
        k:     Number of chunks to retrieve. Default is ``TOP_K`` (3).
               Increasing this provides more context but lengthens the prompt,
               which can reduce LLM output quality on smaller models.

    Returns:
        A string of the top-k retrieved chunks joined by ``---`` separators.
        Returns a fallback string if the index returns no results (e.g. for
        a query about a defect type not covered in the knowledge base).

    Example::

        from app.rag import retrieve_context
        context = retrieve_context("scratch defect on metal surface")
        print(context[:300])  # First 300 chars of retrieved context
    """
    vs = build_index()
    docs = vs.similarity_search(query, k=k)

    if not docs:
        # Graceful degradation: the agent will still generate a report,
        # just without standards-grounded recommended actions.
        return "No relevant context found in knowledge base for this defect type."

    # Join chunks with a separator so the LLM can distinguish boundaries.
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
