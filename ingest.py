"""
ingest.py  —  Phase 1: Knowledge Ingestion
===========================================
Reads das_kapital.pdf, chunks it semantically,
embeds with HuggingFace all-MiniLM-L6-v2 (runs 100% locally, no API needed),
and persists child vectors + parent docs to ChromaDB (marx_db/).

Run ONCE before main.py:
    python ingest.py
"""

import os
import logging
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH    = "das_kapital.pdf"
DB_DIR      = "marx_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 50       # pages per batch — stays well under ChromaDB's 5461 chunk limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_pdf(path: str):
    log.info(f"Loading PDF: {path}")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"\n[ERROR] Cannot find '{path}'.\n"
            "Place das_kapital.pdf in the same folder as ingest.py."
        )
    docs = PyPDFLoader(path).load()
    log.info(f"  Loaded {len(docs)} pages.")
    return docs


def build_embeddings():
    log.info(f"Loading HuggingFace embedding model: {EMBED_MODEL}")
    log.info("  (First run downloads ~22 MB — cached after that)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("  Embedding model ready.")
    return embeddings


def ingest(docs, embeddings):
    """
    Parent-Document Retrieval Strategy
    ------------------------------------
    Child  chunks  (300 chars)  : small, precise targets for vector similarity search.
    Parent chunks  (1500 chars) : larger context windows returned to the LLM.

    Documents are added in batches of BATCH_SIZE pages to stay within
    ChromaDB's internal batch size limit of 5461 vectors.
    """
    log.info("Building parent splitter (1500 char chunks) ...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    log.info("Building child splitter (300 char chunks) ...")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    log.info("Initialising ChromaDB ...")
    os.makedirs(DB_DIR, exist_ok=True)
    vectorstore = Chroma(
        collection_name="das_kapital_children",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    docstore  = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        child_metadata_fields=["page", "source"],
    )

    total   = len(docs)
    batches = range(0, total, BATCH_SIZE)
    log.info(f"Embedding {total} pages in batches of {BATCH_SIZE} — expect 5-15 min on CPU. Do NOT interrupt.")

    for i, start in enumerate(batches):
        batch = docs[start : start + BATCH_SIZE]
        log.info(f"  Batch {i+1}/{len(batches)}  (pages {start+1}–{min(start+BATCH_SIZE, total)})")
        retriever.add_documents(batch, ids=None)

    docstore_path = os.path.join(DB_DIR, "docstore.pkl")
    with open(docstore_path, "wb") as f:
        pickle.dump(dict(docstore.store), f)

    count = vectorstore._collection.count()
    log.info(f"Done. {count} child chunks saved to '{DB_DIR}/'.")
    log.info(f"Parent docs saved to '{docstore_path}'.")


def main():
    docs       = load_pdf(PDF_PATH)
    embeddings = build_embeddings()
    ingest(docs, embeddings)
    log.info("\n SUCCESS  marx_db is ready. Now run:  python main.py")


if __name__ == "__main__":
    main()
