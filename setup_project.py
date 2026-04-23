"""
========================================================
  Das Kapital Voice-AI Agent — Project Setup Script
  Groq (llama-3.3-70b) + HuggingFace Embeddings edition
  Run this ONCE to generate ingest.py, main.py, .env
========================================================
"""

import os

# ── ingest.py ─────────────────────────────────────────────────────────────────
INGEST_PY = '''\
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
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH    = "das_kapital.pdf"
DB_DIR      = "marx_db"
# 384-dim, 22 MB model — fast on CPU, excellent semantic search quality
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
            f"\\n[ERROR] Cannot find \'{path}\'.\\n"
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
    Child  chunks  (300 chars) : small, precise targets for vector similarity search.
    Parent chunks  (semantic)  : full logical arguments returned to the LLM.

    SemanticChunker uses cosine-distance between consecutive sentences to detect
    natural topic boundaries — critical for dense 19th-century political economy text.
    The LLM receives the full parent argument, not an isolated fragment.
    """
    log.info("Building SemanticChunker for parent splits ...")
    parent_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,    # split only at top-15% dissimilarity jumps
    )

    log.info("Building RecursiveCharacterTextSplitter for child chunks ...")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
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

    log.info("Embedding 547 pages — expect 5-15 min on CPU. Do NOT interrupt.")
    retriever.add_documents(docs, ids=None)

    docstore_path = os.path.join(DB_DIR, "docstore.pkl")
    with open(docstore_path, "wb") as f:
        pickle.dump(dict(docstore.store), f)

    count = vectorstore._collection.count()
    log.info(f"Done. {count} child chunks saved to \'{DB_DIR}/\'.")
    log.info(f"Parent docs saved to \'{docstore_path}\'.")


def main():
    docs       = load_pdf(PDF_PATH)
    embeddings = build_embeddings()
    ingest(docs, embeddings)
    log.info("\\n SUCCESS  marx_db is ready. Now run:  python main.py")


if __name__ == "__main__":
    main()
'''

# ── main.py ───────────────────────────────────────────────────────────────────
MAIN_PY = '''\
"""
main.py  —  Phase 2: Voice-AI Agent
=====================================
STT  : faster-whisper  (CPU / int8 — no GPU needed)
RAG  : ChromaDB + Parent-Document Retrieval
LLM  : Groq API  — llama-3.3-70b-versatile
TTS  : edge-tts  (Microsoft Neural voices, free, no API key)

Run AFTER ingest.py:
    python main.py
"""

import os
import sys
import asyncio
import logging
import pickle
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import edge_tts

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_DIR          = "marx_db"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
WHISPER_MODEL   = "base"        # options: tiny | base | small
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"        # quantised for CPU — ~4x faster, negligible accuracy loss
TTS_VOICE       = "en-GB-RyanNeural"    # scholarly British male voice
SAMPLE_RATE     = 16000
SILENCE_THRESH  = 0.01          # RMS below this = silence; raise if mic is noisy
SILENCE_SECS    = 2.0
MAX_RECORD_SECS = 30

SYSTEM_PROMPT = (
    "You are a scholarly authority on Marxian economics. "
    "Your answers are precise, rigorous, and strictly grounded in "
    "the text of Das Kapital, Volume I by Karl Marx. "
    "Cite specific parts, chapters, or sections when possible. "
    "Do not speculate beyond what Marx wrote. "
    "If the retrieved context does not contain a clear answer, say so honestly."
)

# {context} and {question} are filled in at runtime by RetrievalQA
QA_PROMPT_TEMPLATE = (
    SYSTEM_PROMPT + "\\n\\n"
    "Retrieved context from Das Kapital:\\n"
    "\\"\\"\\"{context}\\"\\"\\"\\n\\n"
    "Question: {question}\\n\\n"
    "Answer (as a Marxian scholar):"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validate API key
# ─────────────────────────────────────────────────────────────────────────────
def validate_env():
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        sys.exit(
            "\\n[ERROR] GROQ_API_KEY is missing or still set to the placeholder.\\n"
            "Open .env and replace  your_groq_api_key_here  with your real key.\\n"
            "Get a free key at: https://console.groq.com\\n"
        )
    log.info("  GROQ_API_KEY loaded successfully.")
    return key


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Retriever from persisted ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
def load_retriever():
    if not Path(DB_DIR).exists():
        sys.exit(
            f"\\n[ERROR] \'{DB_DIR}\' not found.\\n"
            "Run  python ingest.py  first to build the knowledge base.\\n"
        )

    log.info("Loading HuggingFace embeddings model ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    log.info("Loading ChromaDB vector store ...")
    vectorstore = Chroma(
        collection_name="das_kapital_children",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    log.info("Loading parent docstore ...")
    docstore      = InMemoryStore()
    docstore_path = os.path.join(DB_DIR, "docstore.pkl")
    if Path(docstore_path).exists():
        with open(docstore_path, "rb") as f:
            docstore.store.update(pickle.load(f))
        log.info(f"  Loaded {len(docstore.store)} parent documents.")
    else:
        log.warning("  docstore.pkl not found — falling back to child-chunk retrieval only.")

    # Splitter config MUST match ingest.py exactly
    parent_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5},
    )
    log.info("  Retriever ready.")
    return retriever


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build RetrievalQA chain — Groq LLM
# ─────────────────────────────────────────────────────────────────────────────
def build_qa_chain(retriever):
    log.info(f"Connecting to Groq: {LLM_MODEL} ...")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,        # low = faithful to text, less hallucination
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    log.info("  QA chain ready.")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 4. STT — faster-whisper
# ─────────────────────────────────────────────────────────────────────────────
def load_whisper():
    log.info(f"Loading Whisper model ({WHISPER_MODEL} / {WHISPER_COMPUTE}) ...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    log.info("  Whisper ready.")
    return model


def record_until_silence() -> np.ndarray:
    log.info("Listening  (speak now — pause 2s to finish)")
    frames        = []
    silent_chunks = 0
    chunk_size    = int(SAMPLE_RATE * 0.1)           # 100ms windows
    silence_limit = int(SILENCE_SECS / 0.1)          # chunks of silence needed

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=chunk_size,
    ) as stream:
        for _ in range(int(MAX_RECORD_SECS / 0.1)):
            chunk, _ = stream.read(chunk_size)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            frames.append(chunk)
            if rms < SILENCE_THRESH:
                silent_chunks += 1
                if silent_chunks >= silence_limit and len(frames) > silence_limit:
                    break
            else:
                silent_chunks = 0

    audio = np.concatenate(frames, axis=0).flatten()
    log.info(f"  Captured {len(audio)/SAMPLE_RATE:.1f}s of audio.")
    return audio


def transcribe(whisper_model, audio: np.ndarray) -> str:
    segments, _ = whisper_model.transcribe(
        audio, beam_size=5, language="en",
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    log.info(f"  You said: \\"{text}\\"")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 5. TTS — edge-tts
# ─────────────────────────────────────────────────────────────────────────────
async def _speak_async(text: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    await edge_tts.Communicate(text, TTS_VOICE).save(tmp_path)
    data, sr = sf.read(tmp_path)
    sd.play(data, sr)
    sd.wait()
    os.unlink(tmp_path)


def speak(text: str):
    log.info("Speaking ...")
    asyncio.run(_speak_async(text))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main conversation loop
# ─────────────────────────────────────────────────────────────────────────────
def print_sources(source_docs):
    if source_docs:
        pages = sorted({d.metadata.get("page", "?") for d in source_docs})
        log.info(f"  Source pages from Das Kapital: {pages}")


def main():
    validate_env()

    print("\\n" + "=" * 60)
    print("  Das Kapital Voice-AI Agent")
    print("  Groq llama-3.3-70b  |  ChromaDB  |  Whisper  |  edge-tts")
    print("  Say \\'exit\\' or \\'quit\\' to stop.")
    print("=" * 60 + "\\n")

    retriever = load_retriever()
    qa_chain  = build_qa_chain(retriever)
    whisper   = load_whisper()

    speak("I am ready. Ask me anything about Das Kapital.")

    while True:
        try:
            audio    = record_until_silence()
            question = transcribe(whisper, audio)

            if not question:
                log.info("  (no speech detected — listening again)")
                continue

            if any(w in question.lower() for w in ["exit", "quit", "stop"]):
                speak("Farewell. The contradictions of capital persist.")
                break

            log.info("  Querying Groq via RAG ...")
            result = qa_chain.invoke({"query": question})
            answer = result["result"].strip()
            print_sources(result.get("source_documents", []))

            print(f"\\n{'─' * 60}\\n{answer}\\n{'─' * 60}\\n")
            speak(answer)

        except KeyboardInterrupt:
            speak("Session interrupted.")
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            speak("I encountered an error. Please try again.")


if __name__ == "__main__":
    main()
'''

# ── requirements.txt ─────────────────────────────────────────────────────────
REQUIREMENTS = """\
# ── Core LangChain ───────────────────────────────────────────────────────────
langchain>=0.3
langchain-community>=0.3
langchain-chroma>=0.1
langchain-experimental>=0.3
chromadb>=0.5

# ── LLM: Groq cloud API ──────────────────────────────────────────────────────
langchain-groq>=0.1

# ── Embeddings: local HuggingFace (no API key needed) ───────────────────────
langchain-huggingface>=0.1
sentence-transformers>=3.0

# ── PDF loading ───────────────────────────────────────────────────────────────
pypdf>=4.0

# ── Speech-to-Text ────────────────────────────────────────────────────────────
faster-whisper>=1.0

# ── Text-to-Speech ────────────────────────────────────────────────────────────
edge-tts>=6.1

# ── Audio I/O ────────────────────────────────────────────────────────────────
sounddevice>=0.4
soundfile>=0.12
numpy>=1.26

# ── Environment variables ────────────────────────────────────────────────────
python-dotenv>=1.0
"""

# ── .env template ─────────────────────────────────────────────────────────────
DOT_ENV = """\
# Groq API Key
# Get a FREE key at: https://console.groq.com
# NEVER commit this file to Git — it is already in .gitignore
GROQ_API_KEY=your_groq_api_key_here
"""

# ── .gitignore ────────────────────────────────────────────────────────────────
GITIGNORE = """\
# Secrets
.env

# Generated knowledge base (large, re-creatable)
marx_db/

# Python cache
__pycache__/
*.pyc
.venv/
venv/
"""

# ── Write all files ───────────────────────────────────────────────────────────
files = {
    "ingest.py":        INGEST_PY,
    "main.py":          MAIN_PY,
    "requirements.txt": REQUIREMENTS,
    ".env":             DOT_ENV,
    ".gitignore":       GITIGNORE,
}

for filename, content in files.items():
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {filename}")

print("""
╔══════════════════════════════════════════════════════════╗
║         Project files generated successfully!            ║
╠══════════════════════════════════════════════════════════╣
║  STEP 1 (critical): Open .env — paste your GROQ_API_KEY  ║
║  STEP 2: pip install -r requirements.txt                 ║
║  STEP 3: Copy das_kapital.pdf to this folder             ║
║  STEP 4: python ingest.py    (once — builds marx_db)     ║
║  STEP 5: python main.py      (starts the agent)          ║
╚══════════════════════════════════════════════════════════╝
""")
