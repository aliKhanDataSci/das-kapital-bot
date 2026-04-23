"""
app.py  —  Das Kapital Web Chatbot (Chainlit)
=============================================
Run with:
    chainlit run app.py
"""

import os
import asyncio
import logging
import pickle
import tempfile
from pathlib import Path

import edge_tts
import chainlit as cl
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
DB_DIR      = "marx_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"
TTS_VOICE   = "en-GB-RyanNeural"

CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the conversation history and a follow-up question, "
    "rephrase the follow-up question to be a standalone question.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a scholarly authority on Marxian economics. "
        "Your answers are precise, rigorous, and strictly grounded in "
        "the text of Das Kapital, Volume I by Karl Marx. "
        "Cite specific parts, chapters, or sections when possible. "
        "Do not speculate beyond what Marx wrote. "
        "If the retrieved context does not contain a clear answer, say so honestly.\n\n"
        "Retrieved context from Das Kapital:\n"
        "\"\"\"{context}\"\"\"\n\n"
        "Question: {question}\n\n"
        "Answer (as a Marxian scholar):"
    ),
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Blocking functions (called via asyncio.to_thread) ─────────────────────────

def _build_retriever() -> ParentDocumentRetriever:
    if not Path(DB_DIR).exists():
        raise RuntimeError(f"'{DB_DIR}' not found. Run  python ingest.py  first.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name="das_kapital_children",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    docstore = InMemoryStore()
    docstore_path = os.path.join(DB_DIR, "docstore.pkl")
    if Path(docstore_path).exists():
        with open(docstore_path, "rb") as f:
            docstore.store.update(pickle.load(f))
        log.info(f"Loaded {len(docstore.store)} parent documents.")
    else:
        log.warning("docstore.pkl not found — child-chunk fallback only.")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5},
    )


def _build_chain(retriever: ParentDocumentRetriever) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        streaming=False,   # keep False — streaming with thread pool causes event loop conflicts
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_PROMPT,
        verbose=False,
    )


def _invoke_chain(chain: ConversationalRetrievalChain, question: str) -> dict:
    return chain.invoke({"question": question})


# ── Async TTS ──────────────────────────────────────────────────────────────────

async def generate_audio(text: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    await edge_tts.Communicate(text, TTS_VOICE).save(tmp_path)
    return tmp_path


# ── Chainlit handlers ──────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    # Validate API key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        await cl.Message(
            content=(
                "⚠️ **GROQ_API_KEY is missing.**\n\n"
                "Add your key to the `.env` file:\n"
                "```\nGROQ_API_KEY=your_real_key_here\n```\n"
                "Get a free key at https://console.groq.com"
            )
        ).send()
        return

    loading = cl.Message(content="⚙️ Loading Das Kapital knowledge base, please wait…")
    await loading.send()

    try:
        # asyncio.to_thread is the modern, safe way to run blocking code
        retriever = await asyncio.to_thread(_build_retriever)
        chain     = await asyncio.to_thread(_build_chain, retriever)
    except RuntimeError as e:
        await loading.remove()
        await cl.Message(content=f"❌ **Error:** {e}").send()
        return

    cl.user_session.set("chain", chain)
    cl.user_session.set("voice_enabled", False)

    await loading.remove()
    await cl.Message(
        content=(
            "# 📖 Das Kapital — Marxian Scholar AI\n\n"
            "I am ready to answer questions about **Das Kapital, Volume I** "
            "by Karl Marx, grounded strictly in the text.\n\n"
            "**Tips:**\n"
            "- Ask about value, capital, labour, commodities, surplus value, etc.\n"
            "- Follow-up questions are supported — I remember the conversation.\n"
            "- Type `/voice on` to enable spoken audio replies.\n"
            "- Type `/voice off` to disable voice.\n\n"
            "*\"Capital is dead labour, which, vampire-like, only lives by sucking living labour.\"*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Voice toggle commands
    cmd = message.content.strip().lower()
    if cmd in ("/voice on", "voice on"):
        cl.user_session.set("voice_enabled", True)
        await cl.Message(content="🔊 Voice responses **enabled**.").send()
        return
    if cmd in ("/voice off", "voice off"):
        cl.user_session.set("voice_enabled", False)
        await cl.Message(content="🔇 Voice responses **disabled**.").send()
        return

    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="❌ Session not initialised. Please refresh the page.").send()
        return

    voice_enabled = cl.user_session.get("voice_enabled", False)

    # Show thinking indicator
    thinking = cl.Message(content="🔍 *Searching Das Kapital…*")
    await thinking.send()

    # Run blocking chain call safely in a background thread
    try:
        result = await asyncio.to_thread(_invoke_chain, chain, message.content)
    except Exception as e:
        await thinking.remove()
        await cl.Message(content=f"❌ Error querying the chain: {e}").send()
        log.error(f"Chain error: {e}", exc_info=True)
        return

    await thinking.remove()

    answer      = result.get("answer", "").strip()
    source_docs = result.get("source_documents", [])

    # Send the answer
    await cl.Message(content=answer).send()

    # Source citations
    if source_docs:
        pages    = sorted({str(d.metadata.get("page", "?")) for d in source_docs})
        snippets = []
        for doc in source_docs[:3]:
            page    = doc.metadata.get("page", "?")
            preview = doc.page_content[:200].replace("\n", " ").strip()
            snippets.append(f"**Page {page}:** …{preview}…")

        await cl.Message(
            content=(
                f"📚 **Sources — Das Kapital, Vol. I** *(pages {', '.join(pages)})*\n\n"
                + "\n\n".join(snippets)
            ),
            author="Sources",
        ).send()

    # Voice output
    if voice_enabled and answer:
        try:
            audio_path = await generate_audio(answer)
            await cl.Message(
                content="🔊 *Audio response:*",
                elements=[cl.Audio(name="answer.mp3", path=audio_path, display="inline")],
            ).send()
        except Exception as e:
            log.error(f"TTS error: {e}")
            await cl.Message(content=f"⚠️ Audio failed: {e}").send()
