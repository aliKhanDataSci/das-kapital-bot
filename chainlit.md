# Das Kapital — Marxian Scholar AI 📖

Welcome to the **Das Kapital Voice-AI Agent**, a RAG-powered chatbot grounded entirely in the text of *Das Kapital, Volume I* by Karl Marx.

## How to use

- **Ask any question** about Marx's economic theory — commodities, surplus value, labour, capital accumulation, etc.
- **Follow-up questions** are supported — the AI remembers your conversation.
- Type `/voice on` to enable **spoken audio** replies (edge-tts).
- Type `/voice off` to disable voice.

## Powered by

| Component | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Vector DB | ChromaDB + Parent-Document Retrieval |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| TTS | Microsoft `edge-tts` (en-GB-RyanNeural) |
| UI | Chainlit |

## Source transparency

Every answer includes the **page numbers** from Das Kapital that were used to generate the response.
