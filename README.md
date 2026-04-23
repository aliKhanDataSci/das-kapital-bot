# 📖 Das Kapital — Marxian Scholar AI

A RAG-powered chatbot grounded entirely in the text of *Das Kapital, Volume I* by Karl Marx.  
Ask questions about commodities, surplus value, capital, labour, and more — with source citations and optional voice output.

---

## ✨ Features

- **Retrieval-Augmented Generation** via ChromaDB + Parent-Document Retrieval
- **Local embeddings** — HuggingFace `all-MiniLM-L6-v2` (no extra API key needed)
- **LLM** — Groq `llama-3.3-70b-versatile` (free tier available)
- **Text-to-Speech** — Microsoft `edge-tts` (en-GB-RyanNeural)
- **Conversational memory** — follow-up questions are supported
- **Page citations** — every answer shows which pages of Das Kapital were used

---

## 🖥️ Tech Stack

| Component   | Technology                              |
|-------------|------------------------------------------|
| LLM         | Groq `llama-3.3-70b-versatile`           |
| Vector DB   | ChromaDB + Parent-Document Retrieval     |
| Embeddings  | HuggingFace `all-MiniLM-L6-v2`          |
| TTS         | Microsoft `edge-tts` (en-GB-RyanNeural) |
| UI          | Chainlit                                 |

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/aliKhanDataSci/das-kapital-bot.git
cd das-kapital-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your [Groq API key](https://console.groq.com):

```
GROQ_API_KEY=your_real_key_here
```

### 5. Add the PDF

Place `das_kapital.pdf` (Volume I) in the project root.  
A public-domain version is available on [Project Gutenberg](https://www.gutenberg.org/) and [Archive.org](https://archive.org/).

### 6. Ingest the knowledge base (run once)

```bash
python ingest.py
```

This embeds the entire book into ChromaDB (`marx_db/`).  
Expect **5–15 minutes** on CPU. Do not interrupt.

### 7. Launch the chatbot

```bash
chainlit run app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 💬 Usage Tips

| Command | Effect |
|---------|--------|
| `/voice on` | Enable spoken audio replies |
| `/voice off` | Disable voice |
| Ask follow-up questions freely | Conversation memory is active |

---

## 📁 Project Structure

```
das-kapital-ai/
├── app.py            # Chainlit web app
├── ingest.py         # One-time PDF ingestion script
├── chainlit.md       # Chainlit welcome screen
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore
└── README.md
```

> `marx_db/` and `.env` are excluded from Git — see `.gitignore`.

---

## ⚠️ Notes

- The `marx_db/` folder is generated locally and is **not** committed to Git (it can be several hundred MB).
- Your `.env` file is **never** committed — use `.env.example` as a safe template.
- The project runs entirely on CPU; no GPU required.

---

## 📜 License

This project is released under the **MIT License**.  
*Das Kapital* (1867) is in the public domain.
