# RAG Document Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

A **Retrieval-Augmented Generation (RAG)** application that lets you chat with your documents. Upload **PDF, DOCX, or TXT** files and ask questions — answers are grounded **strictly** in the document content, not hallucinated.

---

## What This Project Demonstrates

- End-to-end RAG pipeline: document loading → chunking → embedding → retrieval → generation
- Two interfaces: a polished **Streamlit web app** and a lightweight **CLI tool**
- Practical use of **LangChain**, **FAISS** (CPU vector search), and **Google Gemini 2.5 Flash**
- Secure API key handling via Streamlit secrets or environment variables
- Graceful error handling for missing keys, bad files, API quota issues, and injection attempts

---

## Architecture

```
User Query
    │
    ▼
[FAISS Vector Store]  ←─── Uploaded Document
    │                         └─ Loaded → Split → Embedded (text-embedding-004)
    │  similarity_search(k=4)
    ▼
[Relevant Chunks]
    │
    ▼
[Gemini 2.5 Flash]  ← strict "answer from context only" prompt
    │
    ▼
[Grounded Answer]
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kpradyun/RAG.git
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Google API Key

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

**Option A — Streamlit secrets (recommended for web app):**

```bash
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "AIzaSy..."' > .streamlit/secrets.toml
```

**Option B — Environment variable:**

```bash
export GOOGLE_API_KEY="AIzaSy..."      # macOS/Linux
set GOOGLE_API_KEY=AIzaSy...           # Windows CMD
```

**Option C — `.env` file (CLI mode):**

```bash
echo 'GOOGLE_API_KEY=AIzaSy...' > .env
```

**Option D — Enter in app sidebar** (web mode only, no file needed).

---

## Running the App

### Web Interface (recommended)

```bash
streamlit run web_chat.py
```

Opens at `http://localhost:8501` in your browser.

### Terminal / CLI Mode

```bash
python rag_app.py                     # interactive prompt for file path
python rag_app.py path/to/file.pdf    # pass file directly
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web UI | Streamlit |
| AI Model | Google Gemini 2.5 Flash |
| Embeddings | Google text-embedding-004 |
| Vector DB | FAISS (in-memory, CPU) |
| RAG Framework | LangChain |
| Document Loaders | PyPDF, Docx2txt, TextLoader |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY not set` | Set the key as described in Step 4 above |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `API quota exceeded` | Check your quota at [Google Cloud Console](https://console.cloud.google.com/apis/dashboard) |
| Answers seem off | Try re-uploading; very scanned/image PDFs may need OCR pre-processing |
| PDF loads but no text | The PDF may be image-based — convert with a tool like `pdf2image` + `pytesseract` |

---

## Limitations

- No persistent storage — documents are processed in memory per session
- Image-based / scanned PDFs not supported (text extraction only)
- Context window is limited to top-k retrieved chunks per query
- No conversation memory across page reloads in web mode

---

## License

MIT © [kpradyun](https://github.com/kpradyun)
