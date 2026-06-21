# RAG Document Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** application that lets you chat with your documents.  
Upload **PDF, DOCX, or TXT** files and ask questions strictly based on their content.

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
