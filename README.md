# RAG Document Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** application that lets you chat with your documents.  
Upload **PDF, DOCX, or TXT** files and ask questions strictly based on their content.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kpradyun/rag-document-analyzer.git
cd rag-document-analyzer

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Configuration

Get your API Key from **Google AI Studio**.

### Option A (Recommended): Create `.streamlit/secrets.toml`

```toml
GOOGLE_API_KEY = "AIzaSy..."
```

### Option B: Manual Entry

Enter the API key manually in the app sidebar when running.

---

### 3. Run the App

### Web Interface

```bash
streamlit run web_app.py
```

### Terminal Mode

```bash
python rag_app.py
```

##  Tech Stack

- **Language:** Python 3.10+
- **AI Model:** Google Gemini 1.5 Flash
- **Vector DB:** FAISS (CPU)
- **Frameworks:** LangChain, Streamlit
