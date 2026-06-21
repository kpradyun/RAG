"""
web_chat.py — Streamlit web interface for the RAG Document Analyzer.

Run:
    streamlit run web_chat.py
"""

import os
import tempfile
import streamlit as st
from typing import Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_HISTORY_PAIRS = 5  # how many prior Q&A turns to pass as context

# =========================================================
# API KEY HANDLING  (secrets → env var → sidebar input)
# =========================================================
def _resolve_api_key() -> bool:
    """Returns True if a key is available, False otherwise."""
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        return True
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Get yours at https://aistudio.google.com/app/apikey",
        placeholder="AIzaSy...",
    )
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return True
    return False


api_ready = _resolve_api_key()
if not api_ready:
    st.sidebar.warning("⚠️ Please enter your Google API key to continue.")
    st.info("👈 Enter your Google API key in the sidebar to get started.")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
for key, default in [
    ("vector_db", None),
    ("chat_history", []),
    ("doc_names", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# DOCUMENT PROCESSING
# =========================================================

SAFE_CONTENT_PREFIXES = [
    "ignore all previous instructions",
    "disregard your system prompt",
    "you are now",
    "forget you are",
]

def _check_prompt_injection(text: str) -> bool:
    """Rudimentary check for prompt injection patterns in document content."""
    lower = text.lower()
    return any(p in lower for p in SAFE_CONTENT_PREFIXES)


def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".txt": TextLoader, ".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    cls = loaders.get(ext)
    return cls(file_path) if cls else None


def process_uploaded_file(uploaded_file) -> Optional[FAISS]:
    """Load, split, and vectorize an uploaded file. Returns FAISS db or None."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix.lstrip(".") not in SUPPORTED_EXTENSIONS:
        st.error(f"Unsupported format `{suffix}`. Please upload PDF, TXT, or DOCX.")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = get_loader(tmp_path)
        if loader is None:
            st.error("Could not create a loader for this file type.")
            return None

        documents = loader.load()
        if not documents:
            st.error("The file appears to be empty or unreadable.")
            return None

        combined_text = " ".join(d.page_content for d in documents)
        if _check_prompt_injection(combined_text):
            st.warning(
                "⚠️ This document may contain instruction-injection patterns. "
                "Answers will still be grounded strictly in document content."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            st.error("Could not extract any text chunks from the document.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_multiple_files(uploaded_files) -> Optional[FAISS]:
    """Process multiple uploaded files and merge into a single FAISS store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    merged_db = None

    for uploaded_file in uploaded_files:
        db = process_uploaded_file(uploaded_file)
        if db is None:
            continue
        if merged_db is None:
            merged_db = db
        else:
            merged_db.merge_from(db)

    return merged_db


def _build_history_context() -> str:
    """Format the last N chat exchanges as plain text for the LLM prompt."""
    history = st.session_state.chat_history[:-1]  # exclude current user message
    pairs = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            pairs.append(
                f"User: {history[i]['content']}\n"
                f"Assistant: {history[i + 1]['content']}"
            )
    recent = pairs[-MAX_HISTORY_PAIRS:]
    return "\n\n".join(recent)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX (one or more)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help="Max 200 MB per file. Content is never stored permanently.",
    )

    if uploaded_files:
        current_names = sorted(f.name for f in uploaded_files)
        if current_names != sorted(st.session_state.doc_names):
            # File list changed — reset everything
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.doc_names = current_names

        if st.session_state.vector_db is None:
            label = (
                uploaded_files[0].name
                if len(uploaded_files) == 1
                else f"{len(uploaded_files)} documents"
            )
            with st.spinner(f"Processing **{label}**…"):
                db = process_multiple_files(uploaded_files)
            if db:
                st.session_state.vector_db = db
                st.success(f"✅ Ready! {len(uploaded_files)} document(s) loaded.")
            else:
                st.session_state.doc_names = []

    if st.session_state.vector_db:
        st.markdown("**Active documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- `{name}`")
        if st.button("🗑️ Clear & Start Over"):
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.doc_names = []
            st.rerun()

    st.divider()
    st.caption("Powered by Gemini 2.5 Flash · FAISS · LangChain")


# =========================================================
# MAIN UI
# =========================================================
st.title("📊 RAG Document Analyzer")
st.caption(
    "Upload documents and ask questions. Answers are grounded **strictly** in "
    "the uploaded content — the model will not use outside knowledge."
)

if not st.session_state.vector_db:
    st.info("👈 Upload one or more documents in the sidebar to start a conversation.")
    st.stop()

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
user_prompt = st.chat_input("Ask a question about the document(s)…")

if user_prompt:
    user_prompt = user_prompt.strip()
    if not user_prompt:
        st.stop()

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching document…"):
            try:
                docs = st.session_state.vector_db.similarity_search(user_prompt, k=4)
                context = "\n\n---\n\n".join(d.page_content for d in docs)
                history_text = _build_history_context()

                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash",
                    temperature=0.2,
                )

                history_section = (
                    f"\nCONVERSATION HISTORY (for context only — do not use as source):\n"
                    f"{history_text}\n"
                    if history_text
                    else ""
                )

                system_prompt = f"""You are a precise document assistant.
Answer the question based ONLY on the DOCUMENT CONTEXT below.
Do not use any outside knowledge or information from the conversation history as a primary source.
If the answer is not present in the context, respond exactly with:
"I cannot find that information in the uploaded document(s)."

DOCUMENT CONTEXT:
{context}
{history_section}
CURRENT QUESTION:
{user_prompt}

ANSWER:"""

                response = llm.invoke(system_prompt)
                answer = response.content.strip()
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                err_msg = str(e)
                if "API_KEY" in err_msg.upper() or "permission" in err_msg.lower():
                    st.error(
                        "❌ API key error. Please check your Google API key is valid "
                        "and has Generative AI access enabled."
                    )
                elif "quota" in err_msg.lower():
                    st.error(
                        "❌ API quota exceeded. Please wait or check your usage limits "
                        "at https://console.cloud.google.com/apis/dashboard"
                    )
                else:
                    st.error(f"❌ Unexpected error: {e}")
