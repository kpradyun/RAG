import os
import tempfile
import streamlit as st
from typing import Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Document Analysis Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# API KEY HANDLING
# =========================================================
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    user_key = st.sidebar.text_input(
        "Enter Google API Key",
        type="password",
        help="Required to use Gemini models",
    )
    if user_key:
        os.environ["GOOGLE_API_KEY"] = user_key
    elif "GOOGLE_API_KEY" not in os.environ:
        st.sidebar.warning("Please enter your Google API key.")
        st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# DOCUMENT LOADING
# =========================================================
def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return TextLoader(file_path)
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    return None


def process_uploaded_file(uploaded_file) -> Optional[FAISS]:
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = get_loader(tmp_path)
        if loader is None:
            st.error("Unsupported file format.")
            return None

        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separator="\n",
        )
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        vector_db = FAISS.from_documents(chunks, embeddings)
        os.remove(tmp_path)

        return vector_db

    except Exception as e:
        st.exception(e)
        return None

# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_file and st.session_state.vector_db is None:
        with st.status("Processing document...", expanded=True):
            st.write("Loading document...")
            st.write("Splitting text...")
            st.write("Creating embeddings...")
            db = process_uploaded_file(uploaded_file)
            if db:
                st.session_state.vector_db = db
                st.success("Document processed successfully.")

    if st.button("🔄 Clear / New Chat"):
        st.session_state.vector_db = None
        st.session_state.chat_history = []
        st.rerun()

# =========================================================
# MAIN UI
# =========================================================
st.title("📊 Document Analysis Assistant")
st.markdown("Ask questions or request summaries based on the uploaded document.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input("Ask a question or say 'Summarize this document'...")

if prompt:
    if st.session_state.vector_db is None:
        st.warning("Please upload a document first.")
    else:
        # User message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                # Retrieve context
                docs = st.session_state.vector_db.similarity_search(
                    prompt, k=4
                )
                context = "\n\n".join(d.page_content for d in docs)

                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash",
                    temperature=0.2,
                )

                system_prompt = f"""
You are a helpful academic assistant.

Use ONLY the context below.
Summarize clearly and organize answers logically.
If the answer is not present, say:
"I don't know based on the document."

CONTEXT:
{context}

QUESTION:
{prompt}
"""

                response = llm.invoke(system_prompt)
                answer = response.content.strip()

                placeholder.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.exception(e)
