import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Document Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # If running locally without secrets, or on a server without them
    user_key = st.sidebar.text_input("Enter Google API Key", type="password")
    if user_key:
        os.environ["GOOGLE_API_KEY"] = user_key
    elif "GOOGLE_API_KEY" not in os.environ:
         st.sidebar.warning("Please enter your API Key to continue.")
         st.stop()

# --- SESSION STATE ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LOGIC ---
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt": return TextLoader(file_path)
    elif ext == ".pdf": return PyPDFLoader(file_path)
    elif ext == ".docx": return Docx2txtLoader(file_path)
    else: return None

def process_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = get_loader(tmp_path)
        if not loader: return None
            
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.from_documents(chunks, embeddings)
        
        os.remove(tmp_path)
        return db
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Select a file (PDF, TXT, DOCX)", type=["txt", "pdf", "docx"])
    
    if uploaded_file and st.session_state.vector_db is None:
        # Professional "Status" widget
        with st.status("Processing document...", expanded=True) as status:
            st.write("Reading file...")
            time.sleep(0.5)
            st.write("Vectorizing content...")
            db = process_file(uploaded_file)
            if db:
                st.session_state.vector_db = db
                status.update(label="Document Ready", state="complete", expanded=False)
    
    if st.button("Clear / New Chat"):
        st.session_state.vector_db = None
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN PAGE ---
st.title("Document Analysis Assistant")
st.markdown("---")

# 1. Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Input
if prompt := st.chat_input("Ask a question about the document..."):
    if st.session_state.vector_db is None:
        st.warning("Please upload a document in the sidebar to begin.")
    else:
        # User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Retrieve Context
            db = st.session_state.vector_db
            docs = db.similarity_search(prompt, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Generate Answer
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            ai_prompt = f"""
            You are a helpful assistant. Answer the user's question strictly based on the context below.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {prompt}
            """
            
            try:
                response = llm.invoke(ai_prompt)
                message_placeholder.markdown(response.content)
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            except Exception as e:
                message_placeholder.error("An error occurred. Please try again.")