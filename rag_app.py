import os
import streamlit as st
import sys
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def get_loader(file_path):
    """Factory function to pick the right loader based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def main():
    print("==========================================")
    print("      UNIVERSAL RAG: DOC ANALYZER        ")
    print("==========================================")

    # 1. Get User Input
    while True:
        file_path = input("\nEnter the path to your file (.txt, .pdf, .docx): ").strip()
        # Remove quotes if user dragged-and-dropped file
        file_path = file_path.replace('"', '').replace("'", "")
        
        if os.path.exists(file_path):
            break
        print("File not found. Please try again.")

    # 2. Load & Split
    print(f"\nAnalyzing '{file_path}'...")
    try:
        loader = get_loader(file_path)
        documents = loader.load()
        
        # Splitter logic
        text_splitter = CharacterTextSplitter(
            separator="\n",     # Split by newlines for better PDF handling
            chunk_size=1000,    # Larger chunks for full context
            chunk_overlap=100   # Overlap ensures sentences aren't cut in half
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Successfully loaded and split into {len(chunks)} chunks.")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 3. Vectorize (The Brain)
    print("Vectorizing data (Sending to Google)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # We build a temporary in-memory DB for this session
    db = FAISS.from_documents(chunks, embeddings)
    print("Database ready!")

    # 4. Setup Chat Engine
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    print("\n==========================================")
    print("       CHAT SESSION STARTED              ")
    print("       Type 'exit' to quit.              ")
    print("==========================================")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Retrieval
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # Prompt
        prompt = f"""
        You are an intelligent document assistant.
        Answer the question strictly based on the provided Context.
        If the answer is not in the context, say "I cannot find that in the document."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        """

        # Generation
        print("Thinking...", end="\r") # Simple loading effect
        try:
            response = llm.invoke(prompt)
            print(f"AI: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()