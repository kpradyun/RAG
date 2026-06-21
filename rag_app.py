"""
rag_app.py — Terminal (CLI) interface for the RAG Document Analyzer.

Usage:
    python rag_app.py
    python rag_app.py path/to/document.pdf   # optional direct path argument

Requires GOOGLE_API_KEY in environment or .env file.
"""

import os
import sys

# Load .env if present (before importing anything that reads env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; key can be set manually

# Validate API key early — fail fast with a clear message
_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
if not _api_key:
    print(
        "\n[ERROR] GOOGLE_API_KEY is not set.\n"
        "  Option A: export GOOGLE_API_KEY='AIzaSy...'\n"
        "  Option B: create a .env file with: GOOGLE_API_KEY=AIzaSy...\n"
        "  Get a key at: https://aistudio.google.com/app/apikey\n"
    )
    sys.exit(1)

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

SUPPORTED = {".txt", ".pdf", ".docx"}

BANNER = """
╔══════════════════════════════════════════╗
║       RAG Document Analyzer (CLI)        ║
║  Type 'exit' or Ctrl-C to quit.          ║
╚══════════════════════════════════════════╝"""


def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {".txt": TextLoader, ".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    cls = mapping.get(ext)
    if cls is None:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED)}"
        )
    return cls(file_path)


def load_and_vectorize(file_path: str) -> FAISS:
    """Load a document, split it, and return an in-memory FAISS vector store."""
    file_path = file_path.strip().strip("'\"")  # handle drag-and-drop quotes

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED)}")

    print(f"\n→ Loading '{os.path.basename(file_path)}'…")
    loader = get_loader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("The file appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"→ Split into {len(chunks)} chunks.")

    print("→ Creating embeddings (this calls the Google API once)…")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.from_documents(chunks, embeddings)
    print("→ Vector store ready.\n")
    return db


def answer_question(db: FAISS, query: str) -> str:
    """Retrieve relevant chunks and generate a grounded answer."""
    docs = db.similarity_search(query, k=3)
    if not docs:
        return "No relevant content found in the document for that question."

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a precise document assistant.
Answer the question based ONLY on the CONTEXT below.
If the answer is not in the context, say: "I cannot find that in the document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    response = llm.invoke(prompt)
    return response.content.strip()


def get_file_path_from_user() -> str:
    while True:
        try:
            path = input("File path (.txt / .pdf / .docx): ").strip().strip("'\"")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED:
                return path
            print(f"  Unsupported format '{ext}'. Please use: {', '.join(SUPPORTED)}")
        else:
            print("  File not found. Please check the path and try again.")


def main():
    print(BANNER)

    # Accept an optional CLI argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("\nEnter the path to your document (or drag-and-drop it here):")
        file_path = get_file_path_from_user()

    try:
        db = load_and_vectorize(file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to process document: {e}")
        sys.exit(1)

    print("Chat session started. Ask questions about your document.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            print("Goodbye!")
            break

        try:
            answer = answer_question(db, query)
            print(f"\nAI: {answer}\n")
        except Exception as e:
            err = str(e)
            if "API_KEY" in err.upper() or "permission" in err.lower():
                print("[ERROR] API key is invalid or lacks Generative AI access.")
            elif "quota" in err.lower():
                print("[ERROR] API quota exceeded. Please check your usage limits.")
            else:
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
