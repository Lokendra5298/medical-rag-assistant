# create_memory_from_docs.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Step 1: Load PDF Document(s)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "The_GALE_ENCYCLOPEDIA_of_MEDICAL.pdf"

DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"


def load_pdf_files(data_path):
    """
    Load PDF documents from either:
    - A single PDF file
    - A directory containing multiple PDF files
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"PDF file or directory not found: {data_path}"
        )

    if data_path.is_file():

        if data_path.suffix.lower() != ".pdf":
            raise ValueError("The provided file must be a PDF.")

        loader = PyPDFLoader(str(data_path))

    elif data_path.is_dir():

        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

    else:
        raise ValueError("Invalid PDF path.")

    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents were loaded.")

    return documents


# ============================================================
# Step 2: Create Text Chunks
# ============================================================

def create_chunks(extracted_data):
    """
    Split PDF documents into smaller overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    if not text_chunks:
        raise ValueError("No text chunks were generated.")

    return text_chunks


# ============================================================
# Step 3: Initialize Hugging Face Embedding Model
# ============================================================

def get_embedding_model():
    """
    Load the Sentence Transformers embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embedding_model


# ============================================================
# Step 4: Create and Save FAISS Vector Database
# ============================================================

def create_vector_database(text_chunks, embedding_model):
    """
    Convert text chunks into embeddings and store them
    in a FAISS vector database.
    """

    DB_FAISS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    db = FAISS.from_documents(
        documents=text_chunks,
        embedding=embedding_model
    )

    db.save_local(str(DB_FAISS_PATH))

    return db


# ============================================================
# Main Execution
# ============================================================

def main():

    print("\nStarting PDF Processing Pipeline...\n")

    # Step 1: Load PDF
    print("[1/4] Loading PDF documents...")

    documents = load_pdf_files(DATA_PATH)

    print(f"Total PDF pages loaded: {len(documents)}")

    # Step 2: Create chunks
    print("\n[2/4] Splitting documents into chunks...")

    text_chunks = create_chunks(documents)

    print(f"Total text chunks generated: {len(text_chunks)}")

    # Step 3: Load embedding model
    print("\n[3/4] Loading Hugging Face embedding model...")

    embedding_model = get_embedding_model()

    print("Embedding model loaded successfully.")

    # Step 4: Create FAISS database
    print("\n[4/4] Generating embeddings and creating FAISS database...")

    create_vector_database(
        text_chunks=text_chunks,
        embedding_model=embedding_model
    )

    print("\nFAISS vector database created successfully!")
    print(f"Database saved at: {DB_FAISS_PATH}")


if __name__ == "__main__":
    main()