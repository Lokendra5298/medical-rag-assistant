# 🩺 Medical RAG Chatbot

A medical question-answering chatbot built using **Retrieval-Augmented Generation (RAG)**, LangChain, FAISS, Hugging Face embeddings, Groq LLM, and Streamlit.

The chatbot retrieves relevant information from medical PDF documents and generates context-based answers with source references.

## 🚀 Features

- PDF document loading and text chunking
- Semantic search using Hugging Face embeddings
- FAISS vector database for efficient retrieval
- Groq LLM for context-based answer generation
- Interactive chatbot interface using Streamlit
- PDF source references with page numbers
- Chat history displayed within the session

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** LangChain
- **LLM:** Groq (GPT-OSS-20B)
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **Frontend:** Streamlit
- **Document Processing:** PyPDFLoader

## 📁 Project Structure

```text
medical-chatbot/
│
├── create_memory_from_docs.py    # PDF processing and FAISS indexing
├── lllm_with_memory.py           # Command-line RAG chatbot
├── medical_bot.py                # Streamlit chatbot
├── requirements.txt              # Python dependencies
├── .env                          # API credentials (not committed)
├── data/                         # Medical PDF documents (optional)
└── vectorstore/
    └── db_faiss/                 # FAISS vector database
```

## ⚙️ Installation

**1. Clone the repository**

git clone <repository-url>
cd medical-chatbot

**2. Create a virtual environment**

python -m venv venv
source venv/bin/activate

**3. Install dependencies**

pip install -r requirements.txt

**4. Configure environment variables**

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

## ▶️ Usage

**Step 1: Generate FAISS embeddings**

Place your medical PDF in the project directory and update `DATA_PATH`
in `create_memory_from_docs.py`.

python create_memory_from_docs.py

**Step 2: Run the Streamlit chatbot**

streamlit run medical_bot.py

**Step 3: Run the command-line chatbot (optional)**

python lllm_with_memory.py

## 🔄 RAG Workflow
```
Medical PDF → Text Extraction → Text Chunking → Hugging Face Embeddings → FAISS Vector Store → Similarity Search → Groq LLM → Answer with Sources
```
## 💬 Example Questions

- What is diabetes mellitus?
- What are the symptoms of hypertension?
- Explain the difference between benign and malignant tumors.
- What are the common treatments for pneumonia?
- What causes cardiovascular diseases?

## Disclaimer

This chatbot is intended for educational and research purposes only.
It does not provide professional medical diagnosis or treatment advice.
Answers depend on the uploaded documents and should be independently verified.

## Future Improvements
```
- Conversational memory
- Hybrid search (BM25 + dense retrieval)
- Cross-encoder reranking
- RAG evaluation and hallucination detection
- Support for multiple document formats
```
## 📄 License

This project is intended for educational and research purposes.
