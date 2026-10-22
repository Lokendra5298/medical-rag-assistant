import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_PATH = Path(__file__).resolve().parent / "vectorstore/db_faiss"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

st.set_page_config(page_title="MediBot", page_icon="🩺")
st.title("🩺 MediBot")
st.caption("Medical Question Answering using RAG | FAISS + LangChain + Groq")


# Load FAISS database
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        str(DB_PATH), embeddings, allow_dangerous_deserialization=True
    )


# Create RAG chain
@st.cache_resource
def create_qa_chain():
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model=MODEL,
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a medical information assistant.
    Answer the question using only the provided context.
    If the answer is unavailable, say "I don't know based on the provided documents."
    Do not invent information or provide personalized medical prescriptions.

    Context: {context}
    Question: {input}
    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(source)

if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching medical documents..."):
                response = create_qa_chain().invoke({"input": question})

            answer = response["answer"]
            sources = [
                f"{Path(doc.metadata.get('source', 'Unknown')).name} "
                f"— Page {doc.metadata.get('page', 0) + 1}"
                for doc in response["context"]
            ]
            sources = list(dict.fromkeys(sources))

            st.markdown(answer)
            with st.expander("📚 Sources"):
                for source in sources:
                    st.write(source)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error: {e}")