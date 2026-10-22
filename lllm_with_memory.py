import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH = "vectorstore/db_faiss"


# Step 1: Load Embedding Model

def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embedding_model


# Step 2: Load FAISS Database

def load_vectorstore():

    embedding_model = get_embedding_model()

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# Step 3: Load Hugging Face Chat Model

def load_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )

    return llm


# Step 4: Create Custom Prompt

def set_custom_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
        You are a medical assistant.

        Answer the user's question using only the given context.

        If the answer is not available in the context,
        say "I don't know based on the provided documents."

        Do not make up information.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    return prompt


# Step 5: Create RAG Chain

def create_qa_chain():

    vectorstore = load_vectorstore()

    llm = load_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = set_custom_prompt()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    qa_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return qa_chain


# Step 6: Ask Questions

def main():

    print("\n" + "=" * 50)
    print("MEDICAL RAG CHATBOT")
    print("=" * 50)

    print("\nLoading model and FAISS database...")

    try:
        qa_chain = create_qa_chain()
    except Exception as e:
        print(f"\nInitialization error: {e}")
        return

    print("\nChatbot is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        user_query = input("Enter your question: ").strip()

        if user_query.lower() in ["exit", "quit", "q"]:
            print("Chatbot closed.")
            break

        if not user_query:
            continue

        try:

            response = qa_chain.invoke({
                "input": user_query
            })

            print("\nAnswer:")
            print(response["answer"])

            print("\nSource Documents:")

            for i, doc in enumerate(response["context"], 1):

                source = os.path.basename(
                    doc.metadata.get("source", "Unknown")
                )

                page = doc.metadata.get("page")

                print(f"\nSource {i}: {source}")

                if page is not None:
                    print(f"Page: {page + 1}")

                print(f"Content: {doc.page_content[:300]}...")

            print("\n" + "-" * 50)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()