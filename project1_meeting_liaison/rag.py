"""
RAG Module - Voice Style Retrieval
===================================
PURPOSE: This module is the "memory" of the user's writing style.
HOW IT WORKS:
  1. User uploads sample emails they've written before.
  2. We split them into chunks and store them in ChromaDB (a local vector database).
  3. When drafting a new email, we retrieve the most SIMILAR style examples to use as context.

WHY RAG HERE?
  Without RAG, the LLM would write in a generic tone.
  With RAG, it retrieves real examples of how *this specific user* writes,
  making the output sound authentic.
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Directory where user's sample emails live
EMAILS_DIR = os.path.join(os.path.dirname(__file__), "sample_emails")
# ChromaDB will persist its index here (so we don't re-index every run)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_voice_db")


def build_voice_index() -> Chroma:
    """
    Reads all .txt email files from sample_emails/, creates embeddings,
    and stores them in a ChromaDB vector store.
    Call this once after adding new sample emails.
    """
    # Load every .txt file in sample_emails/
    documents = []
    for filename in os.listdir(EMAILS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(EMAILS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(
                Document(page_content=content, metadata={"source": filename})
            )

    if not documents:
        raise ValueError(
            "No sample emails found! Please add .txt files to the sample_emails/ folder."
        )

    # Split long emails into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create embeddings using Azure OpenAI and store in ChromaDB
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[RAG] Indexed {len(chunks)} chunks from {len(documents)} email(s).")
    return vectorstore


def load_voice_index() -> Chroma:
    """
    Loads an existing ChromaDB index from disk.
    Returns None if no index exists yet (user hasn't uploaded emails).
    """
    if not os.path.exists(CHROMA_DIR):
        return None
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def retrieve_voice_examples(query: str, k: int = 3) -> str:
    """
    Given a topic/query (e.g. "follow-up on budget approval"),
    retrieves the top-k most relevant email chunks from the user's past writing.
    Returns them as a single string to inject into the LLM prompt.
    """
    vectorstore = load_voice_index()
    if vectorstore is None:
        return "No voice samples available. Write in a professional tone."

    results = vectorstore.similarity_search(query, k=k)
    examples = "\n\n---\n\n".join([doc.page_content for doc in results])
    return f"Here are examples of how the user writes emails:\n\n{examples}"
