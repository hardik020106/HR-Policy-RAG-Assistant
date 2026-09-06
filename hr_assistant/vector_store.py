# from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import create_embeddings
import os
from hr_assistant.logger import get_logger
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = get_logger(__name__)

def create_vector_store(chunks):
    """
    Embedd the text chunks and store into the qdrant vector store.
    """
    logger.info("Creating Qdrant vector store...")
    embedding_model = create_embeddings()
    logger.info("Qdrant vector store created successfully.")
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name
    )
    return vector_store


## save vector store

# def save_vector_store(vector_store, path=config.VECTOR_STORE_PATH):
#     """
#     Save the FAISS vector store to the specified path.

#     Args:
#         vector_store: An instance of FAISS vector store.
#         path: The path where the vector store will be saved.
#     """
#     logger.info(f"Saving FAISS vector store to {path}...")
#     vector_store.save_local(path)

def load_vector_store():
    """
    Connect to the Qdrant Cloud 
    collection that was created for the HR policy documents.
    """
    logger.info(f"Loading Qdrant vector store ...")
    embedding_model = create_embeddings()
    logger.info(f"Qdrant vector store loaded.")
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name
    )


def vector_store_exists():
    """
    Check if qdrant store alreeady exists.
    """
    logger.info(f"Checking if Qdrant vector store exists...")
    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key
    )
    return client.collection_exists(collection_name=config.qdrant_collection_name)


def get_retriever(vector_store, top_k=config.TOP_K_RESULTS):
    """
    Get a retriever from the FAISS vector store.

    Args:
        vector_store: An instance of FAISS vector store.
        top_k: Number of top results to retrieve.
    Returns:
        A retriever instance.
    """
    logger.info(f"Getting retriever from FAISS vector store with top_k={top_k}...")
    return vector_store.as_retriever(search_kwargs={"k": top_k})