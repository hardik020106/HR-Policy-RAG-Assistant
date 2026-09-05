from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import create_embeddings
import os
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def create_vector_store(chunks):
    """
    Create a FAISS vector store using the provided embeddings.

    Args:
        embeddings: An instance of the embeddings model.

    Returns:
        An instance of FAISS vector store.
    """
    logger.info("Creating FAISS vector store...")
    embedding_model = create_embeddings()
    logger.info("FAISS vector store created successfully.")
    return FAISS.from_documents(chunks, embedding_model)


## save vectore store

def save_vector_store(vector_store, path=config.VECTOR_STORE_PATH):
    """
    Save the FAISS vector store to the specified path.

    Args:
        vector_store: An instance of FAISS vector store.
        path: The path where the vector store will be saved.
    """
    logger.info(f"Saving FAISS vector store to {path}...")
    vector_store.save_local(path)

def load_vector_store(path=config.VECTOR_STORE_PATH):
    """
    Load the FAISS vector store from the specified path.

    Args:
        path: The path from where the vector store will be loaded.
    """
    logger.info(f"Loading FAISS vector store from {path}...")
    embedding_model = create_embeddings()
    logger.info(f"FAISS vector store loaded from {path}.")
    return FAISS.load_local(path, embedding_model)


def vector_store_exists(path=config.VECTOR_STORE_PATH):
    """
    Check if the FAISS vector store exists at the specified path.

    Args:
        path: The path to check for the vector store.
    Returns:
        True if the vector store exists, False otherwise.
    """
    logger.info(f"Checking if FAISS vector store exists at {path}...")
    return os.path.exists(os.path.join(path, "index.faiss")) 


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