from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import create_embeddings
import os

def create_vector_store(chunks):
    """
    Create a FAISS vector store using the provided embeddings.

    Args:
        embeddings: An instance of the embeddings model.

    Returns:
        An instance of FAISS vector store.
    """
    embedding_model = create_embeddings()
    return FAISS.from_documents(chunks, embedding_model)


## save vectore store

def save_vector_store(vector_store, path=config.VECTOR_STORE_PATH):
    """
    Save the FAISS vector store to the specified path.

    Args:
        vector_store: An instance of FAISS vector store.
        path: The path where the vector store will be saved.
    """
    vector_store.save_local(path)

def load_vector_store(path=config.VECTOR_STORE_PATH):
    """
    Load the FAISS vector store from the specified path.

    Args:
        path: The path from where the vector store will be loaded.
    """
    embedding_model = create_embeddings()
    return FAISS.load_local(path, embedding_model)


def vector_store_exists(path=config.VECTOR_STORE_PATH):
    """
    Check if the FAISS vector store exists at the specified path.

    Args:
        path: The path to check for the vector store.
    Returns:
        True if the vector store exists, False otherwise.
    """
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
    return vector_store.as_retriever(search_kwargs={"k": top_k})