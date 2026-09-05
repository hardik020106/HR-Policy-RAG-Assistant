from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_documents
from hr_assistant.llm import get_llm
from hr_assistant.tools import create_search_tool
from hr_assistant.splitter import create_splitter
from hr_assistant.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever
)
from hr_assistant.logger import get_logger
from hr_assistant.tracing import check_langsmith_tracing
logger = get_logger(__name__)

def build_vectore_store_for_documents(file_path: str = config.DATA_PATH_FILE, vector_store_path: str = config.VECTOR_STORE_PATH):
    """
    Build a vector store for the documents loaded from the specified file path.

    Args:
        file_path (str): Path to the data file containing documents.
        vector_store_path (str): Path to save the vector store.
        """
    logger.info(f"Building vector store for documents at {file_path}...")
    if vector_store_exists(vector_store_path):
        logger.info(f"Vector store already exists at {vector_store_path}. Skipping creation.")
        return load_vector_store(vector_store_path)
    logger.info("No vector store found. Creating a new one...")
    documents = load_documents(file_path)
    chunks = create_splitter(documents)
    vector_store = create_vector_store(chunks)
    save_vector_store(vector_store, vector_store_path)
    return vector_store


def build_hr_assistant(file_path: str = config.DATA_PATH_FILE, vector_store_path: str = config.VECTOR_STORE_PATH):
    """
    Build the HR assistant by creating a vector store and initializing the agent.

    Args:
        file_path (str): Path to the data file containing documents.
        vector_store_path (str): Path to save the vector store.
    """
    logger.info("Building HR assistant...")
    config.check_api_keys()
    check_langsmith_tracing()
    vector_store = build_vectore_store_for_documents(file_path, vector_store_path)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    search_tool = create_search_tool(retriever)
    hr_agent = create_hr_agent(llm, search_tool)
    return hr_agent


def ask(agent,question:str) -> str:
    """
    Ask a question to the HR agent and get the response.

    Args:
        agent: The HR agent instance.
        question (str): The question to ask.
        """
    logger.info(f"Asking question to HR agent: {question}")
    response = agent.invoke({"message":[{"role":"user","content":question}]})
    return response["message"][0]["content"]



