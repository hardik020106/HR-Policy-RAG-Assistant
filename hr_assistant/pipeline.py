from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_documents
from hr_assistant.llm import get_llm
from hr_assistant.tools import create_search_tool
from hr_assistant.splitter import split_documents
from hr_assistant.vector_store import (
    create_vector_store,
    # save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever
)
from hr_assistant.guardrails import check_input_policy, check_output_policy,REFUSAL_MESSAGE
from hr_assistant.logger import get_logger
from hr_assistant.tracing import check_langsmith_tracing
logger = get_logger(__name__)

def build_vectore_store_for_documents(file_path: str = config.DATA_PATH_FILE):
    """
    Build a vector store for the documents loaded from the specified file path.

    Args:
        file_path (str): Path to the data file containing documents.
        vector_store_path (str): Path to save the vector store.
        """
    logger.info(f"Building vector store for documents at {file_path}...")
    if vector_store_exists():
        logger.info("Vector store already exists. Skipping creation.")
        return load_vector_store()
    logger.info("No vector store found. Creating a new one...")
    documents = load_documents(file_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    # save_vector_store(vector_store, vector_store_path)
    return vector_store


def build_hr_assistant(file_path: str = config.DATA_PATH_FILE):
    """
    Build the HR assistant by creating a vector store and initializing the agent.

    Args:
        file_path (str): Path to the data file containing documents.
    """
    logger.info("Building HR assistant...")
    config.check_api_keys()
    check_langsmith_tracing()
    vector_store = build_vectore_store_for_documents(file_path)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    search_tool = create_search_tool(retriever)
    hr_agent = create_hr_agent(llm, [search_tool])
    return hr_agent


def ask(agent,question:str) -> str:
    """
    Ask a question to the HR agent and get the response.

    Args:
        agent: The HR agent instance.
        question (str): The question to ask.
        """
    logger.info(f"Asking question to HR agent: {question}")
    #input guard
    if not check_input_policy(question):
        logger.warning("Input policy violation detected. Refusing to answer.")
        return REFUSAL_MESSAGE
    response = agent.invoke({"messages":[{"role":"user","content":question}]})
    #output guard
    if not check_output_policy(response["messages"][-1].content):
        logger.warning("Output policy violation detected. Refusing to provide the answer.")
        return REFUSAL_MESSAGE
    return response["messages"][-1].content



