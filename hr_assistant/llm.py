# from langchain_groq import ChatGroq
# from webbrowser import get

from hr_assistant.gateway import get_gateway_llm,get_judge_llm
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)
def get_llm():
    """
    Get an instance of the ChatGroq model.

    Returns:
        An instance of ChatGroq.
    """
    logger.info(f"Initializing LLm via PortKey...")
    llm = get_gateway_llm()
    logger.info("ChatGroq model initialized successfully.")
    return llm

def get_the_judge_llm():
    """
    Get an instance of the ChatGroq model.

    Returns:
        An instance of ChatGroq.
    """
    logger.info(f"Initializing LLm via PortKey...")
    llm = get_judge_llm()
    logger.info("ChatGroq model initialized successfully.")
    return llm