from langchain_groq import ChatGroq
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)
def get_llm():
    """
    Get an instance of the ChatGroq model.

    Returns:
        An instance of ChatGroq.
    """
    logger.info(f"Initializing ChatGroq model: {config.LLM_MODEL_NAME}...")
    llm = ChatGroq(
        model_name=config.LLM_MODEL_NAME,
        api_key=config.groq_api_key
    )
    logger.info("ChatGroq model initialized successfully.")
    return llm