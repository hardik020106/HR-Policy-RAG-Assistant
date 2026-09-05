from langchain_groq import ChatGroq
from hr_assistant import config

def get_llm():
    """
    Get an instance of the ChatGroq model.

    Returns:
        An instance of ChatGroq.
    """
    llm = ChatGroq(
        model_name=config.LLM_MODEL_NAME,
        api_key=config.groq_api_key
    )
    return llm