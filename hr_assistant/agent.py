from langchain.agents import create_agent
from hr_assistant import config
from hr_assistant.llm import get_llm

from hr_assistant.tools import create_search_tool
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def create_hr_agent(llm,tools):
    """
    Create an HR agent using the specified model and tools.

    Returns:
        An instance of the HR agent.
    """
    logger.info("Creating HR agent...")
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=config.SYSTEM_PROMPT)