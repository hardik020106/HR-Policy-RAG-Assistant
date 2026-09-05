from langchain.agents import create_agent
from hr_assistant import config
from hr_assistant.llm import get_llm

from hr_assistant.tools import create_search_tool
def create_hr_agent(llm,tools):
    """
    Create an HR agent using the specified model and tools.

    Returns:
        An instance of the HR agent.
    """
    return create_agent(
        model=llm,
        tools=tools,
        system_message=config.SYSTEM_PROMPT)