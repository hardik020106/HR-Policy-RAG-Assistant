import json
from langchain_openai import ChatOpenAI

from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from hr_assistant import config
from hr_assistant.logger import get_logger


logger = get_logger(__name__)

PRIMARY_PROVIDER = "@hr-policy"
JUDGE_PROVIDER = "@JUDGE"

def get_gateway_llm() -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured to use the Portkey gateway.
    """

    headers = createHeaders(
        api_key=config.portkey_api_key,
        config=config.portkey_config
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=0.0,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        api_key="dummy"
    )

    logger.info("ChatOpenAI instance created with Portkey gateway.")

    return llm


def get_judge_llm() -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured to use the Portkey gateway.
    """

    headers = createHeaders(
        api_key=config.portkey_api_key,
        provider = JUDGE_PROVIDER,
        config=config.portkey_config
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=0.0,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        api_key="dummy"
    )

    logger.info("ChatOpenAI instance created with Portkey gateway.")

    return llm
