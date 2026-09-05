import json
from langchain_openai import ChatOpenAI

from portkey_ai import PORTKEY_GATWAY_URL, createHeaders
from hr_assistant import config
from hr_assistant.logger import get_logger


logger = get_logger(__name__)

PRIMARY_TARGET = {"provider":"@hr-policy", 
                  "override_params":{"model":config.LLM_MODEL_NAME}}

FALLBACK_TARGET = {"provider":"@HR_POLICY_backup", 
                  "override_params":{"model":"openai/gpt-oss-20b"}}


GATEWAY_CONFIG = {
    "strategy": {
        "mode": "fallback"
    },
    "targets": [PRIMARY_TARGET, FALLBACK_TARGET]
}      

def get_gateway_llm()->ChatOpenAI:
    """
    Create a ChatOpenAI instance configured to use the Portkey gateway.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI configured for the Portkey gateway.
    """
    headers = createHeaders(config.portkey_api_key)
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=0.0,
        portkey_gateway_url=PORTKEY_GATWAY_URL,
        portkey_gateway_config=json.dumps(GATEWAY_CONFIG),
        portkey_gateway_headers=headers
    )
    logger.info("ChatOpenAI instance created with Portkey gateway.")
    return llm