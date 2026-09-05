from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def create_embeddings():
    """
    Create embeddings using the JinaEmbeddings model.

    Returns:
        An instance of JinaEmbeddings.
    """
    logger.info(f"Creating embeddings using model: {config.EMBEDDING_MODEL_NAME}...")
    embeddings = JinaEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        api_key=config.jina_api_key
    )
    logger.info("Embeddings created successfully.")
    return embeddings