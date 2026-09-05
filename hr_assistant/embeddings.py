from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def create_embeddings():
    """
    Create embeddings using the JinaEmbeddings model.

    Returns:
        An instance of JinaEmbeddings.
    """
    embeddings = JinaEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        api_key=config.jina_api_key
    )
    return embeddings