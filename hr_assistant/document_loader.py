from langchain_community.document_loaders import TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def load_documents(file_path: str = config.DATA_PATH_FILE):
    """
    Load documents from the specified data file path.

    Returns:
        List of loaded documents.
    """
    logger.info(f"Loading documents from {file_path}...")
    loader = TextLoader(file_path, encoding="utf-8")
    logger.info("Documents loaded successfully.")
    documents = loader.load()
    return documents