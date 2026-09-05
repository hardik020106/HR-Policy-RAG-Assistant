from langchain_community.document_loaders import TextLoader
from hr_assistant import config

def load_documents(file_path: str = config.DATA_PATH_FILE):
    """
    Load documents from the specified data file path.

    Returns:
        List of loaded documents.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents