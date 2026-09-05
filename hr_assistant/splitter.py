from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)
def split_documents(documents):
    """
    Split documents into smaller chunks based on the specified chunk size and overlap.

    Args:
        documents: List of documents to be split.   
        chunk_size: Size of each text chunk.    
        """
    logger.info(f"Splitting documents into chunks of size {config.CHUNK_SIZE} with overlap {config.CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs