from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config

def split_documents(documents):
    """
    Split documents into smaller chunks based on the specified chunk size and overlap.

    Args:
        documents: List of documents to be split.   
        chunk_size: Size of each text chunk.    
        """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs