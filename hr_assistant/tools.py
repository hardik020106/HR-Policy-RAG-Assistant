from langchain.tools import tool
from hr_assistant.logger import get_logger

logger = get_logger(__name__)
def create_search_tool(retriever):
    """
    Create a search tool using the provided retriever.

    Args:
        retriever: An instance of a retriever.

    Returns:
        A tool for searching documents.
    """
    @tool
    def search_documents(query: str):
        """
        Search documents using the provided query.

        Args:
            query: The search query string.

        Returns:
            List of search results.
        """
        logger.info(f"Searching documents for query: {query}...")
        results = retriever.invoke(query)
        logger.info(f"Search results for query: {query}...")
        return "\n\n".join([result.page_content for result in results])

    return search_documents