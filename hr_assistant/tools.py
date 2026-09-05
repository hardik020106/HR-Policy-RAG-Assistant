from langchain.tools import tool

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
        results = retriever.invoke(query)
        return "\n\n".join([result.page_content for result in results])

    return search_documents