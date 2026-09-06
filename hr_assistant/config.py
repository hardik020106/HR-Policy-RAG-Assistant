import os 
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

groq_api_key = os.getenv("GROQ_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
portkey_api_key = os.getenv("PORTKEY_API_KEY")
portkey_config = os.getenv("PORTKEY_CONFIG")

GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

DATA_PATH_FILE = os.path.join("data", "hr_policy.txt")


LLM_MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

CHUNK_SIZE = 1000  # Size of each text chunk
CHUNK_OVERLAP = 200  # Overlap between chunks   

TOP_K_RESULTS = 5  # Number of top results to retrieve from the vector store

SYSTEM_PROMPT = (
    "You are a friendly HR assistant. "
    "Always use the search_documents tool to look up information "
    "from the HR policy documents before answering. "
    "Use the retrieved information to answer the user's question. "
    "If the answer is not present in the search results, "
    "say you don't know instead of guessing."
)



def check_api_keys():
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is not set in the environment variables.")


