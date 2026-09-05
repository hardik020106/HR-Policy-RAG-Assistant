import os 
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

groq_api_key = os.getenv("GROQ_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

DATA_PATH_FILE = os.path.join("data", "hr_policy.txt")

VECTOR_STORE_PATH = os.path.join("data", "faiss_index") # Path to store the FAISS index

LLM_MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

CHUNK_SIZE = 1000  # Size of each text chunk
CHUNK_OVERLAP = 200  # Overlap between chunks   

TOP_K_RESULTS = 5  # Number of top results to retrieve from the vector store

SYSTEM_PROMPT = """You are an HR assistant. You will be provided with a question and a set of context documents. Use the context to answer the question as accurately as possible. If the answer is not contained within the context, respond with "I don't know." Do not make up answers. Be concise and clear in your response."""


def check_api_keys():
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is not set in the environment variables.")