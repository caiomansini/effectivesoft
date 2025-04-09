
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/vectorstore")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
