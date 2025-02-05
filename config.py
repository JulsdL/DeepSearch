import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai-03-mini-high")
