import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Configuration constants
DEFAULT_LLM_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.1
