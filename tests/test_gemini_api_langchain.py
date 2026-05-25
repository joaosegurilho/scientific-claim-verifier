import os
import sys
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("langchain-gemini-test")

logger.info("Starting LangChain Gemini hello-world test")

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
logger.info("Loading environment variables from .env")
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    logger.error("GEMINI_API_KEY not found after loading .env")
    sys.exit(1)

logger.info("GEMINI_API_KEY loaded successfully")

# --------------------------------------------------
# Initialize LangChain Gemini model
# --------------------------------------------------
try:
    logger.info("Initializing ChatGoogleGenerativeAI")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.0,
        verbose=True,
    )
    logger.info("ChatGoogleGenerativeAI initialized successfully")
except Exception:
    logger.exception("Failed to initialize ChatGoogleGenerativeAI")
    sys.exit(1)

# --------------------------------------------------
# Prepare message
# --------------------------------------------------
prompt = "Hello, world"
logger.info("Preparing prompt")
logger.debug(f"Prompt content: {prompt}")

messages = prompt

# --------------------------------------------------
# Invoke model
# --------------------------------------------------
try:
    logger.info("Sending request via LangChain")
    response = llm.invoke(messages)
    logger.info("Received response from Gemini via LangChain")
except Exception:
    logger.exception("LangChain Gemini invocation failed")
    sys.exit(1)

# --------------------------------------------------
# Log raw response
# --------------------------------------------------
logger.debug("Raw LangChain response object:")
logger.debug(response)

# --------------------------------------------------
# Extract and log text
# --------------------------------------------------
try:
    content = response.content
    logger.info("Extracted response content")
    logger.info("Gemini response text:")
    logger.info(content)
except Exception:
    logger.exception("Failed to extract response content")
    sys.exit(1)

logger.info("Script finished successfully")