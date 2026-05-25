import os
import sys
import logging

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("gemini-sequential-test")

logger.info("Starting sequential Gemini test (native SDK → LangChain)")

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
# Shared prompt
# --------------------------------------------------
prompt = "Hello, world"
model_name = "gemini-2.5-flash"

logger.info("Using shared prompt and model")
logger.debug(f"Prompt: {prompt}")
logger.debug(f"Model: {model_name}")

# ==================================================
# 1. Native Google Gemini Python SDK call
# ==================================================
logger.info("---- Native Gemini SDK call START ----")

try:
    logger.info("Initializing native Gemini client")
    client = genai.Client(api_key=api_key)
    logger.info("Native Gemini client initialized")

    logger.info("Sending request via native Gemini SDK")
    native_response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    logger.info("Received response from native Gemini SDK")
except Exception:
    logger.exception("Native Gemini SDK call failed")
    sys.exit(1)

logger.debug("Raw native Gemini response object:")
logger.debug(native_response)

try:
    native_text = native_response.text
    logger.info("Extracted native Gemini response text")
    logger.info("Native Gemini response:")
    logger.info(native_text)
except Exception:
    logger.exception("Failed to extract native Gemini response text")
    sys.exit(1)

logger.info("---- Native Gemini SDK call END ----")

# ==================================================
# 2. LangChain Gemini call
# ==================================================
logger.info("---- LangChain Gemini call START ----")

try:
    logger.info("Initializing ChatGoogleGenerativeAI")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        verbose=True,
    )
    logger.info("ChatGoogleGenerativeAI initialized")
except Exception:
    logger.exception("Failed to initialize ChatGoogleGenerativeAI")
    sys.exit(1)

try:
    logger.info("Sending request via LangChain")
    langchain_response = llm.invoke(prompt)
    logger.info("Received response from Gemini via LangChain")
except Exception:
    logger.exception("LangChain Gemini invocation failed")
    sys.exit(1)

logger.debug("Raw LangChain response object:")
logger.debug(langchain_response)

try:
    langchain_text = langchain_response.content
    logger.info("Extracted LangChain response content")
    logger.info("LangChain Gemini response:")
    logger.info(langchain_text)
except Exception:
    logger.exception("Failed to extract LangChain response content")
    sys.exit(1)

logger.info("---- LangChain Gemini call END ----")

logger.info("Sequential Gemini test completed successfully")
