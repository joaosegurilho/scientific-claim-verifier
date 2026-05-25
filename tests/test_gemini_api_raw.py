import os
import sys
import logging
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gemini-test")

logger.info("Starting Gemini hello-world test script")

# --------------------------------------------------
# Load API key
# --------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)

logger.info("GEMINI_API_KEY found")

# --------------------------------------------------
# Initialize client
# --------------------------------------------------
try:
    logger.info("Initializing Gemini client")
    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Gemini client")
    sys.exit(1)

# --------------------------------------------------
# Prepare request
# --------------------------------------------------
model_name = "gemini-2.5-flash"
prompt = "Hello, world"

logger.info("Preparing request")
logger.debug(f"Model: {model_name}")
logger.debug(f"Prompt: {prompt}")

# --------------------------------------------------
# Call Gemini API
# --------------------------------------------------
try:
    logger.info("Sending request to Gemini API")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    logger.info("Received response from Gemini API")
except Exception as e:
    logger.exception("Gemini API call failed")
    sys.exit(1)

# --------------------------------------------------
# Log raw response
# --------------------------------------------------
logger.debug("Raw response object:")
logger.debug(response)

# --------------------------------------------------
# Extract and log text output
# --------------------------------------------------
try:
    text = response.text
    logger.info("Extracted text from response")
    logger.info("Gemini response text:")
    logger.info(text)
except Exception as e:
    logger.exception("Failed to extract text from response")
    sys.exit(1)

logger.info("Script finished successfully")
