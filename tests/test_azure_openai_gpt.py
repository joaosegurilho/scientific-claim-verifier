import logging
import os
import sys

from dotenv import load_dotenv

from scverifier.config.settings import Config

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("langchain-azure-gpt-test")

logger.info("Starting LangChain Azure OpenAI (GPT) hello-world test")

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
logger.info("Loading environment variables from .env")
load_dotenv()

api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
deployments = os.getenv("AZURE_DEPLOYMENT_NAMES")

if not api_key:
    logger.error("AZURE_API_KEY not found after loading .env")
    sys.exit(1)

if not endpoint:
    logger.error("AZURE_ENDPOINT not found after loading .env")
    sys.exit(1)

if not deployments:
    logger.error("AZURE_DEPLOYMENT_NAMES not found after loading .env")
    sys.exit(1)

logger.info("AZURE_API_KEY, AZURE_ENDPOINT, and AZURE_DEPLOYMENT_NAMES loaded successfully")

# --------------------------------------------------
# Initialize LangChain Azure OpenAI model
# --------------------------------------------------
try:
    logger.info("Initializing AzureChatOpenAI - through Config.with_llm")

    llm = Config.with_llm(model="gpt-4o-mini", temperature=0.0)

    logger.info("AzureChatOpenAI - through Config.with_llm initialized successfully")
except Exception:
    logger.exception("Failed to initialize AzureChatOpenAI - through Config.with_llm")
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
    logger.info("Received response from Azure OpenAI via LangChain")
except Exception:
    logger.exception("LangChain Azure OpenAI invocation failed")
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
    logger.info("Azure OpenAI response text:")
    logger.info(content)
except Exception:
    logger.exception("Failed to extract response content")
    sys.exit(1)

logger.info("Script finished successfully")
