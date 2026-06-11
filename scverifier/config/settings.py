"""Configuration settings for the Proposition-based Retrieval Pipeline."""

import logging
import os
from time import sleep

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the pipeline."""

    # Azure APIs
    AZURE_API_KEY: str | None = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT: str | None = os.getenv("AZURE_ENDPOINT")
    AZURE_DEPLOYMENT_NAMES: list[str] = os.getenv("AZURE_DEPLOYMENT_NAMES").split(",")
    AZURE_MODEL_DEPLOYMENT_MAPPING: dict[str, str] = dict(
        zip(
            [x.strip("deployment-") for x in AZURE_DEPLOYMENT_NAMES],
            AZURE_DEPLOYMENT_NAMES,
        )
    )
    AZURE_API_VERSION: str = (
        "2024-12-01-preview"  # "2025-04-14"  # Default API version for Azure OpenAI (update as needed)
    )

    # Gemini APIs
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

    # Literature APIs
    SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    CORE_API_KEY: str | None = os.getenv("CORE_API_KEY")
    OPENALEX_API_KEY: str | None = os.getenv("OPENALEX_API_KEY")  # Optional
    OPENALEX_MAILTO: str | None = os.getenv("OPENALEX_MAILTO")  # Required for polite pool

    # Model Settings
    LLM_MODEL: str = (
        "gpt41-mini"  # "gemini-2.5-flash"  # "gemini-2.5-flash-lite"  # "gemini-2.0-flash-lite" # 2.0 flash lite is older but slightly cheaper
    )
    EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"
    BATCH_LLM_MODEL: str = (
        "gemini-2.0-flash-lite"  # "models/gemini-2.5-flash-lite-preview-09-2025" #"models/gemini-2.0-flash-lite"
    )
    LLM_FALLBACK_MODEL: str = "gemini-2.0-flash-lite"  # Fallback model when primary hits rate limits
    LLM_TEMPERATURE: int = 0
    LLM_TIMEOUT: int = 120  # Timeout for LLM calls in seconds
    MAX_RETRIES: int = 2  # Maximum number of retries for LLM calls (2 = one retry after initial failure)

    # Chunking Settings
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50

    # Quality Thresholds
    QUALITY_THRESHOLDS: dict[str, int] = {
        "accuracy": 7,
        "clarity": 7,
        "completeness": 7,
        "conciseness": 7,
    }

    # Default Retrieval Settings
    CHUNK_RETRIEVAL_K: int = 8  # Number of chunks to retrieve
    PROPOSITION_RETRIEVAL_K: int = 50  # Default number of propositions to retrieve per query
    MAX_PROPS_PER_PAPER: int = 5  # Maximum propositions from each paper used in verification (ensures source diversity)

    # Knowledge base storage path
    DB_NAME: str = (
        "data/kb_all"  # "data/kb_benchmarking_scifact_dev" # "data/kb_benchmarking_msvec" # #"data/kb_benchmarking_scifact"
    )

    # Google gRPC logging for ALTS (Application Layer Transport Security) credential
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_CPP_PLUGIN_LOGGER_LEVEL"] = "ERROR"

    # Agent settings
    AGENT_MODEL: str = "gemini-2.5-flash"  # "gemini-flash-latest"
    RECURSION_LIMIT: int = 75  # Maximum reasoning steps for autonomous agents
    AGENT_TEMPERATURE: int = 0  # Temperature setting for agent LLMs
    AGENT_MAX_OUTPUT_TOKENS: int = 65536  # Maximum output tokens for agent responses

    # Batch processing settings
    BATCH_FILE_SPLIT_LIMIT: int = 2000  # Maximum number of requests per batch file (split if exceeded)

    FEATURES: set[str] = set()
    KNOWN_FEAUTURES: set[str] = set()

    # TODO: might be redudndant with environment variable loading at the top - consider consolidating
    # Might only be needed if other libs require the keys to be a differnt name
    @classmethod
    def setup_environment(cls):
        """Set up environment variables."""
        # if cls.GEMINI_API_KEY:
        #     os.environ["GOOGLE_API_KEY"] = str(cls.GEMINI_API_KEY)
        # else:
        #     raise ValueError("GEMINI_API_KEY not found in environment variables")

        if cls.SEMANTIC_SCHOLAR_API_KEY:
            os.environ["SEMANTIC_SCHOLAR_API_KEY"] = str(cls.SEMANTIC_SCHOLAR_API_KEY)
        else:
            raise ValueError("SEMANTIC_SCHOLAR_API_KEY not found in environment variables")

        if cls.CORE_API_KEY:
            os.environ["CORE_API_KEY"] = str(cls.CORE_API_KEY)
        else:
            raise ValueError("CORE_API_KEY not found in environment variables")

        if cls.OPENALEX_MAILTO:
            os.environ["OPENALEX_MAILTO"] = str(cls.OPENALEX_MAILTO)
        # else:
        #     raise ValueError("OPENALEX_MAILTO not found in environment variables")

        if cls.OPENALEX_API_KEY:
            os.environ["OPENALEX_API_KEY"] = str(cls.OPENALEX_API_KEY)
        # API key is optional - no else clause needed

    @staticmethod
    def retry_llm_call(func, max_retries=MAX_RETRIES):
        """Simple retry wrapper for LLM calls."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning("Retry %d/%d after error: %s", attempt + 1, max_retries, str(e)[:50])
                logger.warning(f"\t\tRetry {attempt + 1}/{max_retries} after error: {str(e)}")
                sleep(2)

    @classmethod
    def with_llm(
        cls,
        temperature: int = 0,
        max_tokens: int = 6000,
        timeout: int = 120,
        fallback: bool = False,
        **kwargs,
    ) -> AzureChatOpenAI | BaseChatModel:
        """Deploy an LLM."""

        if fallback:
            model = cls.LLM_FALLBACK_MODEL
        else:
            model = kwargs.get("model", None) or cls.LLM_MODEL

        if model.startswith("gpt") or model.startswith("deepseek"):
            return AzureChatOpenAI(
                azure_endpoint=cls.AZURE_ENDPOINT,
                api_key=cls.AZURE_API_KEY,
                api_version=cls.AZURE_API_VERSION,
                azure_deployment=cls.AZURE_MODEL_DEPLOYMENT_MAPPING.get(model),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif model.startswith("gemini"):
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=cls.GEMINI_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        # else:
        #     # This is not suitable way to do it
        #     # Works for DeepSeek, Mistral, etc.
        #     return init_chat_model(
        #         model,
        #         api_key=cls.AZURE_API_KEY,
        #         api_version=cls.AZURE_API_VERSION,
        #         model_provider="azure_ai",
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         timeout=timeout,
        #     )
