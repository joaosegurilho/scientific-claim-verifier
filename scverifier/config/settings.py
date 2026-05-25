"""Configuration settings for the Proposition-based Retrieval Pipeline."""

import os
from dotenv import load_dotenv
from time import sleep

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the pipeline."""

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    CORE_API_KEY = os.getenv("CORE_API_KEY")
    OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")  # Optional
    OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO")  # Required for polite pool

    # Model Settings
    LLM_MODEL = "gemini-2.5-flash" #"gemini-2.5-flash-lite"  # "gemini-2.0-flash-lite" # 2.0 flash lite is older but slightly cheaper
    EMBEDDING_MODEL = "nomic-embed-text:v1.5"
    BATCH_LLM_MODEL = "gemini-2.0-flash-lite" #"models/gemini-2.5-flash-lite-preview-09-2025" #"models/gemini-2.0-flash-lite"
    LLM_FALLBACK_MODEL = "gemini-2.0-flash-lite"  # Fallback model when primary hits rate limits
    LLM_TEMPERATURE = 0
    LLM_TIMEOUT = 120  # Timeout for LLM calls in seconds
    MAX_RETRIES = 2  # Maximum number of retries for LLM calls (2 = one retry after initial failure)

    # Chunking Settings
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    # Quality Thresholds
    QUALITY_THRESHOLDS = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

    # Default Retrieval Settings
    CHUNK_RETRIEVAL_K = 8  # Number of chunks to retrieve
    PROPOSITION_RETRIEVAL_K = 50  # Default number of propositions to retrieve per query
    MAX_PROPS_PER_PAPER = 5  # Maximum propositions from each paper used in verification (ensures source diversity)

    # Knowledge base storage path
    DB_NAME = "data/kb_all" #"data/kb_benchmarking_scifact_dev" # "data/kb_benchmarking_msvec" # #"data/kb_benchmarking_scifact"

    # Google gRPC logging for ALTS (Application Layer Transport Security) credential
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_CPP_PLUGIN_LOGGER_LEVEL"] = "ERROR"

    # Agent settings
    AGENT_MODEL = "gemini-2.5-flash" #"gemini-flash-latest"
    RECURSION_LIMIT = 75  # Maximum reasoning steps for autonomous agents
    AGENT_TEMPERATURE = 0  # Temperature setting for agent LLMs
    AGENT_MAX_OUTPUT_TOKENS = 65536  # Maximum output tokens for agent responses
    
    # Batch processing settings
    BATCH_FILE_SPLIT_LIMIT = 2000  # Maximum number of requests per batch file (split if exceeded)

    @classmethod
    def setup_environment(cls):
        """Set up environment variables."""
        if cls.GEMINI_API_KEY:
            os.environ["GOOGLE_API_KEY"] = str(cls.GEMINI_API_KEY)
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

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
                print(f"        Retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}")
                sleep(2)
