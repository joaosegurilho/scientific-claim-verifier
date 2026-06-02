import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO, verbose: bool = False, log_file: str | None = None):
    if verbose:
        level = logging.DEBUG
    """Configure logging: stderr + optional file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Suppress third-party noise
    for lib in ("langchain", "httpx", "faiss", "urllib3", "matplotlib", "PIL", "PIL.PngImagePlugin"):
        logging.getLogger(lib).setLevel(logging.WARNING)
