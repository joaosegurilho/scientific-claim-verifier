import logging
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    log_file: str | None = None,
    file_level: int = logging.DEBUG,
):
    """Configure logging: stderr + optional file.

    Parameters
    ----------
    level : int
        Log level for stderr output (default INFO).
    verbose : bool
        If True, sets stderr level to DEBUG (overrides ``level``).
    log_file : str, optional
        Path to a log file. If given, messages at ``file_level``
        or above are written there.
    file_level : int
        Log level for the file handler (default DEBUG).
    """

    if verbose:
        level = logging.DEBUG

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(file_level)
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Suppress third-party noise
    for lib in (
        "langchain",
        "httpx",
        "faiss",
        "urllib3",
        "matplotlib",
        "PIL",
        "PIL.PngImagePlugin",
        "google_genai.models",  # ignore this also
        "google_genai._api_client",  # ignore this also
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)
