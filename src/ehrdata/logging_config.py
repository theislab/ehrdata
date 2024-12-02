import logging


def configure_logging(level=logging.INFO):
    """Configures logging for the package."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(message)s",
        force=True,
    )
