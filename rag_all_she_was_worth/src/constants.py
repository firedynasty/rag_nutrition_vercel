import os
import tomllib
from typing import Any

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config file path
CONFIG_PATH = os.path.join(PROJECT_ROOT, "rag_config.toml")

# LanceDB URI (relative to project root)
LANCEDB_URI = os.path.join(PROJECT_ROOT, "databases", "novel_lancedb")


def get_rag_config() -> dict[str, Any]:
    """Load and return the RAG configuration from TOML file."""
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)
