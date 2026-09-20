"""Central configuration for DermAI.

Values are read from the environment (optionally loaded from a repo-root ``.env``
via python-dotenv). Everything has a sensible default so the app boots without
any configuration; setting a Groq key upgrades chat generation from the offline
extractive fallback to grounded LLM answers.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of the ``dermai`` package directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Load a repo-root .env if present (no error if it is missing).
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


# --- Paths -----------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"

MODEL_PATH = Path(_env("DERMAI_MODEL_PATH", str(MODELS_DIR / "best_model.h5")))
FAISS_INDEX_DIR = Path(_env("DERMAI_FAISS_INDEX_DIR", str(ROOT_DIR / "faiss_index")))

# --- Classifier ------------------------------------------------------------
# The published HAM10000 CNN was trained on 28x28 RGB crops.
IMAGE_SIZE = int(_env("DERMAI_IMAGE_SIZE", "28"))

# --- Retrieval (LangChain + FAISS) -----------------------------------------
EMBEDDING_MODEL = _env("DERMAI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(_env("DERMAI_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env("DERMAI_CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(_env("DERMAI_RETRIEVER_TOP_K", "4"))

# --- LLM (Groq free-tier, OpenAI-compatible) -------------------------------
GROQ_API_KEY = _env("GROQ_API_KEY", _env("DERMAI_GROQ_API_KEY", ""))
GROQ_MODEL = _env("DERMAI_GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TEMPERATURE = float(_env("DERMAI_GROQ_TEMPERATURE", "0.2"))


def groq_enabled() -> bool:
    """True when a Groq API key is configured."""
    return bool(GROQ_API_KEY)
