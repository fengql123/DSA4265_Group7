"""Configuration loader and LLM/embedding factories.

Merges config/settings.yaml with .env secrets into typed settings.
Provides get_llm() and get_embedding_model() factories used by all agents.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file at import time
load_dotenv()

# Project root (directory containing pyproject.toml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_nested(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a nested value using dotted key notation (e.g. 'llm.provider')."""
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Load and merge settings from config/settings.yaml.

    Returns a plain dict. Access nested values with _get_nested() or directly.
    """
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    if settings_path.exists():
        with open(settings_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
):
    """Create a LangChain chat model based on config or overrides.

    Supports four providers:
      - "openrouter": ChatOpenRouter (default) — 290+ models via OpenRouter gateway
      - "openai": ChatOpenAI — direct OpenAI API
      - "anthropic": ChatAnthropic — direct Anthropic API
      - "local": ChatOpenAI with SGLang backend (http://localhost:30000/v1)

    Args:
        provider: Override provider from config.
        model: Override model name from config.
        temperature: Override temperature from config.
    """
    cfg = load_settings()
    llm_cfg = cfg.get("llm", {})

    provider = provider or llm_cfg.get("provider", "openrouter")
    model = model or llm_cfg.get("model", "anthropic/claude-haiku-4.5")
    temperature = temperature if temperature is not None else llm_cfg.get("temperature", 0.1)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        api_key = os.getenv(
            llm_cfg.get("openrouter", {}).get("api_key_env", "OPENROUTER_API_KEY"),
            "",
        )
        return ChatOpenRouter(model=model, temperature=temperature, api_key=api_key)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv(
            llm_cfg.get("openai", {}).get("api_key_env", "OPENAI_API_KEY"),
            "",
        )
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv(
            llm_cfg.get("anthropic", {}).get("api_key_env", "ANTHROPIC_API_KEY"),
            "",
        )
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    elif provider == "local":
        from langchain_openai import ChatOpenAI

        # SGLang serves an OpenAI-compatible API on port 30000 by default.
        # Start server: python -m sglang.launch_server --model-path <model> --port 30000
        base_url = llm_cfg.get("local", {}).get(
            "base_url", "http://localhost:30000/v1"
        )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key="EMPTY",
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Use 'openrouter', 'openai', 'anthropic', or 'local'."
        )


def get_embedding_model():
    """Create a HuggingFace embedding model based on config.

    Returns a sentence-transformers SentenceTransformer instance.
    Model name comes from config/settings.yaml → embedding.model
    """
    from sentence_transformers import SentenceTransformer

    cfg = load_settings()
    model_name = _get_nested(cfg, "embedding.model", "BAAI/bge-base-en-v1.5")
    return SentenceTransformer(model_name)


def load_prompt(agent_name: str) -> str:
    """Load a prompt template from config/prompts/{agent_name}.yaml.

    The YAML file should have a top-level 'system_prompt' key containing
    a template string with placeholders like {ticker}, {date}, {context}.
    """
    prompt_path = PROJECT_ROOT / "config" / "prompts" / f"{agent_name}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path) as f:
        data = yaml.safe_load(f)

    if "system_prompt" not in data:
        raise KeyError(f"Prompt file {prompt_path} must have a 'system_prompt' key")

    return data["system_prompt"]


def get_rag_config() -> dict:
    """Get RAG-specific configuration values."""
    cfg = load_settings()
    rag_cfg = cfg.get("rag", {})
    return {
        "chunk_size": rag_cfg.get("chunk_size", 512),
        "chunk_overlap": rag_cfg.get("chunk_overlap", 64),
        "retrieval_top_k": rag_cfg.get("retrieval_top_k", 10),
        "chroma_persist_dir": rag_cfg.get(
            "chroma_persist_dir", str(PROJECT_ROOT / "data" / "chromadb")
        ),
    }
