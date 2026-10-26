"""Dynamic Ollama model auto-discovery with intelligent Gemma model resolution. / الاكتشاف التلقائي الذكي لنماذج Ollama وترقية نماذج Gemma.

Solves the local-vs-builtin collision by dynamically querying Ollama's local tags endpoint,
matching any installed Gemma family model (gemma3, gemma2, gemma, gemma4), and avoiding
hardcoded version mismatches that cause false fallbacks to the built-in rule brain.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Sequence

from brain import LOCAL_OPENER, BrainError, require_loopback

logger = logging.getLogger("playfellow.brain.ollama_discovery")

# Priority order when multiple Gemma models are found and user hasn't explicitly specified
_GEMMA_PRIORITY_PATTERNS = [
    re.compile(r"^gemma4", re.IGNORECASE),
    re.compile(r"^gemma3", re.IGNORECASE),
    re.compile(r"^gemma2", re.IGNORECASE),
    re.compile(r"^gemma", re.IGNORECASE),
]

_PREFERRED_SIZES = ["4b", "8b", "12b", "9b", "2b", "27b", "latest"]


def discover_ollama_models(
    host: str = "http://127.0.0.1:11434",
    timeout: float = 1.5,
) -> tuple[bool, list[str]]:
    """Query Ollama daemon for installed models.

    Args:
        host: Ollama HTTP base URL.
        timeout: Socket timeout in seconds.

    Returns:
        (is_online, installed_model_names)
        Guarantees sockets are cleanly closed.
    """
    try:
        require_loopback(host)  # the model list is asked of this PC only
    except BrainError as exc:
        logger.debug("Ollama discovery refused %s: %s", host, exc)
        return False, []
    endpoint = f"{host.rstrip('/')}/api/tags"
    req = urllib.request.Request(
        endpoint,
        headers={"User-Agent": "PlayFellow/1.0", "Accept": "application/json"},
    )
    try:
        with LOCAL_OPENER.open(req, timeout=timeout) as response:  # no proxy, no redirects
            if response.status != 200:
                return False, []
            raw_body = response.read()
            data = json.loads(raw_body.decode("utf-8"))
            models = data.get("models", [])
            names: list[str] = []
            for item in models:
                name = item.get("name") or item.get("model")
                if name and isinstance(name, str):
                    names.append(name.strip())
            return True, names
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError, TimeoutError, BrainError) as exc:
        logger.debug("Ollama discovery unreachable at %s: %s", endpoint, exc)
        return False, []


def filter_gemma_models(models: Sequence[str]) -> list[str]:
    """Return models containing 'gemma', sorted by generation and optimal gaming/edge size."""
    gemma_models = [m for m in models if "gemma" in m.lower()]
    if not gemma_models:
        return []

    def _sort_key(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        # Generation score (higher is newer)
        gen_score = 0
        for idx, pattern in enumerate(_GEMMA_PRIORITY_PATTERNS):
            if pattern.search(lower):
                gen_score = len(_GEMMA_PRIORITY_PATTERNS) - idx
                break

        # Size score (prefer edge-friendly models 4b-12b)
        size_score = 0
        for idx, size in enumerate(reversed(_PREFERRED_SIZES)):
            if size in lower:
                size_score = idx + 1
                break

        return (gen_score, size_score, lower)

    return sorted(gemma_models, key=_sort_key, reverse=True)


def resolve_ollama_model(
    preferred_model: str | None = None,
    power_tier: str = "smart",
    installed_models: Sequence[str] | None = None,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 1.5,
) -> str | None:
    """Intelligently resolve the best model to run on Ollama.

    Resolution strategy:
    1. If preferred_model is specified and exactly exists in installed_models -> use it.
    2. If preferred_model is specified and prefix/stem matches (e.g. 'gemma3:4b' matches 'gemma3:4b-instruct-q4_K_M') -> use it.
    3. If any Gemma model is installed -> pick top sorted Gemma model.
    4. If no Gemma models but other models exist -> pick the first non-gemma model.
    5. If Ollama has no models or is unreachable -> return None.
    """
    if installed_models is None:
        online, installed = discover_ollama_models(host=host, timeout=timeout)
        if not online or not installed:
            return None
        models = installed
    else:
        models = list(installed_models)

    if not models:
        return None

    # Step 1: Exact match
    if preferred_model:
        pref = preferred_model.strip()
        for m in models:
            if m.lower() == pref.lower():
                return m

    # Step 2: Prefix / stem match
    if preferred_model:
        pref_stem = preferred_model.split(":")[0].strip().lower()
        pref_tag = preferred_model.split(":")[1].strip().lower() if ":" in preferred_model else ""
        for m in models:
            m_lower = m.lower()
            if pref_tag:
                if pref_stem in m_lower and pref_tag in m_lower:
                    return m
            elif pref_stem == m_lower.split(":")[0]:
                return m

    # Step 3: Any Gemma model (sorted by generation & tier)
    gemmas = filter_gemma_models(models)
    if gemmas:
        return gemmas[0]

    # Step 4: Fallback to any installed model if explicitly requesting local
    return models[0]


def is_ollama_ready_for_local(
    preferred_model: str | None = None,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 1.5,
) -> tuple[bool, str | None]:
    """Check if Ollama is running and has at least one valid model ready.

    Returns:
        (ready, resolved_model_name)
    """
    online, installed = discover_ollama_models(host=host, timeout=timeout)
    if not online or not installed:
        return False, None
    resolved = resolve_ollama_model(
        preferred_model=preferred_model,
        installed_models=installed,
    )
    return (resolved is not None), resolved
