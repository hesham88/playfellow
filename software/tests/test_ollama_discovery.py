"""P0-local-discovery: dynamic Ollama discovery, Gemma resolution, and local vs builtin isolation.
Combines standalone discovery unit tests and brain chain integration tests.
"""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from brain.local import discover_ollama_models as local_discover_ollama, resolve_ollama_model as local_resolve_ollama
from controls import load_card
from brain.ollama_discovery import (
    discover_ollama_models,
    filter_gemma_models,
    is_ollama_ready_for_local,
    resolve_ollama_model,
)

CARD = load_card(Path(__file__).resolve().parent.parent / "controls" / "practice_arena.yaml")


# ==============================================================================
# Part 1: Standalone Ollama Discovery & Gemma Selection Unit Tests
# ==============================================================================

def test_filter_gemma_models_prioritizes_newer_and_right_size():
    models = [
        "llama3:8b",
        "gemma2:2b",
        "gemma3:12b",
        "gemma3:4b",
        "mistral:7b",
        "gemma:2b",
        "gemma3:latest",
    ]
    filtered = filter_gemma_models(models)
    assert "llama3:8b" not in filtered
    assert "mistral:7b" not in filtered
    assert filtered[0].startswith("gemma3:")
    assert "gemma2:2b" in filtered
    assert "gemma:2b" in filtered


def test_resolve_ollama_model_exact_match():
    models = ["llama3:8b", "gemma3:4b", "gemma2:9b"]
    resolved = resolve_ollama_model("gemma2:9b", installed_models=models)
    assert resolved == "gemma2:9b"


def test_resolve_ollama_model_fuzzy_stem_match():
    models = ["llama3:8b", "gemma3:4b-instruct-q4_K_M", "phi3:mini"]
    resolved = resolve_ollama_model("gemma3:4b", installed_models=models)
    assert resolved == "gemma3:4b-instruct-q4_K_M"


def test_resolve_ollama_model_falls_back_to_best_gemma_when_preferred_not_found():
    models = ["llama3:8b", "gemma3:12b", "gemma2:9b"]
    resolved = resolve_ollama_model("gemma4:e4b", installed_models=models)
    assert resolved == "gemma3:12b"


def test_resolve_ollama_model_falls_back_to_non_gemma_if_no_gemma_available():
    models = ["llama3:8b", "qwen2.5:7b"]
    resolved = resolve_ollama_model("gemma3:4b", installed_models=models)
    assert resolved == "llama3:8b"


def test_resolve_ollama_model_returns_none_when_empty():
    assert resolve_ollama_model("gemma3:4b", installed_models=[]) is None


@patch("brain.LOCAL_OPENER.open")  # the proxy-free, redirect-free opener every local request uses
def test_discover_ollama_models_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "gemma3:4b", "size": 2500000000},
            {"name": "llama3:8b", "size": 4700000000},
        ]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    online, models = discover_ollama_models(host="http://127.0.0.1:11434")
    assert online is True
    assert models == ["gemma3:4b", "llama3:8b"]


@patch("brain.LOCAL_OPENER.open")  # the proxy-free, redirect-free opener every local request uses
def test_discover_ollama_models_offline(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    online, models = discover_ollama_models(host="http://127.0.0.1:11434")
    assert online is False
    assert models == []


@patch("brain.ollama_discovery.discover_ollama_models")
def test_is_ollama_ready_for_local_helper(mock_discover):
    mock_discover.return_value = (True, ["gemma3:4b", "llama3:8b"])
    ready, model = is_ollama_ready_for_local()
    assert ready is True
    assert model == "gemma3:4b"

    mock_discover.return_value = (False, [])
    ready, model = is_ollama_ready_for_local()
    assert ready is False
    assert model is None


# ==============================================================================
# Part 2: Brain Chain & UI Label Integration Tests
# ==============================================================================

def tags(*names):
    """A urlopen that answers /api/tags, and records that its response was closed."""
    closed = []

    def opener(req, timeout=None):
        response = Mock()
        response.read.return_value = json.dumps({"models": [{"name": n} for n in names]}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(side_effect=lambda *a: closed.append(True) or False)
        return response
    opener.closed = closed
    return opener


def refused(req, timeout=None):
    raise urllib.error.URLError("connection refused")


def test_discovery_lists_installed_models_and_closes_the_socket():
    opener = tags("gemma3:4b", "llava:7b")
    assert local_discover_ollama(opener=opener) == ["gemma3:4b", "llava:7b"]
    assert opener.closed == [True]


def test_discovery_says_none_when_ollama_is_off_and_empty_when_nothing_installed():
    assert local_discover_ollama(opener=refused) is None
    assert local_discover_ollama(opener=tags()) == []


def test_gemma3_found_when_the_default_gemma4_tag_is_missing():
    assert local_resolve_ollama("gemma4:e4b", "smart", ["gemma3:4b"]) == "gemma3:4b"


def test_smart_takes_the_biggest_gemma_and_thrifty_the_lightest():
    models = ["gemma3:12b", "gemma3:4b", "gemma3:27b", "gemma2:2b", "llava:7b"]
    assert local_resolve_ollama("gemma4:e4b", "smart", models) == "gemma3:27b"
    assert local_resolve_ollama("gemma4:e2b", "thrifty", models) == "gemma2:2b"


def test_the_exact_tag_wins_and_latest_counts():
    assert local_resolve_ollama("gemma4:e4b", "thrifty", ["gemma3:27b", "gemma4:e4b"]) == "gemma4:e4b"
    assert local_resolve_ollama("gemma", "smart", ["gemma:latest"]) == "gemma:latest"


def test_without_gemma_a_vision_model_then_anything_then_none():
    assert local_resolve_ollama("gemma4:e4b", "smart", ["phi3:3b", "llava:7b"]) == "llava:7b"
    assert local_resolve_ollama("gemma4:e4b", "smart", ["phi3:3b"]) == "phi3:3b"
    assert local_resolve_ollama("gemma4:e4b", "smart", []) is None
