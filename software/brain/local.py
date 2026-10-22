"""A brain that runs on this computer with Ollama, no internet needed. / عقل يعمل على هذا الكمبيوتر باستخدام Ollama بدون إنترنت."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import ollama
from PIL import Image

from brain import LOCAL_TIMEOUT, BrainError, classify_error, to_jpeg_bytes

KEEP_ALIVE = "30m"  # default if not specified
FIRST_CALL_TIMEOUT = 600  # seconds for first call (cold load from USB HDD)
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
PULL_HINT = "gemma3:4b"  # what to tell a player who has no Gemma model yet
VISION_MARKERS = ("qwen2.5vl", "qwen2-vl", "llava", "minicpm-v", "llama3.2-vision")


from brain.ollama_discovery import (
    discover_ollama_models as _discover_ollama,
    resolve_ollama_model as _resolve_ollama,
    is_ollama_ready_for_local,
    filter_gemma_models,
)

def discover_ollama_models(timeout: float = 1.0, opener=None) -> list[str] | None:
    """The model names Ollama has installed, or None when Ollama is not running (an empty list means running, nothing installed)."""
    opener = opener or urllib.request.urlopen  # looked up now, so a test can swap it
    try:
        with opener(urllib.request.Request(OLLAMA_TAGS_URL), timeout=timeout) as response:  # always closed: no leaked socket
            data = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError, ValueError):
        return None
    models = data.get("models", []) if isinstance(data, dict) else []
    return [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]


def _size(tag: str) -> float:
    """Billions of parameters from a tag name (gemma3:27b -> 27, gemma4:e4b -> 4); unknown tags count as a middling 4."""
    match = re.search(r"(?<![0-9a-z.])e?(\d+(?:\.\d+)?)b(?![a-z])", tag.lower().split(":", 1)[-1])
    return float(match.group(1)) if match else 4.0


def resolve_ollama_model(preferred: str, power: str, installed: list[str]) -> str | None:
    """Which installed model to use. The preferred tag if it is there; else any Gemma (the biggest for 'smart', the lightest
    for 'thrifty'); else a vision model; else whatever is installed; None when nothing is installed."""
    if not installed:
        return None
    if preferred in installed:
        return preferred
    if f"{preferred}:latest" in installed:
        return f"{preferred}:latest"
    gemmas = [m for m in installed if "gemma" in m.lower()]
    if gemmas:
        return (max if power == "smart" else min)(gemmas, key=_size)
    for marker in VISION_MARKERS:
        for m in installed:
            if marker in m.lower():
                return m
    return installed[0]


class LocalBrain:
    def __init__(self, model: str, keep_alive: str = KEEP_ALIVE):
        self.model = model
        self.keep_alive = keep_alive
        self._is_first_call = True
        # Start with long timeout for first call
        self.client = ollama.Client(timeout=FIRST_CALL_TIMEOUT)

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text, "images": [to_jpeg_bytes(image) for image in images]},
                ],
                format="json",
                think=False,  # thinking adds ~40 s per step on a laptop CPU
                keep_alive=self.keep_alive,  # T31-C2: use the setting value
                options={"temperature": 0.2},
            )
            # After first successful call, switch to normal timeout
            if self._is_first_call:
                self._is_first_call = False
                self.client = ollama.Client(timeout=LOCAL_TIMEOUT)
        except ConnectionError as exc:
            raise BrainError("local_offline", retry=True, wait_seconds=5) from exc
        except Exception as exc:
            # Check if the model is not found
            error_text = str(exc).lower()
            if "model" in error_text and "not found" in error_text:
                from messages import tr
                raise BrainError(
                    "model_missing",
                    retry=False,
                    detail=tr(
                        f"Model {self.model} isn't downloaded. Run: ollama pull {self.model}",
                        f"النموذج {self.model} غير محمّل. شغّل: ollama pull {self.model}"
                    )
                ) from exc
            raise classify_error(exc) from exc
        return response["message"]["content"] or ""
