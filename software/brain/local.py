"""A brain that runs on this computer with Ollama, no internet needed. / عقل يعمل على هذا الكمبيوتر باستخدام Ollama بدون إنترنت."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import ollama
from PIL import Image

from brain import CLIENT_ARGS, ENDPOINTS, LOCAL_OPENER, LOCAL_TIMEOUT, REPLY_SCHEMA, BrainError, classify_error, require_loopback, to_jpeg_bytes
from brain.meter import begin, finish
from brain.preflight import DEFAULT_CONTEXT_TOKENS, check_blocked, ensure_fits, pick_ollama

KEEP_ALIVE = "30m"  # default if not specified
FIRST_CALL_TIMEOUT = 600  # seconds for first call (cold load from USB HDD)
OLLAMA_TAGS_URL = ENDPOINTS["local"] + "/api/tags"
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
    opener = opener or LOCAL_OPENER.open  # no proxy, no redirects; looked up now, so a test can swap it
    try:
        with opener(urllib.request.Request(OLLAMA_TAGS_URL), timeout=timeout) as response:  # always closed: no leaked socket
            data = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError, ValueError, BrainError):
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
    name = "local"

    def __init__(self, model: str, keep_alive: str = KEEP_ALIVE, meter=None, host: str | None = None, structured: bool = False,
                 memory=None, model_bytes: int | None = None, context_tokens: int = DEFAULT_CONTEXT_TOKENS, opener=None):
        self.host = host or ENDPOINTS["local"]
        require_loopback(self.host)  # the local brain never leaves this PC, whatever set the address (LM-2)
        self.structured = structured  # True sends our reply schema in `format`; False keeps plain "json" as before
        self._reads_pictures: bool | None = None
        self.model = model
        self._memory, self.model_bytes, self.context_tokens = memory, model_bytes, context_tokens  # memory: a reader (tests fake it)
        self._opener = opener  # for the read-only lists; None = brain.LOCAL_OPENER.open
        self.meter = meter  # counts each question where it is asked (brain/meter.py); local tokens are free but real
        self.keep_alive = keep_alive
        self._is_first_call = True
        # Start with long timeout for first call
        self.client = ollama.Client(host=self.host, timeout=FIRST_CALL_TIMEOUT, **CLIENT_ARGS)

    def _list(self, path: str):
        """One of Ollama's read-only lists (the model files, the running models) or None: 2 s, loopback only, no proxy, no redirect."""
        try:
            request = urllib.request.Request(self.host.rstrip("/") + path)
            with (self._opener or LOCAL_OPENER.open)(request, timeout=2) as response:
                return json.loads(response.read())
        except BrainError:
            raise
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError, ValueError):
            return None

    def reads_pictures(self) -> bool | None:
        """Whether Ollama lists "vision" in this model's capabilities (POST /api/show, https://github.com/ollama/ollama/blob/main/docs/api.md);
        None when it cannot tell, so an older Ollama or a stopped one does not block play."""
        if self._reads_pictures is None:
            try:
                capabilities = self.client.show(self.model).get("capabilities")
            except Exception:
                return None
            if capabilities is not None:
                self._reads_pictures = "vision" in capabilities
        return self._reads_pictures

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        # R224/D90: room for the model? Before ANY request (the first chat makes Ollama load it), read fresh on every attempt.
        check_blocked(self.model)  # D90: by name, before ANY request
        tags, running = self._list("/api/tags"), self._list("/api/ps")  # read-only: they load nothing; None when no answer
        size, loaded = pick_ollama(tags, running, self.model)
        self.found_bytes = self.model_bytes or size  # LOAD-1: the loading strip's estimate
        ensure_fits(self.model, self.model_bytes or size, self.context_tokens, has_vision=bool(images), reader=self._memory,
                    already_loaded=loaded)
        if images and self.reads_pictures() is False:
            raise BrainError("model_text_only", retry=False, detail=self.model)  # never drop the picture quietly
        start = begin(self)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text, "images": [to_jpeg_bytes(image) for image in images]},
                ],
                format=REPLY_SCHEMA if self.structured else "json",
                think=False,  # thinking adds ~40 s per step on a laptop CPU
                keep_alive=self.keep_alive,  # T31-C2: use the setting value
                options={"temperature": 0.2},
            )
            # After first successful call, switch to normal timeout
            if self._is_first_call:
                self._is_first_call = False
                self.client = ollama.Client(host=self.host, timeout=LOCAL_TIMEOUT, **CLIENT_ARGS)
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
        reply = response["message"]["content"] or ""
        # Ollama leaves prompt_eval_count out when the whole prompt was already in its cache: then 0 were read anew.
        finish(self, start, system, text, len(images), reply, prompt=response.get("prompt_eval_count") or 0,
               completion=response.get("eval_count"))
        return reply
