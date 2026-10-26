"""A brain that runs on this computer with LM Studio, no internet needed. / عقل يعمل على هذا الكمبيوتر باستخدام LM Studio بدون إنترنت.

LM Studio serves the OpenAI chat protocol on 127.0.0.1:1234 (pinned in brain.ENDPOINTS). Docs this follows:
  - structured output (response_format json_schema): https://lmstudio.ai/docs/developer/openai-compat/structured-output
  - idle unload (per-request "ttl", seconds):         https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict
  - model list (capabilities.vision; older "type": "vlm"): https://lmstudio.ai/docs/developer/rest/list
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request

from PIL import Image

from brain import ENDPOINTS, LOCAL_OPENER, LOCAL_TIMEOUT, REPLY_SCHEMA, BrainError, classify_error, require_loopback, to_jpeg_bytes
from brain.meter import begin, finish
from brain.preflight import DEFAULT_CONTEXT_TOKENS, check_blocked, ensure_fits, pick_lmstudio

DEFAULT_TTL = 1800  # seconds: a 7 GB model takes about a minute to load from F:, so it must not unload mid-session
LIST_TIMEOUT = 2  # seconds for a read-only list request
FIRST_CALL_TIMEOUT = 600  # seconds for the first call (cold load from the USB disk)
UNITS = {"s": 1, "m": 60, "h": 3600}


def ttl_seconds(keep_alive: str) -> int | None:
    """The settings' keep-alive text ("30m", "2h", "90s", "-1" = forever) as LM Studio's ttl seconds; None leaves it unset."""
    match = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", str(keep_alive))
    if not match:
        return None
    return int(match.group(1)) * UNITS[match.group(2) or "s"]


# ---- Muse Glimmer (R207, report 02 section 3.3): a model that always thinks. The card's five gotchas, built in ----
GLIMMER = re.compile(r"glimmer", re.IGNORECASE)
GLIMMER_STOPS = ["<|end_of_text|>", "<|eot|>"]  # the real stop tokens (200001, 200008); <|eom|> is NEVER one
NEVER_STOP = "<|eom|>"
MIN_GLIMMER_BUILD = 10353  # llama.cpp b10353 or newer; older builds refuse the files
PROPS_TIMEOUT = 2
# Per part: Glimmer thinks at low strength for the Player (every step waits for it), more for the Captain and the Coach.
# UNVERIFIED on a real Glimmer (D66: it needs a 24 GB GPU): the request field is the OpenAI-style `reasoning_effort`.
REASONING_BY_PART = {"player": "low", "eyes": "low", "chat": "low", "captain": "medium", "coach": "high"}
# Room a request needs besides the thinking, in tokens: a picture (up to 4096 visual tokens), our prompt, the JSON answer. Margins, not measurements.
PICTURE_TOKENS, PROMPT_TOKENS, ANSWER_TOKENS = 4096, 1500, 400
THINKING_TOKENS = {"low": 1024, "medium": 4096, "high": 8192}  # margins for the thinking, by strength


def _error_text(exc) -> str:
    """The body of an HTTP error as text ("" when it cannot be read); the connection is closed either way."""
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:
        return ""
    finally:
        exc.close()


def is_glimmer(model: str) -> bool:
    return bool(GLIMMER.search(model or ""))


def safe_stop(stop) -> list[str] | None:
    """The stop strings to send: whatever was asked for, without <|eom|> (stopping there cuts an answer short). None when nothing is left."""
    kept = [s for s in (stop or []) if isinstance(s, str) and s.strip().lower() != NEVER_STOP]
    return kept or None


def context_per_slot(total: int, slots: int) -> int:
    """Each request gets the total context divided by the number of parallel slots (-c / -np), and the thinking must fit in it."""
    return int(total) // max(1, int(slots))


def parse_build(text) -> int | None:
    """The llama.cpp build number from a build text such as "b10353-4f1a2c" or "version: 10353 (abc)"; None when there is none."""
    match = re.search(r"(?<![0-9A-Za-z])b?(\d{4,6})(?![0-9])", text) if isinstance(text, str) else None
    return int(match.group(1)) if match else None


class LMStudioBrain:
    name = "local"

    def __init__(self, model: str, base_url: str | None = None, ttl_seconds: int | None = DEFAULT_TTL, meter=None, opener=None,
                 memory=None, model_bytes: int | None = None, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
                 stop=None, part: str = "player", context_total: int | None = None, slots: int = 1):
        self.model = model
        self.stop, self.part = stop, part  # stop strings (never <|eom|>); the brain part this serves (Player, Captain, ...)
        self.context_total, self.slots = context_total, slots  # the engine's -c and -np, when known
        self._no_reasoning_field = False  # True once a server has rejected `reasoning_effort`
        self.meter = meter  # counts each question where it is asked (brain/meter.py); local tokens are free but real
        self.base_url = (base_url or ENDPOINTS["lmstudio"]).rstrip("/")
        self.ttl_seconds = ttl_seconds
        self._memory, self.model_bytes, self.context_tokens = memory, model_bytes, context_tokens  # memory: a reader (tests fake it)
        self._opener = opener  # a test passes a fake; None uses the proxy-free, redirect-free LOCAL_OPENER
        self._reads_pictures: bool | None = None
        self._is_first_call = True

    def _open(self, request, timeout):
        require_loopback(request.full_url)  # checked on every request, whatever set the address
        return (self._opener or LOCAL_OPENER.open)(request, timeout=timeout)

    def _list(self):
        """LM Studio's read-only model list (GET /api/v0/models: loads nothing), or None when it does not answer in 2 s or
        answers nonsense. Always through the loopback-only, proxy-free opener."""
        try:
            with self._open(urllib.request.Request(self.base_url + "/api/v0/models"), LIST_TIMEOUT) as response:
                return json.loads(response.read())
        except BrainError:
            raise
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError, ValueError, AttributeError):
            return None

    def _vision(self, answer) -> bool | None:
        """Whether the list says this model reads pictures (remembered once known); None when it cannot tell."""
        if self._reads_pictures is None:
            items = (answer.get("data") or answer.get("models")) if isinstance(answer, dict) else None
            for item in items or []:
                if isinstance(item, dict) and self.model in (item.get("id"), item.get("key")):
                    caps = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
                    self._reads_pictures = bool(caps.get("vision")) or item.get("type") == "vlm"
                    break
        return self._reads_pictures

    def reads_pictures(self) -> bool | None:
        return self._vision(self._list())

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with self._open(request, FIRST_CALL_TIMEOUT if self._is_first_call else LOCAL_TIMEOUT) as response:
            return json.loads(response.read())

    def _build(self) -> int | None:
        """The llama.cpp build behind the server, from a llama-server style GET /props (read-only); None when it does not say."""
        try:
            with self._open(urllib.request.Request(self.base_url + "/props"), PROPS_TIMEOUT) as response:
                data = json.loads(response.read())
        except BrainError:
            raise
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError, ValueError, AttributeError):
            return None
        return parse_build(data.get("build_info")) if isinstance(data, dict) else None

    def _room_check(self, picture: bool) -> int | None:
        """Gotcha 4: the context per slot must cover the picture, the prompt, the answer and the thinking. Returns the tokens left
        for the answer (None when the engine's context figures are not known); raises context_too_small before any request."""
        if not self.context_total:
            return None
        per_slot = context_per_slot(self.context_total, self.slots)
        thinking = THINKING_TOKENS[REASONING_BY_PART.get(self.part, "low")] if is_glimmer(self.model) else 0
        need = (PICTURE_TOKENS if picture else 0) + PROMPT_TOKENS + ANSWER_TOKENS + thinking
        if per_slot < need:
            raise BrainError("context_too_small", retry=False,
                             detail=f"{self.model}: {per_slot} tokens per slot, about {need} needed")
        return per_slot - (PICTURE_TOKENS if picture else 0) - PROMPT_TOKENS

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        require_loopback(self.base_url)  # before anything else, so a bad address sends nothing at all
        check_blocked(self.model)  # D90: by name, before ANY request, the list included
        # R224: room for the model? No chat/load request goes out until this passes. The read-only list (it loads nothing) gives the
        # model's real size and whether THIS model is already loaded; if it does not answer, the conservative guess is used.
        answer = self._list()
        size, loaded = pick_lmstudio(answer, self.model)
        self.found_bytes = self.model_bytes or size  # LOAD-1: the loading strip's estimate
        ensure_fits(self.model, self.model_bytes or size, self.context_tokens, has_vision=bool(images), reader=self._memory,
                    already_loaded=loaded)
        if images and self._vision(answer) is False:
            raise BrainError("model_text_only", retry=False, detail=self.model)  # never drop the picture quietly
        glimmer = is_glimmer(self.model)
        if glimmer:  # gotcha 6: an old llama.cpp refuses the files; ask (read-only) before anything is loaded
            build = self._build()
            if build is not None and build < MIN_GLIMMER_BUILD:
                raise BrainError("runtime_too_old", retry=False, detail=f"{self.model}: llama.cpp b{build}, needs b{MIN_GLIMMER_BUILD} or newer")
        room = self._room_check(bool(images))
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(image)).decode("ascii")}}
            for image in images
        ]
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "playfellow_reply", "strict": True, "schema": REPLY_SCHEMA}},
            "temperature": 0.2,
        }
        if self.ttl_seconds is not None:
            body["ttl"] = self.ttl_seconds
        stops = safe_stop(self.stop)  # gotcha 3: never <|eom|>
        if glimmer:
            stops = GLIMMER_STOPS + [s for s in (stops or []) if s not in GLIMMER_STOPS]
            strength = REASONING_BY_PART.get(self.part, "low")
            if not self._no_reasoning_field:  # dropped for the session once a server has rejected it (LM-7b)
                body["reasoning_effort"] = strength  # gotcha 5
        if stops:
            body["stop"] = stops
        limits = [room] if room is not None else []  # what is left of this slot's context after the picture and the prompt
        if glimmer:  # a ceiling ALWAYS: the thinking budget plus the answer, so no request can run away into endless thinking
            limits.append(THINKING_TOKENS[strength] + ANSWER_TOKENS)
        if limits:
            body["max_tokens"] = max(1, min(limits))
        start = begin(self)
        try:
            try:
                data = self._post(body)
            except urllib.error.HTTPError as exc:
                # LM-7b: `reasoning_effort` is an unverified field name. A 400 that names it is retried ONCE without it, and the field
                # is not sent again this session. Any other error is reported as before.
                if exc.code != 400 or "reasoning_effort" not in body or "reasoning_effort" not in _error_text(exc):
                    raise
                self._no_reasoning_field = True
                del body["reasoning_effort"]
                data = self._post(body)
            self._is_first_call = False
        except BrainError:
            raise
        except urllib.error.HTTPError as exc:
            raise classify_error(exc) from exc
        except (urllib.error.URLError, ConnectionError) as exc:
            raise BrainError("local_offline", retry=True, wait_seconds=5) from exc
        except Exception as exc:
            raise classify_error(exc) from exc
        try:
            choice = data["choices"][0]
            reply = choice["message"]["content"] or ""  # gotcha 1: ONLY the final answer; `reasoning_content` (the thinking) is never read
            ended = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BrainError("brain_trouble", retry=True, wait_seconds=3, detail="unexpected reply from LM Studio") from exc
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish(self, start, system, text, len(images), reply, prompt=usage.get("prompt_tokens"), completion=usage.get("completion_tokens"))
        if not reply.strip():  # gotcha 2: no answer and no error is a FAILURE (usually the thinking used up the room), never "do nothing"
            raise BrainError("empty_answer", retry=True, wait_seconds=3,
                             detail=self.model + (" (ran out of room while thinking)" if ended == "length" else ""))
        return reply
