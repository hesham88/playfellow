"""PlayFellow's brain: one way to ask any AI. / عقل بلاي فيلو: طريقة واحدة لسؤال أي ذكاء اصطناعي."""
from __future__ import annotations

import io
from typing import Protocol

from PIL import Image

from messages import msg
from settings import BRAIN_NAMES

REQUEST_TIMEOUT = 25  # seconds
LOCAL_TIMEOUT = 90  # seconds

# Where each brain sends the game's pictures and the player's key. Pinned on purpose (owner, 2026-10-23): the AI
# libraries would otherwise follow OPENAI_BASE_URL, ANTHROPIC_BASE_URL, GOOGLE_GEMINI_BASE_URL or OLLAMA_HOST, which any
# program on the PC can set. The cloud brains use HTTPS, and the local brain never leaves this PC.
ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/",
    "local": "http://127.0.0.1:11434",
}


class BrainError(Exception):
    """Something went wrong talking to the AI, with a friendly message."""

    def __init__(self, key: str, retry: bool = False, wait_seconds: float = 0.0, detail: str = ""):
        super().__init__(f"{key}: {detail}" if detail else key)
        self.key = key
        self.retry = retry
        self.wait_seconds = wait_seconds
        self.detail = detail

    def friendly(self) -> str:
        return msg(self.key, seconds=round(self.wait_seconds), detail=self.detail)


class Brain(Protocol):
    def ask(self, system: str, text: str, images: list[Image.Image]) -> str: ...


class FallbackBrain:
    """Ask the main brain; if it's busy or unreachable, ask the backup brain for this step."""

    def __init__(self, primary: Brain, backup: Brain, on_event=None, clock=None):
        self.primary, self.backup = primary, backup
        self.last_used = "primary"
        self._on_event = on_event or (lambda kind, data: None)
        self._last_model = getattr(primary, "model", "unknown")  # §6d: track actual model used

        # Circuit breakers for each tier
        from brain.breaker import CircuitBreaker
        self._primary_breaker = CircuitBreaker("primary", clock=clock)
        self._backup_breaker = CircuitBreaker("backup", clock=clock)

    @property
    def model(self) -> str:
        """Return the model of the last brain used (§6d)."""
        return self._last_model

    def ask(self, system: str, text: str, images: list) -> str:
        # T32-4 §7: Check if primary breaker is open (skip tier if not closed)
        if not self._primary_breaker.is_closed():
            # Emit quota-specific message if quota exhausted
            if self._primary_breaker.is_quota_exhausted():
                self._on_event("brain", "primary quota exhausted")
            else:
                self._on_event("brain", self._primary_breaker.state_description())
            # Use backup directly
            try:
                answer = self.backup.ask(system, text, images)
                self._backup_breaker.record_success()
                self.last_used = "backup"
                self._last_model = getattr(self.backup, "model", "unknown")  # §6d
                return answer
            except Exception as exc:
                error = exc if isinstance(exc, BrainError) else BrainError(
                    "brain_trouble", retry=True, wait_seconds=3, detail=str(exc)[:120]
                )
                self._backup_breaker.record_error(error)
                raise error

        try:
            answer = self.primary.ask(system, text, images)
            self._primary_breaker.record_success()
            self.last_used = "primary"
            self._last_model = getattr(self.primary, "model", "unknown")  # §6d
            return answer
        except Exception as exc:
            error = exc if isinstance(exc, BrainError) else BrainError(
                "brain_trouble", retry=True, wait_seconds=3, detail=str(exc)[:120]
            )
            self._primary_breaker.record_error(error)

            # Fall back on quota or retryable errors (treat it like a retryable error for fallback purposes)
            if not error.retry and error.key not in ("quota", "model_missing"):
                raise

            # T32-4 §7: If primary breaker just opened, emit event (quota-specific if quota exhausted)
            if not self._primary_breaker.is_closed():
                if self._primary_breaker.is_quota_exhausted():
                    self._on_event("brain", "primary quota exhausted")
                else:
                    self._on_event("brain", self._primary_breaker.state_description())

        # Try backup
        try:
            answer = self.backup.ask(system, text, images)
            self._backup_breaker.record_success()
            self.last_used = "backup"
            self._last_model = getattr(self.backup, "model", "unknown")  # §6d
            return answer
        except Exception as exc:
            backup_error = exc if isinstance(exc, BrainError) else BrainError(
                "brain_trouble", retry=True, wait_seconds=3, detail=str(exc)[:120]
            )
            self._backup_breaker.record_error(backup_error)
            raise backup_error


def make_brain(name: str, model: str, api_key: str | None, backup_model: str | None = None, is_helper: bool = False, on_event=None, clock=None, local_keep_alive: str = "30m", card=None, language: str = "english", meter=None) -> Brain:
    """One brain, or a main brain with a lighter backup. `meter` (brain/meter.py) counts every question each AI brain
    is asked; the built-in brain is not an AI and has none."""
    if name == "builtin":  # T31-day1-brain: plays offline with no key and no model; needs the game's controls card
        if card is None:
            raise BrainError("brain_trouble", detail="the built-in brain needs the controls card")
        from brain.builtin_nav import BuiltinNavBrain
        return BuiltinNavBrain(card, language)
    if name not in BRAIN_NAMES:
        raise BrainError("brain_trouble", detail=f"unknown brain '{name}'")
    if name != "local" and not api_key:
        raise BrainError("no_key")

    # Build the primary brain
    if name == "gemini":
        from brain.gemini import GeminiBrain
        primary = GeminiBrain(model, api_key, is_helper=is_helper, meter=meter)
    elif name == "claude":
        from brain.claude import ClaudeBrain
        primary = ClaudeBrain(model, api_key, meter=meter)
    elif name == "openai":
        from brain.openai_brain import OpenAIBrain
        primary = OpenAIBrain(model, api_key, meter=meter)
    else:
        from brain.local import LocalBrain
        primary = LocalBrain(model, keep_alive=local_keep_alive, meter=meter)

    # If backup_model is specified and different from model, return a FallbackBrain
    if backup_model and backup_model != model:
        if name == "gemini":
            from brain.gemini import GeminiBrain
            backup = GeminiBrain(backup_model, api_key, is_helper=is_helper, meter=meter)
        elif name == "claude":
            from brain.claude import ClaudeBrain
            backup = ClaudeBrain(backup_model, api_key, meter=meter)
        elif name == "openai":
            from brain.openai_brain import OpenAIBrain
            backup = OpenAIBrain(backup_model, api_key, meter=meter)
        else:
            from brain.local import LocalBrain
            backup = LocalBrain(backup_model, keep_alive=local_keep_alive, meter=meter)
        return FallbackBrain(primary, backup, on_event=on_event, clock=clock)

    return primary


def to_jpeg_bytes(image: Image.Image, quality: int = 80) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def classify_error(exc: Exception) -> BrainError:
    """Turn any AI service error into a friendly BrainError."""
    status = None
    words = [str(exc)]
    for name in ("status_code", "code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, str):
            words.append(value)
        if status is None:
            try:
                status = int(value)
            except (TypeError, ValueError):
                pass
    text = " ".join(words).lower()
    if status in (401, 403) or any(word in text for word in ("api key", "api_key", "permission", "unauthorized")):
        return BrainError("bad_key")
    # Check for quota exhaustion (different from rate limits)
    if any(word in text for word in ("exceeded your current quota", "insufficient_quota", "billing", "quota exceeded")):
        return BrainError("quota", retry=False)
    # Plain 429 rate limits are "busy" and retryable
    if status == 429 or any(word in text for word in ("rate limit", "resource_exhausted", "resource exhausted")):
        return BrainError("busy", retry=True, wait_seconds=10)
    if status == 504 or status is not None and status >= 500:
        return BrainError("busy", retry=True, wait_seconds=5)
    if status in (400, 404):
        return BrainError("brain_trouble", detail=str(exc)[:120])
    # Timeouts and deadlines mean the service is busy, not offline
    if isinstance(exc, TimeoutError) or any(word in text for word in ("timed out", "timeout", "deadline")):
        return BrainError("busy", retry=True, wait_seconds=3)
    # Connection failures are offline
    if isinstance(exc, ConnectionError) or any(word in text for word in ("connect", "getaddrinfo", "network")):
        return BrainError("offline", retry=True, wait_seconds=5)
    return BrainError("brain_trouble", retry=True, wait_seconds=3, detail=str(exc)[:120])
