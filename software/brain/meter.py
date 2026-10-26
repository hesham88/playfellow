"""Counting every AI question where it is asked. / عدّ كل سؤال للذكاء الاصطناعي في مكان طرحه.

Like a taxi meter that lives in the car, not in each passenger's pocket: each AI brain (Gemini, Claude, OpenAI, local)
holds its engine's meter, so no caller can forget it. The engine, the mind, the coach, the captain and the trainer all
ask through a brain. Before a question the meter asks the token budget; after the answer it writes the provider's own
token counts to the usage ledger. When a provider sends no counts, an estimate is written and marked as one. The
built-in brain is not an AI and costs nothing, so it has no meter.
"""
from __future__ import annotations

import threading
import time

IMAGE_TOKENS = 258  # an estimate per picture, used only when a provider sends no counts (a small picture on Gemini)
REPLY_KEPT = 2000  # characters of each reply kept in the ledger, so a runaway reply cannot bloat it
MINUTE_WAIT = 15  # seconds to wait when the runaway-minute limit is hit


def estimate(system: str, text: str, pictures: int, reply: str) -> tuple[int, int]:
    """(prompt, completion) tokens at about four characters a token and a fixed count a picture."""
    return (len(system) + len(text)) // 4 + pictures * IMAGE_TOKENS, max(1, len(reply) // 4)


def count(value) -> int | None:
    """A provider's token count as an int, or None when it sent none."""
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class Meter:
    """One engine's meter: the budget check before each question and the ledger row after it, plus running totals."""

    def __init__(self, ledger, governor=None, session: str = "", game: str = ""):
        self.ledger, self.governor, self.session, self.game = ledger, governor, session, game
        self._lock = threading.Lock()
        self.calls = self.tokens = self.estimated_calls = 0

    def before(self, brain: str) -> float:
        """Stop the question before it is sent when the budget says no. Returns the start time for after()."""
        over = self.governor.over_budget(brain) if self.governor is not None else None
        if over is not None:
            from brain import BrainError
            limit, used = over
            if limit == "minute":
                raise BrainError("token_cap_minute", retry=True, wait_seconds=MINUTE_WAIT, detail=f"{used:,}")
            raise BrainError("token_cap_day", retry=False, detail=f"{used:,}")
        return time.monotonic()

    def after(self, brain: str, model: str, prompt_tokens: int, completion_tokens: int, seconds: float, text: str,
              reply: str, estimated: bool = False) -> None:
        with self._lock:
            self.calls += 1
            self.tokens += prompt_tokens + completion_tokens
            self.estimated_calls += int(estimated)
        self.ledger.record_ai_call(session=self.session, brain=brain, model=model, prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens, latency_ms=round(seconds * 1000),
                                   prompt_text=text, response_text=reply[:REPLY_KEPT], armor_verdict="", game=self.game,
                                   estimated=estimated)
        if self.governor is not None:
            self.governor.counted(brain, prompt_tokens + completion_tokens)  # the budget sees it at once

    def totals(self) -> tuple[int, int, int]:
        """(questions, tokens, estimated questions) so far; the engine subtracts its round's start from these."""
        with self._lock:
            return self.calls, self.tokens, self.estimated_calls


def begin(brain) -> float:
    """Call at the top of an AI brain's ask(): checks the budget when the brain has a meter."""
    meter = getattr(brain, "meter", None)
    return meter.before(brain.name) if meter is not None else time.monotonic()


def finish(brain, start: float, system: str, text: str, pictures: int, reply: str, prompt=None, completion=None) -> None:
    """Call once an AI brain has its answer: writes the provider's counts, or an estimate marked as one."""
    meter = getattr(brain, "meter", None)
    if meter is None:
        return
    prompt, completion = count(prompt), count(completion)
    estimated = prompt is None or completion is None
    if estimated:
        guess = estimate(system, text, pictures, reply)
        prompt = guess[0] if prompt is None else prompt
        completion = guess[1] if completion is None else completion
    meter.after(brain.name, brain.model, prompt, completion, time.monotonic() - start, text, reply, estimated)
