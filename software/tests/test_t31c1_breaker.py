"""Tests for the circuit breaker (Task 31-C1)."""
from __future__ import annotations

import pytest

from brain import BrainError
from brain.breaker import CircuitBreaker


class FakeClock:
    """Testable clock for the circuit breaker."""
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_breaker_allows_initial_calls():
    """The breaker starts closed: calls go through."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    assert breaker.is_closed()
    assert not breaker.is_open()


def test_breaker_opens_after_two_consecutive_overload_errors():
    """After 2 consecutive 503/504 errors, the breaker opens."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    # First error: stays closed
    breaker.record_error(BrainError("busy", retry=True, wait_seconds=5))
    assert breaker.is_closed()

    # Second consecutive error: opens
    breaker.record_error(BrainError("busy", retry=True, wait_seconds=5))
    assert breaker.is_open()


def test_breaker_opens_after_two_consecutive_timeout_errors():
    """Timeout errors also open the breaker."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    breaker.record_error(TimeoutError("timeout"))
    breaker.record_error(TimeoutError("timeout"))
    assert breaker.is_open()


def test_breaker_resets_on_success():
    """A successful call resets the error counter."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.is_closed()

    breaker.record_success()

    # Now need 2 more errors to open
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.is_closed()


def test_breaker_does_not_open_on_non_overload_errors():
    """Only 503/504/timeout errors open the breaker."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    # Bad key error doesn't open the breaker
    breaker.record_error(BrainError("bad_key", retry=False))
    breaker.record_error(BrainError("bad_key", retry=False))

    assert breaker.is_closed()


def test_breaker_opens_permanently_on_quota_error():
    """A quota error opens the breaker until session ends."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    breaker.record_error(BrainError("quota", retry=False))

    assert breaker.is_open()
    assert breaker.is_quota_exhausted()

    # Even after cooldown time, stays open
    clock.advance(1000)
    assert breaker.is_open()


def test_breaker_closes_after_cooldown():
    """After cooldown period, the breaker half-opens."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    # Open the breaker
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.is_open()

    # First cooldown is 1 minute (60 seconds)
    clock.advance(59)
    assert breaker.is_open()

    clock.advance(1)
    assert not breaker.is_open()  # Can try again


def test_breaker_exponential_backoff():
    """Cooldown doubles on repeated failures, capped at 1 hour."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    # First trip: 10 min cooldown
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.cooldown_seconds() == 60

    clock.advance(60)

    # Second trip: 20 min cooldown
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.cooldown_seconds() == 120

    clock.advance(120)

    # Third trip: 40 min cooldown
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.cooldown_seconds() == 240

    clock.advance(240)

    # Fourth trip: capped at 10 min after 60/120/240/480 (600 seconds)
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    assert breaker.cooldown_seconds() == 480


def test_breaker_emits_tier_name():
    """The breaker knows which tier it's protecting."""
    clock = FakeClock()
    breaker = CircuitBreaker("local", clock=clock)

    assert breaker.tier_name == "local"


def test_breaker_state_description():
    """The breaker can describe its state for events."""
    clock = FakeClock()
    breaker = CircuitBreaker("cloud", clock=clock)

    # Closed
    assert "ready" in breaker.state_description().lower()

    # Open
    breaker.record_error(BrainError("busy", retry=True))
    breaker.record_error(BrainError("busy", retry=True))
    desc = breaker.state_description()
    assert "resting" in desc.lower()
    assert "cloud" in desc
