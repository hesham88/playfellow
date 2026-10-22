"""Circuit breaker for brain tiers (Task 31-C1).

After N consecutive overload/timeout/quota errors on a tier, that tier is skipped for a cool-down
(exponential, capped). Quota errors open it until the session ends.
"""
from __future__ import annotations

import time
from typing import Callable

from brain import BrainError


class CircuitBreaker:
    """Protects a brain tier from repeated overload."""

    def __init__(self, tier_name: str, clock: Callable[[], float] | None = None):
        """Create a circuit breaker for a tier.

        Args:
            tier_name: Name of the tier being protected (for events).
            clock: Time source (monotonic). Injected for tests.
        """
        self.tier_name = tier_name
        self._clock = clock or time.monotonic
        self._consecutive_errors = 0
        self._trip_count = 0  # Number of times the breaker has opened
        self._opened_at: float | None = None
        self._quota_exhausted = False

    def record_error(self, error: Exception) -> None:
        """Record an error. Opens the breaker after 2 consecutive overload/timeout/quota errors."""
        # Check if this is an overload/timeout error
        if isinstance(error, BrainError):
            if error.key == "quota":
                # Quota errors open the breaker permanently
                self._quota_exhausted = True
                self._opened_at = self._clock()
                self._consecutive_errors = 0
                self._trip_count += 1
                return

            if error.key == "busy" or error.retry:
                # Overload or retryable error
                self._consecutive_errors += 1
                if self._consecutive_errors >= 2:
                    # Open the breaker
                    self._opened_at = self._clock()
                    self._consecutive_errors = 0
                    self._trip_count += 1
                return

        # Timeout errors
        if isinstance(error, TimeoutError):
            self._consecutive_errors += 1
            if self._consecutive_errors >= 2:
                self._opened_at = self._clock()
                self._consecutive_errors = 0
                self._trip_count += 1
            return

        # Other errors don't affect the breaker

    def record_success(self) -> None:
        """Record a successful call. Resets the error counter."""
        self._consecutive_errors = 0

    def is_open(self) -> bool:
        """True if the breaker is open (tier should be skipped)."""
        if self._quota_exhausted:
            return True

        if self._opened_at is None:
            return False

        # Check if cooldown has elapsed
        elapsed = self._clock() - self._opened_at
        if elapsed >= self.cooldown_seconds():
            # Cooldown elapsed, half-open (ready to try again)
            self._opened_at = None
            return False

        return True

    def is_closed(self) -> bool:
        """True if the breaker is closed (tier is ready)."""
        return not self.is_open()

    def is_quota_exhausted(self) -> bool:
        """True if quota error has opened the breaker permanently."""
        return self._quota_exhausted

    def cooldown_seconds(self) -> float:
        """Current cooldown period in seconds (exponential backoff, capped at 1 hour)."""
        if self._trip_count == 0:
            return 60.0  # 1 minute: overloads are usually brief (supervisor ruling 2026-10-16)

        # Exponential: 10 min, 20 min, 40 min, capped at 60 min
        seconds = 60.0 * (2 ** (self._trip_count - 1))
        return min(seconds, 600.0)  # cap at 10 minutes; quota errors stay open for the session

    def state_description(self) -> str:
        """Describe the breaker's state for events."""
        if self._quota_exhausted:
            return f"{self.tier_name} quota exhausted"
        if self.is_open():
            return f"{self.tier_name} resting"
        return f"{self.tier_name} ready"
