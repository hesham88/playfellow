from safety import Safety


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_time_limit():
    clock = Clock()
    safety = Safety(max_minutes=1, max_ai_calls=100, clock=clock)
    safety.start()
    clock.now = 59
    assert safety.check() is None
    clock.now = 60
    assert safety.check() == "stopped_time" and safety.stopped


def test_call_limit():
    safety = Safety(max_minutes=5, max_ai_calls=2, clock=Clock())
    safety.start()
    safety.count_ai_call()
    assert safety.check() is None
    safety.count_ai_call()
    assert safety.check() == "stopped_calls"


def test_first_reason_wins():
    safety = Safety(5, 5, clock=Clock())
    safety.start()
    safety.stop("stopped_f12")
    safety.stop("stopped_button")
    assert safety.reason == "stopped_f12"


def test_start_resets():
    safety = Safety(5, 1, clock=Clock())
    safety.start()
    safety.count_ai_call()
    safety.check()
    safety.start()
    assert not safety.stopped and safety.calls == 0 and safety.check() is None


def test_wait_returns_early_when_stopped():
    safety = Safety(5, 5, clock=Clock())
    safety.start()
    assert safety.wait(0) is False
    safety.stop("stopped_button")
    assert safety.wait(10) is True
