from PIL import Image

import brain.local as local


class FakeClient:
    def __init__(self, host=None, timeout=None):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": '{"thought": "ok", "say": "", "actions": []}'}}


def test_local_brain_turns_thinking_off_and_stays_loaded(monkeypatch):
    monkeypatch.setattr(local.ollama, "Client", FakeClient)
    brain = local.LocalBrain("gemma4:e2b")
    first_client = brain.client  # the first call uses a long cold-start timeout, then the client is swapped (T31-C2)
    reply = brain.ask("system", "turn", [Image.new("RGB", (8, 8))])
    call = first_client.calls[0]
    assert call["think"] is False
    assert call["keep_alive"] == local.KEEP_ALIVE == "30m"
    assert call["format"] == "json"
    assert '"thought"' in reply


# [RATCHET-LOCAL-MODEL-MISSING]
def test_missing_smart_local_model_falls_back_to_the_installed_one(monkeypatch):
    from brain import BrainError, FallbackBrain

    class Missing:
        def ask(self, system, text, images):
            raise BrainError("model_missing", retry=False, detail="gemma4:e4b")

    class Installed:
        def ask(self, system, text, images):
            return "ok"

    brain = FallbackBrain(Missing(), Installed())
    assert brain.ask("s", "t", []) == "ok" and brain.last_used == "backup"
    assert "gemma4:e4b" in BrainError("model_missing", detail="gemma4:e4b").friendly()


def test_local_brain_reports_a_missing_model(monkeypatch):
    from brain import BrainError

    class NotFound:
        def __init__(self, host=None, timeout=None):
            pass

        def chat(self, **kwargs):
            raise Exception("model 'gemma4:e4b' not found")

    monkeypatch.setattr(local.ollama, "Client", NotFound)
    try:
        local.LocalBrain("gemma4:e4b").ask("s", "t", [])
    except BrainError as error:
        assert error.key == "model_missing" and "ollama pull gemma4:e4b" in error.friendly()
    else:
        raise AssertionError("expected BrainError")
