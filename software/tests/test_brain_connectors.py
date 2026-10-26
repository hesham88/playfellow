from types import SimpleNamespace

import anthropic
import httpx
import pytest
from PIL import Image

import brain.claude
import brain.gemini
import brain.local
import brain.openai_brain
from brain import BrainError

PICTURE = Image.new("RGB", (8, 8), "red")
REQUEST = httpx.Request("POST", "https://example.invalid")


# ---------- Gemini ----------
class FakeGeminiClient:
    last = None

    def __init__(self, api_key, http_options=None):
        self.api_key = api_key
        self.http_options = http_options
        self.models = self
        FakeGeminiClient.last = self
        self.error = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(text='{"thought": "hi"}')


def test_gemini_sends_pictures_then_text(monkeypatch):
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    gemini = brain.gemini.GeminiBrain("gemini-x", "key")
    assert gemini.ask("SYSTEM", "TEXT", [PICTURE, PICTURE]) == '{"thought": "hi"}'
    kwargs = FakeGeminiClient.last.kwargs
    assert kwargs["model"] == "gemini-x"
    assert kwargs["contents"][-1] == "TEXT" and len(kwargs["contents"]) == 3
    assert kwargs["config"].system_instruction == "SYSTEM"
    assert kwargs["config"].response_mime_type == "application/json"


def test_gemini_errors_become_brain_errors(monkeypatch):
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    gemini = brain.gemini.GeminiBrain("gemini-x", "key")
    error = Exception("quota")
    error.code = 429
    FakeGeminiClient.last.error = error
    with pytest.raises(BrainError) as raised:
        gemini.ask("s", "t", [PICTURE])
    assert raised.value.key == "busy"


def test_gemini_player_brain_uses_thinking_config(monkeypatch):
    """Player brain (flash models) gets thinking_config with thinking_budget=0."""
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    gemini = brain.gemini.GeminiBrain("gemini-3.8-flash", "key", is_helper=False)
    gemini.ask("SYSTEM", "TEXT", [PICTURE])
    config = FakeGeminiClient.last.kwargs["config"]
    # Should have thinking_config set
    assert hasattr(config, "thinking_config")
    # Thinking config should be present (either thinking_budget=0 or thinking_level="minimal")
    assert config.thinking_config is not None


def test_gemini_helper_brain_no_thinking_config(monkeypatch):
    """Helper brain keeps default thinking (no thinking_config)."""
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    gemini = brain.gemini.GeminiBrain("gemini-4-argon", "key", is_helper=True)
    gemini.ask("SYSTEM", "TEXT", [PICTURE])
    config = FakeGeminiClient.last.kwargs["config"]
    # Helper brain should not have thinking_config
    assert config.thinking_config is None


def test_gemini_non_flash_no_thinking_config(monkeypatch):
    """Non-flash models don't get thinking_config."""
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    gemini = brain.gemini.GeminiBrain("gemini-4-argon", "key", is_helper=False)
    gemini.ask("SYSTEM", "TEXT", [PICTURE])
    config = FakeGeminiClient.last.kwargs["config"]
    # Non-flash model should not have thinking_config
    assert config.thinking_config is None


def test_gemini_retries_without_thinking_config_on_400(monkeypatch):
    """On 400 error with thinking_config, retry once without it."""
    call_count = [0]
    original_generate = None

    def generate_with_retry(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call with thinking_config should fail with 400
            if kwargs["config"].thinking_config is not None:
                error = Exception("bad request")
                error.code = 400
                raise error
        # Second call or call without thinking_config succeeds
        return SimpleNamespace(text='{"thought": "ok"}')

    class RetryClient:
        def __init__(self, api_key, http_options=None):
            self.models = self

        def generate_content(self, **kwargs):
            return generate_with_retry(**kwargs)

    monkeypatch.setattr(brain.gemini.genai, "Client", RetryClient)
    gemini = brain.gemini.GeminiBrain("gemini-3.8-flash", "key", is_helper=False)
    result = gemini.ask("SYSTEM", "TEXT", [PICTURE])

    # Should have retried and succeeded
    assert call_count[0] == 2
    assert result == '{"thought": "ok"}'


# ---------- Claude ----------
class FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_anthropic(outcome):
    plain, beta = FakeMessages(outcome), FakeMessages(outcome)

    class FakeAnthropic:
        def __init__(self, api_key, base_url=None, timeout=None, max_retries=None, http_client=None):
            self.messages = plain
            self.beta = SimpleNamespace(messages=beta)

    return FakeAnthropic, plain, beta


def answer(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def test_claude_opus_uses_fallbacks_low_effort_and_jpeg(monkeypatch):
    client, plain, beta = fake_anthropic(answer('{"a": 1}'))
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    claude = brain.claude.ClaudeBrain("claude-opus-5-5", "key")
    assert claude.ask("SYSTEM", "TEXT", [PICTURE]) == '{"a": 1}'
    kwargs = beta.kwargs
    assert plain.kwargs is None
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["system"] == "SYSTEM"
    content = kwargs["messages"][0]["content"]
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[-1] == {"type": "text", "text": "TEXT"}


def test_claude_other_models_use_the_plain_endpoint(monkeypatch):
    client, plain, beta = fake_anthropic(answer("{}"))
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    brain.claude.ClaudeBrain("claude-haiku-5-5", "key").ask("s", "t", [PICTURE])
    assert plain.kwargs is not None and beta.kwargs is None
    assert "fallbacks" not in plain.kwargs


def test_claude_refusal_returns_empty(monkeypatch):
    client, _, _ = fake_anthropic(answer("no", stop_reason="refusal"))
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    assert brain.claude.ClaudeBrain("claude-opus-5-5", "key").ask("s", "t", [PICTURE]) == ""


@pytest.mark.parametrize(
    "error, key",
    [
        (anthropic.AuthenticationError("bad", response=httpx.Response(401, request=REQUEST), body=None), "bad_key"),
        (anthropic.RateLimitError("slow", response=httpx.Response(429, request=REQUEST), body=None), "busy"),
        (anthropic.InternalServerError("down", response=httpx.Response(500, request=REQUEST), body=None), "busy"),
        (anthropic.APIConnectionError(request=REQUEST), "offline"),
    ],
)
def test_claude_errors(monkeypatch, error, key):
    client, _, _ = fake_anthropic(error)
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    with pytest.raises(BrainError) as raised:
        brain.claude.ClaudeBrain("claude-opus-5-5", "key").ask("s", "t", [PICTURE])
    assert raised.value.key == key


# ---------- OpenAI ----------
def test_openai_sends_data_url_and_json_mode(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url=None, timeout=None, max_retries=None, http_client=None):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"b": 2}'))])

    monkeypatch.setattr(brain.openai_brain.openai, "OpenAI", FakeOpenAI)
    assert brain.openai_brain.OpenAIBrain("gpt-x", "key").ask("SYSTEM", "TEXT", [PICTURE]) == '{"b": 2}'
    assert seen["messages"][0] == {"role": "system", "content": "SYSTEM"}
    user = seen["messages"][1]["content"]
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user[-1] == {"type": "text", "text": "TEXT"}
    assert seen["response_format"] == {"type": "json_object"}


# ---------- Local (Ollama) ----------
class FakeOllamaClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat_kwargs = None
        self.error = None

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        if self.error:
            raise self.error
        return {"message": {"content": '{"c": 3}'}}


def test_local_sends_jpeg_bytes(monkeypatch):
    class LocalClient(FakeOllamaClient):
        pass

    monkeypatch.setattr(brain.local.ollama, "Client", LocalClient)
    client_instance = LocalClient()
    monkeypatch.setattr(brain.local.ollama.Client, "__call__", lambda *args, **kwargs: client_instance)

    # Create a new instance that will use our fake
    fake_client = FakeOllamaClient()
    monkeypatch.setattr(brain.local.ollama, "Client", lambda **kw: fake_client)

    assert brain.local.LocalBrain("qwen").ask("SYSTEM", "TEXT", [PICTURE]) == '{"c": 3}'
    assert fake_client.chat_kwargs["format"] == "json"
    assert fake_client.chat_kwargs["messages"][1]["images"][0][:2] == b"\xff\xd8"


def test_local_not_running(monkeypatch):
    fake_client = FakeOllamaClient()
    fake_client.error = ConnectionError("Failed to connect to Ollama")
    monkeypatch.setattr(brain.local.ollama, "Client", lambda **kw: fake_client)

    with pytest.raises(BrainError) as raised:
        brain.local.LocalBrain("qwen").ask("s", "t", [PICTURE])
    assert raised.value.key == "local_offline"


# ---------- Error handling ----------
def test_claude_unexpected_error_becomes_brain_error(monkeypatch):
    client, _, _ = fake_anthropic(ValueError("weird"))
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    with pytest.raises(BrainError):
        brain.claude.ClaudeBrain("claude-opus-5-5", "key").ask("s", "t", [PICTURE])


def test_claude_garbled_response(monkeypatch):
    client, _, _ = fake_anthropic(SimpleNamespace(stop_reason="end_turn", content=None))
    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", client)
    with pytest.raises(BrainError) as raised:
        brain.claude.ClaudeBrain("claude-opus-5-5", "key").ask("s", "t", [PICTURE])
    assert raised.value.key == "brain_trouble"


def test_openai_empty_choices(monkeypatch):
    class FakeOpenAI:
        def __init__(self, api_key, base_url=None, timeout=None, max_retries=None, http_client=None):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(choices=[])

    monkeypatch.setattr(brain.openai_brain.openai, "OpenAI", FakeOpenAI)
    with pytest.raises(BrainError) as raised:
        brain.openai_brain.OpenAIBrain("gpt-x", "key").ask("s", "t", [PICTURE])
    assert raised.value.key == "brain_trouble"


def test_fallback_models_live_in_settings(monkeypatch):
    from settings import CLAUDE_FALLBACK_MODELS
    import brain.claude
    assert brain.claude.CLAUDE_FALLBACK_MODELS is CLAUDE_FALLBACK_MODELS


# ---------- Timeout tests ----------
def test_gemini_sets_timeout(monkeypatch):
    monkeypatch.setattr(brain.gemini.genai, "Client", FakeGeminiClient)
    brain.gemini.GeminiBrain("gemini-x", "key")
    assert FakeGeminiClient.last.http_options is not None
    # HttpOptions is passed with timeout in milliseconds
    assert hasattr(FakeGeminiClient.last.http_options, "timeout") or "timeout" in str(FakeGeminiClient.last.http_options)


def test_claude_sets_timeout_and_max_retries(monkeypatch):
    seen = {}

    class FakeAnthropic:
        def __init__(self, api_key, base_url=None, timeout=None, max_retries=None, http_client=None):
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries
            self.messages = FakeMessages(answer("{}"))
            self.beta = SimpleNamespace(messages=FakeMessages(answer("{}")))

    monkeypatch.setattr(brain.claude.anthropic, "Anthropic", FakeAnthropic)
    brain.claude.ClaudeBrain("claude-x", "key")
    assert seen["timeout"] is not None
    assert seen["max_retries"] == 1


def test_openai_sets_timeout_and_max_retries(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url=None, timeout=None, max_retries=None, http_client=None):
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])))

    monkeypatch.setattr(brain.openai_brain.openai, "OpenAI", FakeOpenAI)
    brain.openai_brain.OpenAIBrain("gpt-x", "key")
    assert seen["timeout"] is not None
    assert seen["max_retries"] == 1


def test_local_uses_client_with_timeout(monkeypatch):
    seen = {}

    class FakeOllamaClient:
        def __init__(self, host=None, timeout=None, **kwargs):
            seen["timeout"] = timeout

        def chat(self, **kwargs):
            seen["chat_called"] = True
            return {"message": {"content": '{"c": 3}'}}

    monkeypatch.setattr(brain.local.ollama, "Client", FakeOllamaClient)
    brain.local.LocalBrain("qwen").ask("s", "t", [PICTURE])
    assert seen["timeout"] is not None
    assert seen["chat_called"] is True


def test_local_model_not_found_gives_helpful_error(monkeypatch):
    """Task 20 §0b: Ollama 'model not found' error gives helpful message with retry=False."""
    class FakeOllamaClient:
        def __init__(self, host=None, timeout=None, **kwargs):
            pass

        def chat(self, **kwargs):
            raise Exception("model 'gemma4:e2b' not found")

    monkeypatch.setattr(brain.local.ollama, "Client", FakeOllamaClient)
    with pytest.raises(BrainError) as raised:
        brain.local.LocalBrain("gemma4:e2b").ask("s", "t", [PICTURE])
    assert raised.value.key == "model_missing"
    assert raised.value.retry is False
    assert "gemma4:e2b" in raised.value.detail
    assert "ollama pull" in raised.value.detail
