"""Gemini brain (the default). / عقل Gemini (الاختيار الافتراضي)."""
from __future__ import annotations

from google import genai
from google.genai import types
from PIL import Image

from brain import network
from brain import ENDPOINTS, REQUEST_TIMEOUT, classify_error
from brain.meter import begin, finish


def _client_args(use_proxy: bool = False) -> dict:
    """CLIENT_ARGS plus our own trust store. Without a `verify` context, google-genai builds one from SSL_CERT_FILE and
    SSL_CERT_DIR (google/genai/_api_client.py:1208-1210), so the environment could vouch for any server [RATCHET-NO-ENV-CA]."""
    return network.client_args(use_proxy)  # CLIENT_ARGS, our own roots, and the opt-in proxy


class GeminiBrain:
    name = "gemini"

    def __init__(self, model: str, api_key: str, is_helper: bool = False, meter=None, use_proxy: bool = False):
        self.model = model
        self.meter = meter  # counts each question where it is asked (brain/meter.py)
        # client_args reach httpx.Client (google/genai/_api_client.py:916); trust_env=False ignores HTTP_PROXY and friends
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            base_url=ENDPOINTS["gemini"], timeout=REQUEST_TIMEOUT * 1000,
            client_args=_client_args(use_proxy), async_client_args=_client_args(use_proxy)))
        self.is_helper = is_helper

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        # Turn thinking off for the player brain (flash/flash-lite models only), helper keeps it
        thinking_config = None
        if not self.is_helper and ("flash" in self.model.lower()):
            try:
                thinking_config = types.ThinkingConfig(thinking_budget=0)
            except (TypeError, AttributeError):
                # Fallback: try thinking_level if thinking_budget isn't supported
                try:
                    thinking_config = types.ThinkingConfig(thinking_level="minimal")
                except (TypeError, AttributeError):
                    thinking_config = None

        start = begin(self)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[*images, text],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                    thinking_config=thinking_config,
                ),
            )
        except Exception as exc:
            # If we get a 400 error with thinking_config, retry once without it
            if thinking_config and getattr(exc, "code", None) == 400:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[*images, text],
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            temperature=0.2,
                            response_mime_type="application/json",
                            max_output_tokens=1024,
                        ),
                    )
                except Exception as retry_exc:
                    raise classify_error(retry_exc) from retry_exc
            else:
                raise classify_error(exc) from exc
        reply = response.text or ""
        counts = getattr(response, "usage_metadata", None)
        written = getattr(counts, "candidates_token_count", None)
        thoughts = getattr(counts, "thoughts_token_count", None) or 0  # thinking is billed as output
        finish(self, start, system, text, len(images), reply, prompt=getattr(counts, "prompt_token_count", None),
               completion=None if written is None else written + thoughts)
        return reply
