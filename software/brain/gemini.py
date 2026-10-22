"""Gemini brain (the default). / عقل Gemini (الاختيار الافتراضي)."""
from __future__ import annotations

from google import genai
from google.genai import types
from PIL import Image

from brain import REQUEST_TIMEOUT, classify_error


class GeminiBrain:
    def __init__(self, model: str, api_key: str, is_helper: bool = False):
        self.model = model
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000))
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
        return response.text or ""
