"""Claude brain. / عقل Claude."""
from __future__ import annotations

import base64

import anthropic
from PIL import Image

from brain import REQUEST_TIMEOUT, BrainError, classify_error, to_jpeg_bytes
from settings import CLAUDE_FALLBACK_MODELS


class ClaudeBrain:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=1)

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(to_jpeg_bytes(image)).decode("utf-8"),
                },
            }
            for image in images
        ]
        content.append({"type": "text", "text": text})
        request = {
            "model": self.model,
            "max_tokens": 4000,
            "system": system,
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": content}],
        }
        try:
            if self.model in CLAUDE_FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                response = self.client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise BrainError("bad_key") from exc
        except anthropic.RateLimitError as exc:
            raise BrainError("busy", retry=True, wait_seconds=10) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise BrainError("busy", retry=True, wait_seconds=5) from exc
            raise BrainError("brain_trouble", detail=str(exc)[:120]) from exc
        except anthropic.APIConnectionError as exc:
            raise BrainError("offline", retry=True, wait_seconds=5) from exc
        except Exception as exc:
            raise classify_error(exc) from exc
        try:
            if response.stop_reason == "refusal":
                return ""
            return "".join(block.text for block in response.content if block.type == "text")
        except (AttributeError, TypeError, IndexError) as exc:
            raise BrainError("brain_trouble", retry=True, wait_seconds=3, detail="unexpected reply from the AI") from exc
