"""OpenAI brain. / عقل OpenAI."""
from __future__ import annotations

import base64

import openai
import httpx
from PIL import Image

from brain import network
from brain import ENDPOINTS, REQUEST_TIMEOUT, BrainError, classify_error, to_jpeg_bytes
from brain.meter import begin, finish


class OpenAIBrain:
    name = "openai"

    def __init__(self, model: str, api_key: str, meter=None, use_proxy: bool = False):
        self.model = model
        self.meter = meter  # counts each question where it is asked (brain/meter.py)
        # A plain httpx client with trust_env=False: no HTTP_PROXY and friends, no redirects [RATCHET-PINNED-ENDPOINTS]
        self.client = openai.OpenAI(api_key=api_key, base_url=ENDPOINTS["openai"], timeout=REQUEST_TIMEOUT, max_retries=1,
                                    http_client=httpx.Client(**network.client_args(use_proxy)))

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(image)).decode("ascii")},
            }
            for image in images
        ]
        content.append({"type": "text", "text": text})
        start = begin(self)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise classify_error(exc) from exc
        try:
            reply = response.choices[0].message.content or ""
        except (IndexError, AttributeError, TypeError) as exc:
            raise BrainError("brain_trouble", retry=True, wait_seconds=3, detail="unexpected reply from the AI") from exc
        usage = getattr(response, "usage", None)
        finish(self, start, system, text, len(images), reply, prompt=getattr(usage, "prompt_tokens", None),
               completion=getattr(usage, "completion_tokens", None))
        return reply
