"""OpenAI brain. / عقل OpenAI."""
from __future__ import annotations

import base64

import openai
from PIL import Image

from brain import REQUEST_TIMEOUT, BrainError, classify_error, to_jpeg_bytes


class OpenAIBrain:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=1)

    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(image)).decode("ascii")},
            }
            for image in images
        ]
        content.append({"type": "text", "text": text})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise classify_error(exc) from exc
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError, TypeError) as exc:
            raise BrainError("brain_trouble", retry=True, wait_seconds=3, detail="unexpected reply from the AI") from exc
