"""PlayFellow's memory: the last few moments it saw, thought and did. / ذاكرة بلاي فيلو: آخر اللحظات التي رآها وفكر فيها وفعلها."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Moment:
    image: Image.Image
    thought: str
    actions: tuple[str, ...]
    said: str


class Memory:
    def __init__(self, size: int):
        self.size = size
        self._moments: deque[Moment] = deque(maxlen=max(0, size))

    def add(self, moment: Moment) -> None:
        self._moments.append(moment)

    def recent(self) -> list[Moment]:
        return list(self._moments)

    def clear(self) -> None:
        self._moments.clear()

    def as_text(self) -> str:
        moments = self.recent()
        lines = []
        for index, moment in enumerate(moments):
            ago = len(moments) - index
            line = (
                f"{ago} step{'s' if ago > 1 else ''} ago you thought: {moment.thought or '...'}"
                f" | you did: {', '.join(moment.actions) or 'nothing'}"
            )
            if moment.said:
                line += f' | you said: "{moment.said}"'
            lines.append(line)
        return "\n".join(lines)
