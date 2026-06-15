from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class TeeEvent:
    ts: float
    kind: str  # "http" | "exchange" | "note"
    method: str | None = None
    path: str | None = None
    status: int | None = None
    client: str | None = None
    endpoint: str | None = None
    prompt: str | None = None
    response: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    streaming: bool = False
    done: bool = True
    text: str | None = None


class TeeBuffer:
    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque[TeeEvent] = deque(maxlen=maxlen)

    def push(self, event: TeeEvent) -> None:
        self._events.append(event)

    def recent(self, n: int) -> list[TeeEvent]:
        return list(self._events)[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self._events)
