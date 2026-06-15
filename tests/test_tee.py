from __future__ import annotations


def test_tee_event_defaults():
    from vllmstat.core.tee import TeeEvent

    e = TeeEvent(ts=1.0, kind="http", method="POST", path="/v1/chat/completions", status=200)
    assert e.prompt is None and e.done is True and e.client is None


def test_tee_buffer_recent_and_maxlen():
    from vllmstat.core.tee import TeeBuffer, TeeEvent

    b = TeeBuffer(maxlen=3)
    for i in range(5):
        b.push(TeeEvent(ts=float(i), kind="note", text=str(i)))
    assert len(b) == 3
    assert [e.text for e in b.recent(2)] == ["3", "4"]  # newest last
    assert b.recent(0) == []
