from vllmtop.format import sparkline, fmt_si, fmt_bytes, fmt_dur, fmt_pct


def test_sparkline_basic():
    out = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
    assert len(out) == 8
    assert out[0] == "▁" and out[-1] == "█"


def test_sparkline_empty_and_flat():
    assert sparkline([]) == ""
    assert set(sparkline([5, 5, 5])) <= set("▁▂▃▄▅▆▇█")


def test_fmt_si():
    assert fmt_si(1500) == "1.5k"
    assert fmt_si(5_762_688) == "5.8M"
    assert fmt_si(42) == "42"


def test_fmt_bytes():
    assert fmt_bytes(24_000_000_000).endswith("GB") or fmt_bytes(24_000_000_000).endswith("GiB")


def test_fmt_dur():
    assert fmt_dur(0.073).endswith("ms")
    assert fmt_dur(1.8).endswith("s")
    assert fmt_dur(None) == "—"


def test_fmt_pct():
    assert fmt_pct(0.381) == "38.1%"
    assert fmt_pct(None) == "—"
