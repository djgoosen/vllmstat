from vllmstat.plot import braille_plot


def _dots(s: str) -> int:
    """Total number of set braille dots across a string."""
    return sum(bin(ord(c) - 0x2800).count("1") for c in s if 0x2800 <= ord(c) <= 0x28FF)


def test_returns_height_by_width():
    rows = braille_plot([1, 2, 3, 4, 5], width=12, height=4)
    assert len(rows) == 4
    assert all(len(r) == 12 for r in rows)


def test_default_height_is_four():
    rows = braille_plot([1, 2, 3], width=8)
    assert len(rows) == 4


def test_rising_ramp_later_columns_taller():
    width = 20
    rows = braille_plot(list(range(width * 2)), width=width, height=4)
    # Compare the first vs last character-column (stack of `height` chars).
    first_col = "".join(r[0] for r in rows)
    last_col = "".join(r[-1] for r in rows)
    assert _dots(last_col) > _dots(first_col)


def test_ramp_is_monotonic_nondecreasing_across_columns():
    width = 16
    rows = braille_plot(list(range(width * 2)), width=width, height=4)
    counts = [_dots("".join(r[i] for r in rows)) for i in range(width)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_flat_series_is_consistent():
    rows = braille_plot([5, 5, 5, 5, 5, 5], width=6, height=4)
    assert len(rows) == 4
    assert all(len(r) == 6 for r in rows)
    # The data columns (rightmost) are full-height and identical to each other.
    last_col = "".join(r[-1] for r in rows)
    second_last = "".join(r[-2] for r in rows)
    assert last_col == second_last
    # A flat non-zero series fills the full height at the data columns: each of
    # the `height` cells is a full braille block (8 dots) -> height * 8 dots.
    assert _dots(last_col) == 4 * 8


def test_empty_returns_blank_lines():
    rows = braille_plot([], width=10, height=4)
    assert rows == [" " * 10] * 4


def test_empty_default_height():
    rows = braille_plot([], width=7)
    assert rows == [" " * 7] * 4


def test_single_value_at_hi_exact_mask():
    # width=1, height=1 -> grid is 2 dot-cols x 4 dot-rows.
    # One value (==hi) goes in the rightmost dot-col, filling all 4 rows.
    # The left dot-col is baseline (lo) -> only the bottom dot.
    #   left  col: cy=3            -> bit 0x40
    #   right col: cy=0,1,2,3      -> 0x08|0x10|0x20|0x80 = 0xB8
    #   mask = 0x40 | 0xB8 = 0xF8
    rows = braille_plot([1.0], width=1, height=1, lo=0.0, hi=1.0)
    assert rows == [chr(0x2800 + 0xF8)]


def test_never_raises_on_weird_input():
    # zero width / height, negative-ish, all None-like guarded by caller use of floats
    assert braille_plot([1, 2, 3], width=0, height=4) == ["", "", "", ""]
    assert braille_plot([1, 2, 3], width=5, height=0) == []
    # single value, default range (hi<=lo guarded) must not raise
    rows = braille_plot([7.0], width=4, height=4)
    assert len(rows) == 4 and all(len(r) == 4 for r in rows)


def test_explicit_lo_hi_clamps():
    # Values above hi clamp to full height, below lo clamp to baseline.
    # width=2 -> 4 dot-cols; tail left-pads to [lo, lo, 100, -100].
    # The right cell holds dot-cols (100 -> full, -100 -> bottom), so it has
    # more dots than the left cell which is all baseline (bottom only).
    rows = braille_plot([100, -100], width=2, height=4, lo=0.0, hi=10.0)
    assert len(rows) == 4 and all(len(r) == 2 for r in rows)
    left_col = "".join(r[0] for r in rows)  # baseline only
    right_col = "".join(r[-1] for r in rows)  # includes the clamped-high value
    assert _dots(right_col) > _dots(left_col)
