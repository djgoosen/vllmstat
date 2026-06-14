"""Dependency-free braille area plotter.

A braille cell packs 2 dots wide x 4 dots tall, giving high-fidelity little
plots inside a text panel. Each character is ``chr(0x2800 + mask)`` where
``mask`` is the OR of the set dot bits.

Dot-bit map within a cell (``cx`` in 0..1 left->right, ``cy`` in 0..3 top->bottom)::

    (0,0)=0x01  (1,0)=0x08
    (0,1)=0x02  (1,1)=0x10
    (0,2)=0x04  (1,2)=0x20
    (0,3)=0x40  (1,3)=0x80
"""

from __future__ import annotations

from collections.abc import Sequence

# bits[(cx, cy)] -> dot bit, cy is top(0)->bottom(3) within a cell.
_DOT_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}


def braille_plot(
    values: Sequence[float],
    width: int,
    height: int = 4,
    lo: float | None = None,
    hi: float | None = None,
) -> list[str]:
    """Render ``values`` as an area plot of braille characters.

    Returns exactly ``height`` strings, each exactly ``width`` characters.
    Larger values produce taller (top-reaching) columns. Never raises.
    """
    width = max(0, int(width))
    height = max(0, int(height))
    if width == 0 or height == 0:
        return [" " * width for _ in range(height)]

    dot_cols = width * 2
    dot_rows = height * 4
    blank = [" " * width for _ in range(height)]

    vals = [float(v) for v in values if v is not None]
    if not vals:
        return blank

    # Default y-range: clamp lo to 0 for non-negative data; otherwise use min.
    if lo is None:
        lo = 0.0 if min(vals) >= 0 else float(min(vals))
    else:
        lo = float(lo)
    if hi is None:
        hi = float(max(vals))
    else:
        hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo

    # Take the last dot_cols values; left-pad with the baseline (lo) if fewer.
    tail = list(vals[-dot_cols:])
    if len(tail) < dot_cols:
        tail = [lo] * (dot_cols - len(tail)) + tail

    # Per dot-column height in dots (0..dot_rows-1).
    col_d: list[int] = []
    top = dot_rows - 1
    for v in tail:
        d = round((v - lo) / span * (dot_rows - 1))
        if d < 0:
            d = 0
        elif d > top:
            d = top
        col_d.append(d)

    # Build the dot grid into per-cell masks, area-filling from the bottom.
    masks = [[0] * width for _ in range(height)]
    for dx in range(dot_cols):
        d = col_d[dx]
        cell_x = dx // 2
        cx = dx % 2
        # Fill dot-rows from the bottom (dot_rows-1) up to dot_rows-1-d inclusive.
        for dy in range(dot_rows - 1 - d, dot_rows):
            cell_y = dy // 4
            cy = dy % 4
            masks[cell_y][cell_x] |= _DOT_BITS[(cx, cy)]

    return ["".join(chr(0x2800 + m) for m in row) for row in masks]
