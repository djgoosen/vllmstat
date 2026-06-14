"""Intel GPU backend for the ``xe`` (and ``i915``) drivers via sysfs.

The ``xe`` driver exposes no ``gpu_busy_percent`` and no ``mem_info_vram_*`` in
sysfs, so util% and VRAM come from the DRM ``fdinfo`` aggregator (see
``gpu_fdinfo``) rather than from here. What this module reads out of the box:
GPU clock, package temperature, fan RPM, power cap, and a power figure derived
from the ``energy1_input`` counter delta. It also resolves a card's PCI address
(``pdev``) so the fdinfo aggregator can be pointed at the right device.

Every read catches OS errors and degrades to ``None``; nothing here ever raises.
"""

from __future__ import annotations

import glob
import os

from vllmstat.core.state import GpuSample
from vllmstat.providers.gpu_sysfs import pci_name, read_int, read_text

# Energy-delta carry: (energy_microjoules, monotonic_seconds).
EnergyState = tuple[int, float]


def _hwmon_dir(card_path: str) -> str | None:
    """Return the xe/i915 hwmon dir (else the first hwmon under the card)."""
    base = os.path.join(card_path, "device", "hwmon")
    try:
        candidates = sorted(glob.glob(os.path.join(base, "hwmon*")))
    except OSError:
        return None
    if not candidates:
        return None
    for hw in candidates:
        if read_text(os.path.join(hw, "name")) in ("xe", "i915"):
            return hw
    return candidates[0]


def _div(path: str, denom: float) -> float | None:
    val = read_int(path)
    return (val / denom) if val is not None else None


def read_intel_sysfs(
    card_path: str,
    prev_energy: EnergyState | None,
    now: float,
) -> tuple[GpuSample, EnergyState | None]:
    """Build a GpuSample from Intel xe/i915 sysfs.

    ``prev_energy`` is the ``(energy_uj, time)`` from the previous sample (or
    ``None`` on the first call). Power is computed as
    ``(e - e_prev) / 1e6 / (now - t_prev)`` and ``None`` when there is no prior
    sample or no time elapsed. Returns ``(sample, new_energy_state)`` where the
    caller carries the state forward per card.
    """
    dev = os.path.join(card_path, "device")

    # GPU clock (MHz): tile0/gt0/freq0/cur_freq.
    clock_sm = read_int(os.path.join(dev, "tile0", "gt0", "freq0", "cur_freq"))

    temp_c = power_limit_w = None
    fan_rpm = None
    new_energy: EnergyState | None = None
    power_w: float | None = None

    hw = _hwmon_dir(card_path)
    if hw is not None:
        # Package temp (temp2_input); fall back to temp1_input.
        temp_c = _div(os.path.join(hw, "temp2_input"), 1000.0)
        if temp_c is None:
            temp_c = _div(os.path.join(hw, "temp1_input"), 1000.0)
        power_limit_w = _div(os.path.join(hw, "power1_cap"), 1e6)
        fan_rpm = read_int(os.path.join(hw, "fan1_input"))

        energy = read_int(os.path.join(hw, "energy1_input"))
        if energy is not None:
            new_energy = (energy, now)
            if prev_energy is not None:
                e_prev, t_prev = prev_energy
                dt = now - t_prev
                if dt > 0:
                    power_w = (energy - e_prev) / 1e6 / dt

    return (
        GpuSample(
            index=0,
            name=pci_name(card_path),
            vendor="intel",
            util_gpu=None,  # not available on xe
            mem_used=None,  # not available on xe
            mem_total=None,
            temp_c=temp_c,
            power_w=power_w,
            power_limit_w=power_limit_w,
            fan_rpm=fan_rpm,
            clock_sm_mhz=clock_sm,
        ),
        new_energy,
    )


def pdev_for_card(card_path: str) -> str | None:
    """Return the PCI address (``pdev``) backing ``card_path``, e.g.
    ``0000:06:00.0``.

    ``<card_path>/device`` is a symlink into the PCI tree; its realpath
    basename is the PCI bus address that ``fdinfo`` reports as ``drm-pdev``.
    Returns ``None`` when the link can't be resolved. Never raises.
    """
    try:
        target = os.path.realpath(os.path.join(card_path, "device"))
    except OSError:
        return None
    name = os.path.basename(target)
    return name or None
