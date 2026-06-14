from __future__ import annotations

from pathlib import Path

from vllmstat.providers.gpu_intel import pdev_for_card, read_intel_sysfs


def _make_intel_card(
    tmp_path: Path,
    *,
    hwmon_name: str = "xe",
    energy_uj: int = 84023575462341,
    temp2: int | None = 57000,
    temp1: int | None = None,
) -> Path:
    """Build a fake Intel xe sysfs tree mirroring the real Battlemage layout."""
    card = tmp_path / "card0"
    dev = card / "device"
    # freq: tile0/gt0/freq0/cur_freq
    freq = dev / "tile0" / "gt0" / "freq0"
    freq.mkdir(parents=True)
    (freq / "cur_freq").write_text("2800\n")
    (freq / "max_freq").write_text("2800\n")
    # hwmon
    hw = dev / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (hw / "name").write_text(hwmon_name + "\n")
    if temp2 is not None:
        (hw / "temp2_input").write_text(f"{temp2}\n")
        (hw / "temp2_label").write_text("pkg\n")
    if temp1 is not None:
        (hw / "temp1_input").write_text(f"{temp1}\n")
    (hw / "temp3_input").write_text("52000\n")
    (hw / "temp3_label").write_text("vram\n")
    (hw / "energy1_input").write_text(f"{energy_uj}\n")
    (hw / "energy1_label").write_text("card\n")
    (hw / "power1_cap").write_text("275000000\n")  # micro-W -> 275 W
    (hw / "fan1_input").write_text("1060\n")  # RPM
    return card


def test_read_intel_sysfs_temp_freq_fan_limit(tmp_path: Path):
    card = _make_intel_card(tmp_path)
    g, energy = read_intel_sysfs(str(card), prev_energy=None, now=100.0)
    assert g.vendor == "intel"
    assert g.clock_sm_mhz == 2800
    assert g.temp_c == 57.0  # temp2_input (pkg) / 1000
    assert g.power_limit_w == 275.0
    assert g.fan_rpm == 1060
    # util and VRAM are not available on the xe driver
    assert g.util_gpu is None
    assert g.mem_used is None and g.mem_total is None
    # energy carried forward for the next power delta
    assert energy == (84023575462341, 100.0)


def test_read_intel_sysfs_power_none_on_first_call(tmp_path: Path):
    card = _make_intel_card(tmp_path)
    g, _ = read_intel_sysfs(str(card), prev_energy=None, now=100.0)
    assert g.power_w is None  # need two samples for a delta


def test_read_intel_sysfs_power_from_energy_delta(tmp_path: Path):
    # 60 J consumed over 2 s -> 30 W.  60 J == 60_000_000 µJ.
    e1 = 84_000_000_000_000
    card = _make_intel_card(tmp_path, energy_uj=e1 + 60_000_000)
    g, energy = read_intel_sysfs(str(card), prev_energy=(e1, 100.0), now=102.0)
    assert g.power_w is not None
    assert abs(g.power_w - 30.0) < 1e-6
    assert energy == (e1 + 60_000_000, 102.0)


def test_read_intel_sysfs_temp1_fallback(tmp_path: Path):
    # No temp2_input -> fall back to temp1_input.
    card = _make_intel_card(tmp_path, temp2=None, temp1=49000)
    g, _ = read_intel_sysfs(str(card), prev_energy=None, now=1.0)
    assert g.temp_c == 49.0


def test_read_intel_sysfs_zero_dt_yields_no_power(tmp_path: Path):
    e1 = 84_000_000_000_000
    card = _make_intel_card(tmp_path, energy_uj=e1 + 1_000_000)
    g, _ = read_intel_sysfs(str(card), prev_energy=(e1, 100.0), now=100.0)
    assert g.power_w is None  # dt == 0 must not divide-by-zero


def test_read_intel_sysfs_missing_energy_keeps_prev(tmp_path: Path):
    """If energy can't be read, power is None and carry stays None (no crash)."""
    card = tmp_path / "card0"
    (card / "device").mkdir(parents=True)
    g, energy = read_intel_sysfs(str(card), prev_energy=(123, 1.0), now=2.0)
    assert g.power_w is None
    assert energy is None


def test_pdev_for_card_resolves_pci_address(tmp_path: Path):
    # card0/device -> symlink into the PCI tree; basename of its realpath is pdev.
    pci = tmp_path / "sys" / "devices" / "pci0000:00" / "0000:06:00.0"
    pci.mkdir(parents=True)
    card = tmp_path / "sys" / "class" / "drm" / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci)
    assert pdev_for_card(str(card)) == "0000:06:00.0"


def test_pdev_for_card_missing_device_returns_none_or_str(tmp_path: Path):
    # A card dir with no device link must not raise.
    card = tmp_path / "card0"
    card.mkdir()
    # realpath of a non-existent path yields its basename ("device"); never raises.
    assert pdev_for_card(str(card)) in ("device", None)
