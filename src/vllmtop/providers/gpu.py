from __future__ import annotations

import shutil
import subprocess

from vllmtop.core.state import GpuSample, GpuSnapshot

_SMI_QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "power.draw,power.limit,clocks.sm,clocks.mem,fan.speed"
)


def _f(x: str) -> float | None:
    x = x.strip()
    if not x or x.upper() in ("N/A", "[N/A]"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_nvml(nvml: object) -> GpuSnapshot:
    nvml.nvmlInit()  # type: ignore[attr-defined]
    try:
        gpus: list[GpuSample] = []
        for i in range(nvml.nvmlDeviceGetCount()):  # type: ignore[attr-defined]
            h = nvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[attr-defined]
            util = nvml.nvmlDeviceGetUtilizationRates(h)  # type: ignore[attr-defined]
            mem = nvml.nvmlDeviceGetMemoryInfo(h)  # type: ignore[attr-defined]
            name = nvml.nvmlDeviceGetName(h)  # type: ignore[attr-defined]
            if isinstance(name, bytes):
                name = name.decode()

            def _try(fn, *a):  # type: ignore[no-untyped-def]
                try:
                    return fn(*a)
                except Exception:  # noqa: BLE001 - optional metric not supported
                    return None

            power = _try(nvml.nvmlDeviceGetPowerUsage, h)  # type: ignore[attr-defined]
            limit = _try(nvml.nvmlDeviceGetEnforcedPowerLimit, h)  # type: ignore[attr-defined]
            gpus.append(
                GpuSample(
                    index=i,
                    name=name,
                    util_gpu=float(util.gpu),
                    mem_used=int(mem.used),
                    mem_total=int(mem.total),
                    temp_c=_try(nvml.nvmlDeviceGetTemperature, h, nvml.NVML_TEMPERATURE_GPU),  # type: ignore[attr-defined]
                    power_w=(power / 1000.0) if power is not None else None,
                    power_limit_w=(limit / 1000.0) if limit is not None else None,
                    fan_pct=_try(nvml.nvmlDeviceGetFanSpeed, h),  # type: ignore[attr-defined]
                    clock_sm_mhz=_try(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_SM),  # type: ignore[attr-defined]
                    clock_mem_mhz=_try(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_MEM),  # type: ignore[attr-defined]
                )
            )
        return GpuSnapshot(available=True, source="nvml", gpus=gpus)
    finally:
        try:
            nvml.nvmlShutdown()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def parse_nvidia_smi_csv(text: str) -> list[GpuSample]:
    gpus: list[GpuSample] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        mu, mt = _f(parts[3]), _f(parts[4])
        clk_sm, clk_mem = _f(parts[8]), _f(parts[9])
        gpus.append(
            GpuSample(
                index=int(_f(parts[0]) or 0),
                name=parts[1],
                util_gpu=_f(parts[2]),
                mem_used=int(mu * 1024 * 1024) if mu is not None else None,
                mem_total=int(mt * 1024 * 1024) if mt is not None else None,
                temp_c=_f(parts[5]),
                power_w=_f(parts[6]),
                power_limit_w=_f(parts[7]),
                clock_sm_mhz=int(clk_sm) if clk_sm is not None else None,
                clock_mem_mhz=int(clk_mem) if clk_mem is not None else None,
                fan_pct=_f(parts[10]),
            )
        )
    return gpus


class GpuProvider:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._mode: str | None = None
        self._nvml: object | None = None

    def sample(self) -> GpuSnapshot:
        if not self.enabled:
            return GpuSnapshot(available=False, source="none", error="disabled")
        # try NVML
        if self._mode in (None, "nvml"):
            try:
                if self._nvml is None:
                    import pynvml  # nvidia-ml-py exposes the `pynvml` module

                    self._nvml = pynvml
                assert self._nvml is not None
                snap = read_nvml(self._nvml)
                self._mode = "nvml"
                return snap
            except Exception:  # noqa: BLE001 - fall back to nvidia-smi
                self._nvml = None
        # try nvidia-smi
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.run(
                    [smi, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                ).stdout
                self._mode = "nvidia-smi"
                return GpuSnapshot(
                    available=True, source="nvidia-smi", gpus=parse_nvidia_smi_csv(out)
                )
            except Exception:  # noqa: BLE001
                pass
        return GpuSnapshot(available=False, source="none", error="no NVML or nvidia-smi")
