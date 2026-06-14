from vllmstat.providers.gpu import GpuProvider, parse_nvidia_smi_csv, read_nvml


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def nvmlInit(self): ...  # noqa: ANN201

    def nvmlShutdown(self): ...  # noqa: ANN201

    def nvmlDeviceGetCount(self):  # noqa: ANN201
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):  # noqa: ANN201
        return i

    def nvmlDeviceGetName(self, h):  # noqa: ANN201
        return "NVIDIA Test GPU"

    def nvmlDeviceGetUtilizationRates(self, h):  # noqa: ANN201
        class U:
            gpu = 81
            memory = 60

        return U()

    def nvmlDeviceGetMemoryInfo(self, h):  # noqa: ANN201
        class M:
            used = 23_100_000_000
            total = 24_000_000_000

        return M()

    def nvmlDeviceGetTemperature(self, h, sensor):  # noqa: ANN201
        return 61

    def nvmlDeviceGetPowerUsage(self, h):  # noqa: ANN201
        return 142_000  # mW

    def nvmlDeviceGetEnforcedPowerLimit(self, h):  # noqa: ANN201
        return 200_000

    def nvmlDeviceGetFanSpeed(self, h):  # noqa: ANN201
        return 45

    def nvmlDeviceGetClockInfo(self, h, c):  # noqa: ANN201
        return 2520 if c == 1 else 9501


def test_read_nvml_builds_samples():
    snap = read_nvml(FakeNvml())
    assert snap.available and snap.source == "nvml"
    g = snap.gpus[0]
    assert g.name == "NVIDIA Test GPU"
    assert g.util_gpu == 81
    assert g.mem_used == 23_100_000_000
    assert g.power_w == 142.0
    assert g.clock_sm_mhz == 2520


def test_parse_nvidia_smi_csv():
    line = "0, NVIDIA Test GPU, 81, 23100, 24000, 61, 142.0, 200.0, 2520, 9501, 45"
    gpus = parse_nvidia_smi_csv(line)
    g = gpus[0]
    assert g.index == 0 and g.name == "NVIDIA Test GPU"
    assert g.mem_used == 23100 * 1024 * 1024  # MiB -> bytes
    assert g.power_w == 142.0


def test_provider_disabled_returns_unavailable():
    snap = GpuProvider(enabled=False).sample()
    assert snap.available is False and snap.source == "none"
