from __future__ import annotations

import argparse
from dataclasses import dataclass

from vllmtop import __version__


@dataclass
class Config:
    url: str = "http://localhost:8000"
    metrics_path: str = "/metrics"
    interval: float = 1.0
    api_key: str | None = None
    gpu: bool = True
    mock: bool = False
    once: bool = False
    json: bool = False

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="vllmtop", description="nvtop for vLLM")
        p.add_argument("-u", "--url", default="http://localhost:8000")
        p.add_argument("--metrics-path", default="/metrics")
        p.add_argument("-i", "--interval", type=float, default=1.0)
        p.add_argument("--api-key", default=None)
        p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)
        p.add_argument("--mock", action="store_true", default=False)
        p.add_argument("--once", action="store_true", default=False)
        p.add_argument("--json", action="store_true", default=False)
        p.add_argument("--version", action="version", version=f"vllmtop {__version__}")
        return p

    @classmethod
    def from_sources(cls, argv: list[str], env: dict[str, str]) -> "Config":
        ns = cls.build_parser().parse_args(argv)
        api_key = ns.api_key or env.get("VLLM_API_KEY")
        return cls(
            url=ns.url, metrics_path=ns.metrics_path, interval=ns.interval,
            api_key=api_key, gpu=ns.gpu, mock=ns.mock, once=ns.once, json=ns.json,
        )
