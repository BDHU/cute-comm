"""NVCC bridge compilation and PTX caching."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import subprocess
import sysconfig
from pathlib import Path

from ._bridge_src import BRIDGE_SRC
from ._lib import get_nccl_include


def _nvcc_path() -> str:
    """Locate nvcc installed by the nvidia-cuda-nvcc wheel."""
    site_packages = Path(sysconfig.get_path("purelib"))
    dist = importlib.metadata.distribution("nvidia-cuda-nvcc")
    for f in dist.files:
        if f.name == "nvcc" and "/bin/" in str(f):
            candidate = site_packages / f
            if candidate.exists():
                return str(candidate.resolve())
    raise FileNotFoundError(
        "Could not find nvcc in the nvidia-cuda-nvcc distribution. "
        "Reinstall with: uv add nvidia-cuda-nvcc"
    )


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "cute_comm"


@functools.cache
def get_bridge_ptx(sm: str) -> str:
    key = hashlib.md5(f"{sm}\0{_nvcc_path()}\0{BRIDGE_SRC}".encode()).hexdigest()[:16]
    ptx = _cache_dir() / f"nccl_bridge_{key}.ptx"
    if ptx.exists():
        return str(ptx)

    src = ptx.with_suffix(".cu")
    ptx.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(BRIDGE_SRC, encoding="utf-8")

    subprocess.run(
        [
            _nvcc_path(),
            "-ptx",
            f"-arch={sm}",
            "-std=c++17",
            f"-I{get_nccl_include()}",
            str(src),
            "-o",
            str(ptx),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return str(ptx)
