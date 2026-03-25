"""cute_comm.nccl — NCCL device-initiated API for CuTe Python DSL."""

from __future__ import annotations

from ._bridge import get_bridge_ptx
from ._device_ops import (
    gin_put,
    gin_put_sigadd,
    gin_put_siginc,
    gin_read_signal,
    gin_wait_signal,
    local_ptr,
    lsa_barrier,
    lsa_ptr,
)
from ._host import (
    create_dev_comm,
    destroy,
    destroy_dev_comm,
    deregister_window,
    finalize,
    get_unique_id,
    init_rank,
    query_properties,
    register_window,
)


def link_options(sm: str):
    """Return a ``LinkLibraries`` that includes the compiled NCCL bridge PTX.

    *sm* is the target architecture, e.g. ``"sm_90a"``.
    """
    from cutlass.base_dsl.compiler import LinkLibraries
    return LinkLibraries(get_bridge_ptx(sm))


__all__ = [
    # compile
    "link_options",
    # device ops
    "local_ptr",
    "lsa_ptr",
    "lsa_barrier",
    "gin_put",
    "gin_put_siginc",
    "gin_put_sigadd",
    "gin_read_signal",
    "gin_wait_signal",
    # host setup
    "get_unique_id",
    "init_rank",
    "query_properties",
    "create_dev_comm",
    "destroy_dev_comm",
    "register_window",
    "deregister_window",
    "finalize",
    "destroy",
]
