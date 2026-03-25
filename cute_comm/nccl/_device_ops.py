"""CuTe DSL device-side wrappers via cute.ffi."""

from __future__ import annotations

from cutlass import cute
from cutlass import Int32, Int64, Uint32, Uint64

# ── Pointer access ──

local_ptr = cute.ffi(
    name="cute_nccl_local_ptr",
    params_types=[Uint64, Uint64],
    return_type=Uint64,
)

lsa_ptr = cute.ffi(
    name="cute_nccl_lsa_ptr",
    params_types=[Uint64, Uint64, Int32],
    return_type=Uint64,
)

# ── LSA barrier ──

lsa_barrier = cute.ffi(
    name="cute_nccl_lsa_barrier",
    params_types=[Uint64, Uint32],
)

# ── GIN put ──

gin_put = cute.ffi(
    name="cute_nccl_gin_put",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64],
)

gin_put_siginc = cute.ffi(
    name="cute_nccl_gin_put_siginc",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64, Uint32],
)

gin_put_sigadd = cute.ffi(
    name="cute_nccl_gin_put_sigadd",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64, Uint32, Uint64],
)

# ── GIN signals ──

gin_read_signal = cute.ffi(
    name="cute_nccl_gin_read_signal",
    params_types=[Uint64, Int32, Uint32],
    return_type=Uint64,
)

gin_wait_signal = cute.ffi(
    name="cute_nccl_gin_wait_signal",
    params_types=[Uint64, Int32, Uint32, Uint64],
)

