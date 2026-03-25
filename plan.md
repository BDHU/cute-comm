# cute-comm: NCCL device-initiated API for CuTe Python DSL

## Direction

Add NCCL device-initiated primitives callable from CuTe Python DSL `@cute.kernel`
functions by compiling a small NVCC bridge and linking its PTX into the CuTe compile
pipeline via `cute.LinkLibraries`.

Use `cutlass.cute` as the user-facing DSL entry point, `cute.ffi` for device-call
declarations, and opaque 64-bit values to represent NCCL handles on the Python/CuTe
side. Cast those opaque values back to NCCL types inside the bridge.

Include `lsa_barrier` in v1 because phase-correct LSA kernels need it. Include GIN
signals in v1. Add `gin_barrier` only if the first target kernel requires a true
cross-rank phase barrier instead of point-to-point completion ordering.

## Why This Shape

NCCL already exposes the important team derivations on device:

- `ncclTeamWorld(dev_comm)`
- `ncclTeamLsa(dev_comm)`
- `ncclTeamRail(dev_comm)`

That means the bridge does not need to marshal NCCL team structs through Python.
The bridge can take only opaque handles and scalar indices, derive the team locally,
and call the actual NCCL templated API.

This keeps the Python surface small and avoids trying to model NCCL internal structs
inside the DSL type system.

## Project Structure

```text
cute_comm/
├── __init__.py
└── nccl/
    ├── __init__.py      # user API + link_options()
    ├── _lib.py          # NCCL path discovery, nccl.so loading
    ├── _bridge.py       # NVCC bridge compile + cache
    ├── _bridge_src.py   # BRIDGE_SRC extern "C" __device__ wrappers
    ├── _host.py         # ctypes host setup: comm, dev comm, windows, optional barriers
    └── _device_ops.py   # cute.ffi wrappers
```

## V1 Scope

### Required in v1

- `local_ptr`
- `lsa_ptr`
- `lsa_barrier`
- `gin_put`
- `gin_put_siginc`
- `gin_put_sigadd`
- `gin_read_signal`
- `gin_wait_signal`

### Optional in v1

- `gin_barrier`

Only include `gin_barrier` in the first implementation if the first intended kernel
has a bulk-synchronous cross-rank phase boundary. If the kernel can be expressed with
point-to-point completion via signal increments and waits, GIN signals are enough and
the global barrier can be deferred.

## Bridge API

The bridge exposes plain `extern "C" __device__` functions with simple, stable ABIs.
All NCCL handles appear as raw `uint64_t` values in the ABI and are cast internally.

### Handle conventions

- `dev_comm_u64`: stores `ncclDevComm_t*`
- `window_u64`: stores `ncclWindow_t`
- `gin_barrier_u64`: stores `ncclGinBarrierHandle_t`

### Bridge source sketch

```c
#include <stdint.h>
#include <nccl_device.h>

static inline ncclDevComm* as_dev_comm(uint64_t x) {
  return reinterpret_cast<ncclDevComm*>(x);
}

static inline ncclWindow_t as_window(uint64_t x) {
  return reinterpret_cast<ncclWindow_t>(x);
}

static inline ncclGinBarrierHandle_t as_gin_barrier(uint64_t x) {
  return reinterpret_cast<ncclGinBarrierHandle_t>(x);
}

extern "C" __device__ __noinline__
uint64_t cute_nccl_local_ptr(uint64_t w, uint64_t offset) {
  return reinterpret_cast<uint64_t>(ncclGetLocalPointer(as_window(w), offset));
}

extern "C" __device__ __noinline__
uint64_t cute_nccl_lsa_ptr(uint64_t w, uint64_t offset, int peer) {
  return reinterpret_cast<uint64_t>(ncclGetLsaPointer(as_window(w), offset, peer));
}

extern "C" __device__ __noinline__
void cute_nccl_lsa_barrier(uint64_t comm_u64, uint32_t idx) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclLsaBarrierSession<ncclCoopThread> bar(
      ncclCoopThread{}, *comm, ncclTeamTagLsa{}, idx);
  bar.sync(ncclCoopThread{}, cuda::memory_order_relaxed);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_put(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclGin gin(*comm, ctx);
  gin.put(ncclTeamWorld(*comm), peer,
          as_window(dst_w), dst_off,
          as_window(src_w), src_off, bytes);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_put_siginc(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes, uint32_t signal) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclGin gin(*comm, ctx);
  gin.put<ncclGin_SignalInc>(ncclTeamWorld(*comm), peer,
      as_window(dst_w), dst_off,
      as_window(src_w), src_off, bytes,
      ncclGin_SignalInc{signal});
}

extern "C" __device__ __noinline__
void cute_nccl_gin_put_sigadd(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes,
    uint32_t signal, uint64_t value) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclGin gin(*comm, ctx);
  gin.put<ncclGin_SignalAdd>(ncclTeamWorld(*comm), peer,
      as_window(dst_w), dst_off,
      as_window(src_w), src_off, bytes,
      ncclGin_SignalAdd{signal, value});
}

extern "C" __device__ __noinline__
uint64_t cute_nccl_gin_read_signal(uint64_t comm_u64, int ctx, uint32_t signal) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  return ncclGin(*comm, ctx).readSignal(signal);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_wait_signal(uint64_t comm_u64, int ctx, uint32_t signal, uint64_t threshold) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclGin(*comm, ctx).waitSignal(ncclCoopThread{}, signal, threshold);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_barrier(uint64_t comm_u64, int ctx, uint64_t handle_u64, uint32_t idx) {
  ncclDevComm* comm = as_dev_comm(comm_u64);
  ncclGinBarrierSession<ncclCoopThread> bar(
      ncclCoopThread{}, ncclGin(*comm, ctx), ncclTeamWorld(*comm),
      as_gin_barrier(handle_u64), idx);
  bar.sync(ncclCoopThread{}, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
}
```

## Python Device Wrappers

Use `cute.ffi` directly.

This is simpler than manually declaring extern functions in MLIR, and it matches how
CuTe already expects external function calls to be inserted into the module.

```python
from cutlass import cute
from cutlass import Int32, Int64, Uint32, Uint64

_lsa_ptr_ffi = cute.ffi(
    name="cute_nccl_lsa_ptr",
    params_types=[Uint64, Uint64, Int32],
    return_type=Uint64,
)

_local_ptr_ffi = cute.ffi(
    name="cute_nccl_local_ptr",
    params_types=[Uint64, Uint64],
    return_type=Uint64,
)

_lsa_barrier_ffi = cute.ffi(
    name="cute_nccl_lsa_barrier",
    params_types=[Uint64, Uint32],
)

_gin_put_ffi = cute.ffi(
    name="cute_nccl_gin_put",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64],
)

_gin_put_siginc_ffi = cute.ffi(
    name="cute_nccl_gin_put_siginc",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64, Uint32],
)

_gin_put_sigadd_ffi = cute.ffi(
    name="cute_nccl_gin_put_sigadd",
    params_types=[Uint64, Int32, Int32, Uint64, Uint64, Uint64, Uint64, Uint64, Uint32, Uint64],
)

_gin_read_signal_ffi = cute.ffi(
    name="cute_nccl_gin_read_signal",
    params_types=[Uint64, Int32, Uint32],
    return_type=Uint64,
)

_gin_wait_signal_ffi = cute.ffi(
    name="cute_nccl_gin_wait_signal",
    params_types=[Uint64, Int32, Uint32, Uint64],
)

_gin_barrier_ffi = cute.ffi(
    name="cute_nccl_gin_barrier",
    params_types=[Uint64, Int32, Uint64, Uint32],
)
```

Then wrap these in small user-facing helpers if needed, but do not build a second
custom declaration mechanism.

## Path Discovery

Do not use `nvidia.nccl.__file__`. The wheel is installed as a namespace package and
`__file__` is `None`.

Use the package path directly:

```python
from pathlib import Path
import nvidia.nccl


def _nccl_root() -> Path:
    return Path(next(iter(nvidia.nccl.__path__)))


def get_nccl_include() -> str:
    return str(_nccl_root() / "include")


def get_nccl_so() -> str:
    root = _nccl_root()
    for name in ["libnccl.so", "libnccl.so.2"]:
        path = root / "lib" / name
        if path.exists():
            return str(path)
    raise FileNotFoundError("Could not find libnccl shared library")
```

## NVCC Bridge Build

Compile bridge PTX lazily and cache by:

- bridge source text
- target SM
- CUDA toolkit path / NVCC path

Use the actual CuTe compile target arch when possible. If needed, expose an override.

```python
@functools.cache
def get_bridge_ptx(sm: str) -> str:
    key = hashlib.md5(f"{sm}\0{_nvcc_path()}\0{BRIDGE_SRC}".encode()).hexdigest()[:16]
    ptx = cache_dir() / f"nccl_bridge_{key}.ptx"
    if ptx.exists():
        return str(ptx)

    src = ptx.with_suffix(".cu")
    src.parent.mkdir(parents=True, exist_ok=True)
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
```

## Compile Option

The user-facing compile hook should return a `cute.LinkLibraries` option.

```python
from cutlass import cute


def link_options(sm: str | None = None):
    sm = sm or detect_arch()
    return cute.LinkLibraries(get_bridge_ptx(sm))
```

## Host Setup

### Communicator capability query

Before enabling device-initiated features, query communicator properties and fail
early if the communicator does not support the required feature set.

Checks to consider:

- `deviceApiSupport`
- `ginType` if GIN is requested
- `multimemSupport` if LSA multimem is requested

### `ncclDevCommRequirements`

Construct the full requirements struct from the real NCCL layout, not a partial
Python-only subset. Important fields include:

- `lsaBarrierCount`
- `railGinBarrierCount`
- `ginContextCount`
- `ginSignalCount`
- `ginCounterCount`
- `ginConnectionType`
- `ginForceEnable`

The host layer should expose a higher-level Python API, but internally it must fill
the real NCCL ABI struct exactly.

**Critical**: The struct has `size`, `magic`, and `version` fields that must be
initialized using the C macro `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER` defaults
(`core.h:100-119`). In ctypes, this means:
- `size` = `ctypes.sizeof(NcclDevCommRequirements)`
- `magic` = `NCCL_API_MAGIC` (from `nccl.h`)
- `version` = `NCCL_VERSION(major, minor, patch)` (from installed NCCL)

The ctypes struct field order must be byte-identical to `core.h:72-98`. This is
the trickiest part of `_host.py` — one misaligned field silently corrupts all
subsequent fields. Verify by comparing `ctypes.sizeof()` against the C
`sizeof(ncclDevCommRequirements)` (print from a test bridge function if needed).

### Windows

Register user buffers with `ncclCommWindowRegister`.

```python
def register_window(comm, buf, size):
    win = ctypes.c_void_p()
    _check(
        lib.ncclCommWindowRegister(
            comm,
            buf,
            size,
            ctypes.byref(win),
            NCCL_WIN_COLL_SYMMETRIC,
        )
    )
    return win
```

### Barrier handles

For LSA barrier, the simplest v1 path is to rely on the comm-backed barrier stored in
`dev_comm->lsaBarrier`, so no extra Python-visible handle is needed.

For GIN barrier:

- if rail-team semantics are enough, prefer the comm-backed `dev_comm->railGinBarrier`
  path and add a dedicated bridge wrapper later.
- if world-team barrier is required, add host-side requirement creation and pass the
  resulting handle to the kernel as an opaque `uint64`.

## Target User API

```python
from cutlass import cute
import cute_comm.nccl as nccl

comm = nccl.init_rank(nranks, rank)
dev_comm = nccl.create_dev_comm(
    comm,
    gin_contexts=1,
    gin_signals=4,
    lsa_barriers=1,
)
src_win = nccl.register_window(comm, src_buf, nbytes)
dst_win = nccl.register_window(comm, dst_buf, nbytes)

@cute.kernel(options=nccl.link_options())
def ring_kernel(dev_comm, src_win, dst_win, nbytes):
    remote = nccl.lsa_ptr(dst_win, 0, 0)
    nccl.lsa_barrier(dev_comm, 0)

    nccl.gin_put_siginc(
        dev_comm, 0, 0,
        dst_win, 0,
        src_win, 0,
        nbytes, 0,
    )
    nccl.gin_wait_signal(dev_comm, 0, 0, 1)
```

## Implementation Order

1. `_lib.py`
   Correct NCCL include/library discovery from the installed wheel.
2. `_bridge_src.py`
   Add only v1 bridge functions.
3. `_bridge.py`
   Add NVCC compile + cache.
4. `_device_ops.py`
   Implement `cute.ffi` wrappers and very thin Python helpers.
5. `_host.py`
   Add communicator creation, property query, dev comm creation, and window registration.
6. Minimal kernel test
   Confirm PTX linking works and at least one bridge function can be called inside a CuTe kernel.
7. Barrier test
   Add a dedicated correctness test for `lsa_barrier`.
8. Optional `gin_barrier`
   Add only if required by the first real algorithm.

## Open Questions

1. Does the first intended kernel require a true GIN barrier, or only signal-based
   completion ordering?
2. Should device-visible rank metadata be bridged, or should `rank` / `nranks` just be
   passed as normal scalar kernel arguments in v1?
3. Do we want rail-team GIN barrier semantics first, or full world-team barrier semantics?

## Summary

The implementation is a thin NCCL device bridge with CuTe-linked PTX:

- use `cutlass.cute`
- use `cute.ffi`
- pass opaque 64-bit handles
- keep LSA barrier in v1
- keep GIN signals in v1
- add GIN barrier only if the first real kernel needs a true global phase barrier
