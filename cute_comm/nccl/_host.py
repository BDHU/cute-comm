"""Host-side NCCL setup: communicator, dev comm, windows."""

from __future__ import annotations

import ctypes

from ._lib import load_nccl

# ── Constants ──

NCCL_UNIQUE_ID_BYTES = 128
NCCL_API_MAGIC = 0xCAFEBEEF
NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH = 2, 29, 7
NCCL_VERSION_CODE = NCCL_MAJOR * 10000 + NCCL_MINOR * 100 + NCCL_PATCH

NCCL_WIN_COLL_SYMMETRIC = 0x01

NCCL_GIN_CONNECTION_NONE = 0
NCCL_GIN_CONNECTION_FULL = 1
NCCL_GIN_CONNECTION_RAIL = 2


# ── ctypes structs ──

class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class NcclDevCommRequirements(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("resourceRequirementsList", ctypes.c_void_p),
        ("teamRequirementsList", ctypes.c_void_p),
        ("lsaMultimem", ctypes.c_bool),
        ("barrierCount", ctypes.c_int),
        ("lsaBarrierCount", ctypes.c_int),
        ("railGinBarrierCount", ctypes.c_int),
        ("lsaLLA2ABlockCount", ctypes.c_int),
        ("lsaLLA2ASlotCount", ctypes.c_int),
        ("ginForceEnable", ctypes.c_bool),
        ("ginContextCount", ctypes.c_int),
        ("ginSignalCount", ctypes.c_int),
        ("ginCounterCount", ctypes.c_int),
        ("ginConnectionType", ctypes.c_int),
        ("ginExclusiveContexts", ctypes.c_bool),
        ("ginQueueDepth", ctypes.c_int),
    ]


class NcclCommProperties(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("rank", ctypes.c_int),
        ("nRanks", ctypes.c_int),
        ("cudaDev", ctypes.c_int),
        ("nvmlDev", ctypes.c_int),
        ("deviceApiSupport", ctypes.c_bool),
        ("multimemSupport", ctypes.c_bool),
        ("ginType", ctypes.c_int),
        ("nLsaTeams", ctypes.c_int),
        ("hostRmaSupport", ctypes.c_bool),
        ("railedGinType", ctypes.c_int),
    ]


# ── Helpers ──

def _check(result: int) -> None:
    if result != 0:
        raise RuntimeError(f"NCCL error: {result}")


def _default_requirements(**overrides) -> NcclDevCommRequirements:
    req = NcclDevCommRequirements()
    req.size = ctypes.sizeof(NcclDevCommRequirements)
    req.magic = NCCL_API_MAGIC
    req.version = NCCL_VERSION_CODE
    req.resourceRequirementsList = None
    req.teamRequirementsList = None
    req.lsaMultimem = False
    req.barrierCount = 0
    req.lsaBarrierCount = 0
    req.railGinBarrierCount = 0
    req.lsaLLA2ABlockCount = 0
    req.lsaLLA2ASlotCount = 0
    req.ginForceEnable = False
    req.ginContextCount = 4
    req.ginSignalCount = 0
    req.ginCounterCount = 0
    req.ginConnectionType = NCCL_GIN_CONNECTION_NONE
    req.ginExclusiveContexts = False
    req.ginQueueDepth = 0
    for k, v in overrides.items():
        setattr(req, k, v)
    return req


# ── Public API ──

def get_unique_id() -> NcclUniqueId:
    lib = load_nccl()
    uid = NcclUniqueId()
    _check(lib.ncclGetUniqueId(ctypes.byref(uid)))
    return uid


def init_rank(nranks: int, rank: int, unique_id: NcclUniqueId) -> ctypes.c_void_p:
    """Initialize a communicator for *rank* out of *nranks*.

    *unique_id* must be a single ID obtained via :func:`get_unique_id` on one
    rank and broadcast to all others.  Every rank must pass the **same** ID.
    """
    lib = load_nccl()
    comm = ctypes.c_void_p()
    _check(lib.ncclCommInitRank(ctypes.byref(comm), nranks, unique_id, rank))
    return comm


def query_properties(comm: ctypes.c_void_p) -> NcclCommProperties:
    lib = load_nccl()
    props = NcclCommProperties()
    props.size = ctypes.sizeof(NcclCommProperties)
    props.magic = NCCL_API_MAGIC
    props.version = NCCL_VERSION_CODE
    _check(lib.ncclCommQueryProperties(comm, ctypes.byref(props)))
    return props


def create_dev_comm(
    comm: ctypes.c_void_p,
    *,
    gin_contexts: int = 0,
    gin_signals: int = 0,
    gin_counters: int = 0,
    gin_connection: int = NCCL_GIN_CONNECTION_NONE,
    lsa_barriers: int = 0,
) -> ctypes.c_void_p:
    lib = load_nccl()
    req = _default_requirements(
        ginContextCount=gin_contexts,
        ginSignalCount=gin_signals,
        ginCounterCount=gin_counters,
        ginConnectionType=gin_connection,
        lsaBarrierCount=lsa_barriers,
        ginForceEnable=gin_contexts > 0,
    )
    dev_comm = ctypes.c_void_p()
    _check(lib.ncclDevCommCreate(comm, ctypes.byref(req), ctypes.byref(dev_comm)))
    return dev_comm


def destroy_dev_comm(comm: ctypes.c_void_p, dev_comm: ctypes.c_void_p) -> None:
    lib = load_nccl()
    _check(lib.ncclDevCommDestroy(comm, dev_comm))


def register_window(
    comm: ctypes.c_void_p,
    buf: int,
    size: int,
    flags: int = NCCL_WIN_COLL_SYMMETRIC,
) -> ctypes.c_void_p:
    lib = load_nccl()
    win = ctypes.c_void_p()
    _check(lib.ncclCommWindowRegister(comm, ctypes.c_void_p(buf), size, ctypes.byref(win), flags))
    return win


def deregister_window(comm: ctypes.c_void_p, win: ctypes.c_void_p) -> None:
    lib = load_nccl()
    _check(lib.ncclCommWindowDeregister(comm, win))


def finalize(comm: ctypes.c_void_p) -> None:
    lib = load_nccl()
    _check(lib.ncclCommFinalize(comm))


def destroy(comm: ctypes.c_void_p) -> None:
    lib = load_nccl()
    _check(lib.ncclCommDestroy(comm))
