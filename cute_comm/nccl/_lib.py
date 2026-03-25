"""NCCL path discovery and shared library loading."""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import nvidia.nccl


def _nccl_root() -> Path:
    return Path(next(iter(nvidia.nccl.__path__)))


def get_nccl_include() -> str:
    return str(_nccl_root() / "include")


def get_nccl_so() -> str:
    root = _nccl_root()
    for name in ("libnccl.so", "libnccl.so.2"):
        path = root / "lib" / name
        if path.exists():
            return str(path)
    raise FileNotFoundError("Could not find libnccl shared library")


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype for every NCCL function we call."""
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t
    c_void_p = ctypes.c_void_p
    POINTER = ctypes.POINTER

    # Avoid circular import — these structs live in _host
    from ._host import NcclUniqueId, NcclDevCommRequirements, NcclCommProperties

    # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)
    lib.ncclGetUniqueId.argtypes = [POINTER(NcclUniqueId)]
    lib.ncclGetUniqueId.restype = c_int

    # ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
    #                               ncclUniqueId commId, int rank)
    lib.ncclCommInitRank.argtypes = [POINTER(c_void_p), c_int, NcclUniqueId, c_int]
    lib.ncclCommInitRank.restype = c_int

    # ncclResult_t ncclCommQueryProperties(ncclComm_t comm,
    #                                      ncclCommProperties* props)
    lib.ncclCommQueryProperties.argtypes = [c_void_p, POINTER(NcclCommProperties)]
    lib.ncclCommQueryProperties.restype = c_int

    # ncclResult_t ncclDevCommCreate(ncclComm_t comm,
    #                                ncclDevCommRequirements* req,
    #                                ncclDevComm** devComm)
    lib.ncclDevCommCreate.argtypes = [c_void_p, POINTER(NcclDevCommRequirements), POINTER(c_void_p)]
    lib.ncclDevCommCreate.restype = c_int

    # ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm* devComm)
    lib.ncclDevCommDestroy.argtypes = [c_void_p, c_void_p]
    lib.ncclDevCommDestroy.restype = c_int

    # ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buf,
    #                                     size_t size, ncclWindow_t* win,
    #                                     int flags)
    lib.ncclCommWindowRegister.argtypes = [c_void_p, c_void_p, c_size_t, POINTER(c_void_p), c_int]
    lib.ncclCommWindowRegister.restype = c_int

    # ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win)
    lib.ncclCommWindowDeregister.argtypes = [c_void_p, c_void_p]
    lib.ncclCommWindowDeregister.restype = c_int

    # ncclResult_t ncclCommFinalize(ncclComm_t comm)
    lib.ncclCommFinalize.argtypes = [c_void_p]
    lib.ncclCommFinalize.restype = c_int

    # ncclResult_t ncclCommDestroy(ncclComm_t comm)
    lib.ncclCommDestroy.argtypes = [c_void_p]
    lib.ncclCommDestroy.restype = c_int


@functools.cache
def load_nccl() -> ctypes.CDLL:
    lib = ctypes.CDLL(get_nccl_so())
    _declare_signatures(lib)
    return lib
