"""Bridge source: extern "C" __device__ wrappers around NCCL device API."""

BRIDGE_SRC = r"""
#include <nccl_device.h>

// ── Pointer access ──

extern "C" __device__ __noinline__
uint64_t cute_nccl_local_ptr(uint64_t w, uint64_t offset) {
    return reinterpret_cast<uint64_t>(
        ncclGetLocalPointer(reinterpret_cast<ncclWindow_t>(w), offset));
}

extern "C" __device__ __noinline__
uint64_t cute_nccl_lsa_ptr(uint64_t w, uint64_t offset, int peer) {
    return reinterpret_cast<uint64_t>(
        ncclGetLsaPointer(reinterpret_cast<ncclWindow_t>(w), offset, peer));
}

// ── LSA barrier ──

extern "C" __device__ __noinline__
void cute_nccl_lsa_barrier(uint64_t comm_u64, uint32_t idx) {
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    ncclLsaBarrierSession<ncclCoopThread> bar(
        ncclCoopThread{}, *comm, ncclTeamTagLsa{}, idx);
    bar.sync(ncclCoopThread{}, cuda::memory_order_relaxed);
}

// ── GIN put ──

extern "C" __device__ __noinline__
void cute_nccl_gin_put(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes)
{
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    ncclGin gin(*comm, ctx);
    gin.put(ncclTeamWorld(*comm), peer,
            reinterpret_cast<ncclWindow_t>(dst_w), dst_off,
            reinterpret_cast<ncclWindow_t>(src_w), src_off, bytes);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_put_siginc(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes, uint32_t signal)
{
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    ncclGin gin(*comm, ctx);
    gin.put<ncclGin_SignalInc>(ncclTeamWorld(*comm), peer,
        reinterpret_cast<ncclWindow_t>(dst_w), dst_off,
        reinterpret_cast<ncclWindow_t>(src_w), src_off, bytes,
        ncclGin_SignalInc{signal});
}

extern "C" __device__ __noinline__
void cute_nccl_gin_put_sigadd(uint64_t comm_u64, int ctx, int peer,
    uint64_t dst_w, uint64_t dst_off,
    uint64_t src_w, uint64_t src_off, uint64_t bytes,
    uint32_t signal, uint64_t value)
{
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    ncclGin gin(*comm, ctx);
    gin.put<ncclGin_SignalAdd>(ncclTeamWorld(*comm), peer,
        reinterpret_cast<ncclWindow_t>(dst_w), dst_off,
        reinterpret_cast<ncclWindow_t>(src_w), src_off, bytes,
        ncclGin_SignalAdd{signal, value});
}

// ── GIN signals ──

extern "C" __device__ __noinline__
uint64_t cute_nccl_gin_read_signal(uint64_t comm_u64, int ctx, uint32_t signal) {
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    return ncclGin(*comm, ctx).readSignal(signal);
}

extern "C" __device__ __noinline__
void cute_nccl_gin_wait_signal(uint64_t comm_u64, int ctx,
    uint32_t signal, uint64_t threshold)
{
    ncclDevComm* comm = reinterpret_cast<ncclDevComm*>(comm_u64);
    ncclGin(*comm, ctx).waitSignal(ncclCoopThread{}, signal, threshold);
}

"""
