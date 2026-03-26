# CuTe-comm

NCCL device-initiated API bindings for the [CuTe Python DSL](https://github.com/NVIDIA/cutlass).

Exposes NCCL's GIN (GPU-initiated networking) and LSA (local symmetric access) APIs via `cute.ffi`, with a thin host-side layer for communicator setup.

## Requirements

- Python >= 3.10, CUDA 13, NCCL >= 2.29.7
- GPU with `deviceApiSupport` (check via `query_properties`)

## Installation

```bash
pip install cute-comm
# or from source
uv sync
```

## Usage

```python
import cute_comm.nccl as cn

# Host: bootstrap communicator, register memory
uid = cn.get_unique_id()          # broadcast from rank 0
comm = cn.init_rank(nranks, rank, uid)
dev_comm = cn.create_dev_comm(comm, gin_contexts=1, gin_signals=1, lsa_barriers=1)
win = cn.register_window(comm, buf.data_ptr(), buf.nbytes)

# Device: use inside @cute.jit kernels
@cute.jit
def my_kernel(dev_comm, ctx, win, peer):
    ptr = cn.lsa_ptr(win, offset, peer)
    cn.gin_put_siginc(dev_comm, ctx, peer, dst_win, dst_off, src_win, src_off, nbytes, signal_idx)
    cn.gin_wait_signal(dev_comm, ctx, signal_idx, threshold=1)

# Pass link_options to bundle the bridge PTX (compiled lazily, cached in ~/.cache/cute_comm/)
my_kernel[grid, block, cn.link_options("sm_90a")](dev_comm, ctx, win, peer)

# Cleanup
cn.deregister_window(comm, win)
cn.destroy_dev_comm(comm, dev_comm)
cn.finalize(comm)
cn.destroy(comm)
```

## API

**Host** — `get_unique_id`, `init_rank`, `query_properties`, `create_dev_comm`, `destroy_dev_comm`, `register_window`, `deregister_window`, `finalize`, `destroy`

**Device** (inside `@cute.jit`) — `local_ptr`, `lsa_ptr`, `lsa_barrier`, `gin_put`, `gin_put_siginc`, `gin_put_sigadd`, `gin_read_signal`, `gin_wait_signal`

## License

BSD 3-Clause. See [LICENSE](LICENSE).
