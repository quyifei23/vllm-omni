#!/usr/bin/env python3
"""Standalone GPU tensor transport microbenchmark.

Producer (GPU 0) creates tensors, sends via CUDA IPC or CUDA copy.
Consumer (GPU 1) receives, optionally sleeps to simulate queue delay,
performs **full-tensor access** (tensor.sum() on all elements) to measure
true remote-memory cost for cuda_ipc, then signals release.

Key metric:
  recv_latency_ms   — time to get a usable tensor handle (IPC open or P2P copy)
  access_latency_ms — time to compute .sum() on the **entire** tensor
  cuda_ipc:  recv is fast (handle only), access is slow (remote VRAM bandwidth)
  cuda_copy: recv is slow (full P2P copy), access is fast (local memory)

Usage:
  python benchmarks/gpu_transport/run_benchmark.py \\
    --mode cuda_ipc --tensor-size-mb 4,16,64,256 \\
    --queue-delay-ms 0,1,5,10,20,50,100 \\
    --warmup 10 --iterations 50 --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import multiprocessing as mp
import os
import statistics
import sys
import time
import types

import torch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_VLLM_OMNI_BASE = os.path.join(_PROJECT_ROOT, "vllm_omni")
_GPU_TRANSPORT_BASE = os.path.join(
    _VLLM_OMNI_BASE, "distributed", "gpu_transport"
)
_GPU_TRANSPORT_PKG = "vllm_omni.distributed.gpu_transport"

sys.path.insert(0, _PROJECT_ROOT)


def _setup_gpu_transport_imports():
    """Load ``gpu_transport`` submodules without triggering
    ``vllm_omni/__init__.py`` (which imports vllm and may fail due to
    dependency mismatches).

    Idempotent -- safe to call in parent and child processes.
    """
    # 1. Stub parent packages
    for name, path in [
        ("vllm_omni", _VLLM_OMNI_BASE),
        ("vllm_omni.distributed",
         os.path.join(_VLLM_OMNI_BASE, "distributed")),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    # 2. Stub gpu_transport package
    if _GPU_TRANSPORT_PKG not in sys.modules:
        pkg_mod = types.ModuleType(_GPU_TRANSPORT_PKG)
        pkg_mod.__path__ = [_GPU_TRANSPORT_BASE]
        pkg_mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[_GPU_TRANSPORT_PKG] = pkg_mod

    # 3. Load submodules in dependency order
    def _load(name, file):
        full = f"{_GPU_TRANSPORT_PKG}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        path = os.path.join(_GPU_TRANSPORT_BASE, file)
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    # Layer 1: no intra-package deps
    _load("logging", "logging.py")
    _load("protocol", "protocol.py")
    _load("config", "config.py")

    # Layer 2: depend on logging
    _load("tensor_registry", "tensor_registry.py")
    _load("ipc_utils", "ipc_utils.py")

    # Layer 3: depend on everything above
    _load("cuda_ipc_transport", "cuda_ipc_transport.py")
    _load("cuda_copy_transport", "cuda_copy_transport.py")


# ---------------------------------------------------------------------------
# Worker functions  (run in spawned child processes)
# ---------------------------------------------------------------------------


def producer_worker(conn, nbytes, mode, src_dev, dst_dev,
                    total_rounds, dtype, result_queue):
    """Run on GPU src_dev: create tensor, send, wait for ACK, release.

    Args:
        conn: Pipe connection to consumer (send metadata, recv ACK).
        nbytes: Size of each tensor in bytes.
        mode: Transport mode ('cuda_ipc' or 'cuda_copy').
        src_dev: Source GPU index.
        dst_dev: Destination GPU index.
        total_rounds: Number of send/recv cycles to perform.
        dtype: torch dtype for the tensor.
        result_queue: mp.Queue for sending timing results to main process.
    """
    _setup_gpu_transport_imports()

    if mode == "cuda_ipc":
        from vllm_omni.distributed.gpu_transport.config import (
            GPUTransportConfig,
        )
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )
        transport = CudaIpcTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev,
                               dst_device=dst_dev))
    else:
        from vllm_omni.distributed.gpu_transport.config import (
            GPUTransportConfig,
        )
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )
        transport = CudaCopyTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev,
                               dst_device=dst_dev))

    try:
        for i in range(total_rounds):
            n_elems = max(1, nbytes // dtype.itemsize)
            tensor = torch.randn(n_elems, dtype=dtype,
                                 device=f"cuda:{src_dev}")
            torch.cuda.synchronize(tensor.device)

            t0 = time.perf_counter()
            handle = transport.send(tensor, dst_rank=dst_dev,
                                    tensor_id=f"bm-{i}")
            t1 = time.perf_counter()
            send_ms = (t1 - t0) * 1000

            conn.send({
                "iter": i,
                "metadata": handle.metadata.to_dict(),
                "ipc_args": handle.metadata.ipc_args,
            })

            # Wait for consumer ACK with timing
            ack = conn.recv()
            t2 = time.perf_counter()
            total_hold_ms = (t2 - t0) * 1000

            transport.release(f"bm-{i}")

            result_queue.put({
                "send_ms": send_ms,
                "recv_ms": ack.get("recv_ms", 0.0),
                "consumer_access_ms": ack.get("access_ms", 0.0),
                "total_hold_ms": total_hold_ms,
            })
            del tensor
    finally:
        transport.close()


def consumer_worker(conn, mode, src_dev, dst_dev,
                    queue_delay_ms, total_rounds):
    """Run on GPU dst_dev: receive tensor, optionally delay, access, ACK.

    Args:
        conn: Pipe connection to producer (recv metadata, send ACK).
        mode: Transport mode ('cuda_ipc' or 'cuda_copy').
        src_dev: Source GPU index.
        dst_dev: Destination GPU index.
        queue_delay_ms: Simulated queue delay in milliseconds.
        total_rounds: Number of recv/ACK cycles to perform.
    """
    _setup_gpu_transport_imports()

    from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
    from vllm_omni.distributed.gpu_transport.protocol import (
        TensorMetadata, TransportHandle,
    )

    if mode == "cuda_ipc":
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )
        transport = CudaIpcTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev,
                               dst_device=dst_dev))
    else:
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )
        transport = CudaCopyTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev,
                               dst_device=dst_dev))

    try:
        for _ in range(total_rounds):
            msg = conn.recv()
            meta_dict = msg["metadata"]
            ipc_args = msg["ipc_args"]
            metadata = TensorMetadata.from_dict(meta_dict)
            metadata.ipc_args = ipc_args
            handle = TransportHandle(tensor_id=metadata.tensor_id,
                                     metadata=metadata)

            t0 = time.perf_counter()
            tensor = transport.recv(handle, src_rank=src_dev,
                                    dst_device=f"cuda:{dst_dev}")
            t1 = time.perf_counter()
            recv_ms = (t1 - t0) * 1000

            if queue_delay_ms > 0:
                time.sleep(queue_delay_ms / 1000.0)

            t2 = time.perf_counter()
            # Full-tensor access: touches every element to measure true cost.
            # For cuda_ipc this exercises remote GPU memory bandwidth across
            # PCIe/NVLink.  For cuda_copy this is fast local access.
            _ = tensor.sum().item()
            torch.cuda.synchronize(tensor.device)
            t3 = time.perf_counter()
            access_ms = (t3 - t2) * 1000

            conn.send({
                "recv_ms": recv_ms,
                "access_ms": access_ms,
            })
            del tensor
    finally:
        transport.close()


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def run_benchmark_config(nbytes, mode, src, dst, delay_ms,
                         warmup, iterations, dtype):
    """Run one (size, delay) config with a single producer/consumer pair.

    Spawns a producer process on GPU src and a consumer process on GPU dst.
    Performs ``warmup + iterations`` rounds in the same processes to avoid
    per-iteration CUDA context creation overhead.

    Returns:
        List of dicts with keys ``send_ms``, ``recv_ms``,
        ``access_latency_ms``, ``total_hold_ms`` for the
        non-warmup iterations.  May be shorter than *iterations*
        if the processes crashed or timed out.
    """
    total = warmup + iterations
    parent_conn, child_conn = mp.Pipe()
    result_queue = mp.Queue()

    prod = mp.Process(target=producer_worker,
                      args=(child_conn, nbytes, mode, src, dst,
                            total, dtype, result_queue))
    cons = mp.Process(target=consumer_worker,
                      args=(parent_conn, mode, src, dst,
                            delay_ms, total))

    prod.start()
    cons.start()

    prod.join(timeout=300)
    cons.join(timeout=300)

    for p in [prod, cons]:
        if p.is_alive():
            p.terminate()

    parent_conn.close()
    child_conn.close()

    # Collect all results from the queue
    all_results: list[dict] = []
    for _ in range(total):
        try:
            all_results.append(result_queue.get(timeout=5))
        except Exception:
            break

    # Return only measured (non-warmup) results
    if len(all_results) > warmup:
        return all_results[warmup:]
    return []


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    "mode", "tensor_size_mb", "queue_delay_ms", "iteration",
    "metadata_latency_ms", "recv_latency_ms", "access_latency_ms",
    "end_to_end_ms", "total_hold_ms", "success", "error",
]


def _aggregate_stats(rows, metric):
    """Compute aggregate statistics for a metric across all rows."""
    vals = [float(r[metric]) for r in rows if r.get(metric)]
    if not vals:
        return None
    svals = sorted(vals)
    n = len(svals)
    return {
        "count": n,
        "mean": statistics.mean(vals),
        "p50": svals[int(n * 0.50)],
        "p95": svals[int(n * 0.95)],
        "p99": svals[min(int(n * 0.99), n - 1)],
        "min": svals[0],
        "max": svals[-1],
    }


def print_aggregate(path: str) -> None:
    """Print aggregate statistics from CSV."""
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("success") == "1"]
    if not rows:
        print("No successful data rows")
        return

    print("\n--- Aggregate Summary ---")
    for metric in ["end_to_end_ms", "metadata_latency_ms", "recv_latency_ms",
                   "access_latency_ms", "total_hold_ms"]:
        stats = _aggregate_stats(rows, metric)
        if stats is None:
            continue
        print(f"  {metric}:")
        print(f"    count={stats['count']}  "
              f"mean={stats['mean']:.4f}  "
              f"p50={stats['p50']:.4f}  "
              f"p95={stats['p95']:.4f}  "
              f"p99={stats['p99']:.4f}  "
              f"min={stats['min']:.4f}  "
              f"max={stats['max']:.4f}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GPU Tensor Transport Microbenchmark")
    parser.add_argument("--mode", choices=["cuda_ipc", "cuda_copy"],
                        default="cuda_ipc")
    parser.add_argument("--src-device", type=int, default=0)
    parser.add_argument("--dst-device", type=int, default=1)
    parser.add_argument("--tensor-size-mb", type=str, default="4,16,64,256")
    parser.add_argument("--queue-delay-ms", type=str,
                        default="0,1,5,10,20,50,100")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    sizes_mb = [int(x) for x in args.tensor_size_mb.split(",")]
    delays_ms = [float(x) for x in args.queue_delay_ms.split(",")]
    dt = torch.float16 if args.dtype == "float16" else torch.float32

    mp.set_start_method("spawn", force=True)

    print(f"# GPU Tensor Transport Benchmark: {args.mode}")
    print(f"# Sizes (MB): {sizes_mb}")
    print(f"# Delays (ms): {delays_ms}")
    print(f"# Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"# Output: {args.output}")
    sys.stdout.flush()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()

        for size_mb in sizes_mb:
            nbytes = size_mb * 1024 * 1024
            for delay_ms in delays_ms:
                label = f"size={size_mb:3d}MB delay={delay_ms:4.0f}ms"
                print(f"\n  {label} ", end="", flush=True)

                metrics_list = []
                try:
                    metrics_list = run_benchmark_config(
                        nbytes, args.mode, args.src_device,
                        args.dst_device, delay_ms,
                        args.warmup, args.iterations, dt)

                    for it, metrics in enumerate(metrics_list):
                        e2e = metrics["send_ms"] + metrics["recv_ms"]
                        writer.writerow({
                            "mode": args.mode,
                            "tensor_size_mb": size_mb,
                            "queue_delay_ms": delay_ms,
                            "iteration": it,
                            "metadata_latency_ms":
                                f"{metrics['send_ms']:.3f}",
                            "recv_latency_ms":
                                f"{metrics['recv_ms']:.3f}",
                            "access_latency_ms":
                                f"{metrics['consumer_access_ms']:.3f}",
                            "end_to_end_ms": f"{e2e:.3f}",
                            "total_hold_ms":
                                f"{metrics['total_hold_ms']:.3f}",
                            "success": "1",
                            "error": "",
                        })

                    # Fill missing iterations as failures
                    for it in range(len(metrics_list), args.iterations):
                        writer.writerow({
                            "mode": args.mode,
                            "tensor_size_mb": size_mb,
                            "queue_delay_ms": delay_ms,
                            "iteration": it,
                            "metadata_latency_ms": "",
                            "recv_latency_ms": "",
                            "access_latency_ms": "",
                            "end_to_end_ms": "",
                            "total_hold_ms": "",
                            "success": "0",
                            "error": "no result from benchmark round",
                        })

                except Exception as exc:
                    for it in range(args.iterations):
                        writer.writerow({
                            "mode": args.mode,
                            "tensor_size_mb": size_mb,
                            "queue_delay_ms": delay_ms,
                            "iteration": it,
                            "metadata_latency_ms": "",
                            "recv_latency_ms": "",
                            "access_latency_ms": "",
                            "end_to_end_ms": "",
                            "total_hold_ms": "",
                            "success": "0",
                            "error": str(exc),
                        })

                f.flush()
                print(f"OK ({len(metrics_list)}/{args.iterations})",
                      end="", flush=True)

    print(f"\n\nResults written to {args.output}\n")
    print_aggregate(args.output)


if __name__ == "__main__":
    main()
