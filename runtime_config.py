"""Cross-platform runtime helpers for local Windows and Linux GPU servers."""

import os


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting Linux CPU affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def prepare_process_environment() -> int:
    """Set conservative thread defaults before NumPy/PyTorch are imported."""
    available = available_cpu_count()
    explicit_threads = "AACLIP_CPU_THREADS" in os.environ
    requested = int(os.environ.get("AACLIP_CPU_THREADS", min(4, available)))
    cpu_threads = max(1, min(requested, available))

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        if explicit_threads:
            os.environ[name] = str(cpu_threads)
        else:
            os.environ.setdefault(name, str(cpu_threads))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return cpu_threads


RUNTIME_CPU_THREADS = prepare_process_environment()


def configure_torch_runtime(torch_module, enable_tf32: bool = False) -> dict:
    """Apply PyTorch runtime settings after the package has been imported."""
    torch_module.set_num_threads(RUNTIME_CPU_THREADS)
    use_cuda = torch_module.cuda.is_available()
    if use_cuda:
        torch_module.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        torch_module.backends.cudnn.allow_tf32 = bool(enable_tf32)
        if enable_tf32:
            torch_module.set_float32_matmul_precision("high")
    return {
        "cpu_threads": RUNTIME_CPU_THREADS,
        "available_cpus": available_cpu_count(),
        "tf32": bool(enable_tf32 and use_cuda),
    }


def dataloader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    use_cuda: bool,
) -> dict:
    """Build valid DataLoader options without worker-only arguments at 0."""
    workers = max(0, int(num_workers))
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory and use_cuda),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs
