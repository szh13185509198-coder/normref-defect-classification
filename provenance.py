"""Reproducibility manifests for training and formal evaluation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import numbers
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch


MANIFEST_SCHEMA_VERSION = 1
SOURCE_SUFFIXES = {
    ".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}
IGNORED_DIRECTORY_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: os.PathLike | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_source_sha256(path: os.PathLike | str) -> Optional[dict]:
    """Hash source/config files in a directory independently of its location."""
    root = Path(path)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if any(part in IGNORED_DIRECTORY_PARTS for part in relative.parts):
            continue
        if candidate.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative_text = relative.as_posix()
        content_hash = file_sha256(candidate)
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        files.append(relative_text)
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "included_suffixes": sorted(SOURCE_SUFFIXES),
    }


def _run_git(args: Sequence[str], cwd: os.PathLike | str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
    )


def collect_git_provenance(repo_path: os.PathLike | str = ".") -> dict:
    repo = Path(repo_path).resolve()
    try:
        commit = _run_git(["rev-parse", "HEAD"], repo).decode().strip()
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo).decode().strip()
        status = _run_git(["status", "--porcelain=v1", "--untracked-files=all"], repo)
        diff = _run_git(["diff", "--binary", "HEAD", "--"], repo)
        untracked_output = _run_git(
            ["ls-files", "--others", "--exclude-standard", "-z"], repo
        )
        untracked = [item for item in untracked_output.split(b"\0") if item]
        digest = hashlib.sha256()
        digest.update(diff)
        for raw_relative in sorted(untracked):
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            candidate = repo / relative
            digest.update(raw_relative)
            digest.update(b"\0")
            if candidate.is_file():
                digest.update(file_sha256(candidate).encode("ascii"))
            digest.update(b"\n")
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
            "diff_sha256": digest.hexdigest(),
            "untracked_file_count": len(untracked),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_porcelain_sha256": None,
            "diff_sha256": None,
            "untracked_file_count": None,
        }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_environment() -> dict:
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    faiss_version = _package_version("faiss-gpu") or _package_version("faiss-cpu")
    if faiss_version is None:
        try:
            import faiss
            faiss_version = getattr(faiss, "__version__", None)
        except ImportError:
            pass
    return {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "packages": {
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "transformers": _package_version("transformers"),
            "faiss": faiss_version,
            "numpy": _package_version("numpy"),
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpus": gpus,
        },
    }


def _hash_named_tensors(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> dict:
    digest = hashlib.sha256()
    tensor_count = 0
    parameter_count = 0
    names = []
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # reshape(-1) first: dtype-reinterpretation ``view(torch.uint8)`` rejects
        # 0-D scalar tensors (``self.dim() cannot be 0 to view Float as Byte``),
        # which occur in real state_dicts (e.g. CLIP logit_scale). Flattening
        # only obtains the raw byte sequence; dtype/shape are hashed separately
        # above, so no shape metadata is lost.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        parameter_count += value.numel()
        names.append(name)
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "parameter_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
    }


def parameter_fingerprint(
    modules: Sequence[Tuple[str, torch.nn.Module]], *, trainable_only: bool
) -> dict:
    tensors = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            tensors.append((f"{prefix}.{name}", parameter))
    result = _hash_named_tensors(tensors)
    result["trainable_only"] = trainable_only
    return result


def model_state_fingerprint(module: torch.nn.Module) -> dict:
    return _hash_named_tensors(module.state_dict().items())


def state_dicts_fingerprint(
    mappings: Sequence[Tuple[str, Mapping[str, torch.Tensor]]]
) -> dict:
    tensors = []
    for prefix, mapping in mappings:
        for name, tensor in mapping.items():
            if torch.is_tensor(tensor):
                tensors.append((f"{prefix}.{name}", tensor))
    return _hash_named_tensors(tensors)


def file_identity(path: Optional[str], *, include_sha256: bool = True) -> Optional[dict]:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    result = {
        "path": str(resolved),
        "filename": resolved.name,
        "exists": True,
        "size_bytes": resolved.stat().st_size,
    }
    if include_sha256:
        result["sha256"] = file_sha256(resolved)
    return result


def jsonl_identity(path: os.PathLike | str) -> dict:
    identity = file_identity(str(path)) or {}
    if identity.get("exists"):
        with open(path, "r", encoding="utf-8") as source:
            identity["records"] = sum(1 for line in source if line.strip())
    return identity


def collect_backbone_provenance(
    *,
    model,
    model_name: str,
    actual_dino_backend: str,
    dino_hub_repo: Optional[str],
    dino_hub_checkpoint: Optional[str],
    clip_weight_path: Optional[str],
) -> dict:
    dino_extractor = model.dino_extractor
    dino_cfg = getattr(dino_extractor, "cfg", {})
    dino = {
        "model_name": "dinov2_vits14",
        "hub_name": dino_cfg.get("hub_name"),
        "huggingface_id": dino_cfg.get("hf_name"),
        "actual_backend": actual_dino_backend,
        "patch_policy": getattr(dino_extractor, "patch_policy", None),
        "rng_protocol": getattr(model, "rng_protocol", None),
        "weight_file": file_identity(dino_hub_checkpoint),
        "loaded_state": model_state_fingerprint(dino_extractor.model),
        "local_source_path": (
            str(Path(dino_hub_repo).expanduser().resolve()) if dino_hub_repo else None
        ),
        "local_source": directory_source_sha256(dino_hub_repo) if dino_hub_repo else None,
    }
    clip = {
        "model_name": model_name,
        "pretrained_identifier": f"openai:{model_name}",
        "weight_file": file_identity(clip_weight_path),
    }
    return {"clip": clip, "dino": dino}


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, numbers.Number):
        return value.item() if hasattr(value, "item") else value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return repr(value)


def load_manifest(path: os.PathLike | str) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "evaluations": [],
            "artifacts": {"checkpoints": {}},
        }
    with open(manifest_path, "r", encoding="utf-8") as source:
        return json.load(source)


def write_manifest(path: os.PathLike | str, manifest: Mapping[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_json_compatible(manifest))
    payload["schema_version"] = MANIFEST_SCHEMA_VERSION
    payload["updated_utc"] = utc_now()
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, manifest_path)


def initialize_training_manifest(path: os.PathLike | str, training: Mapping[str, Any]) -> None:
    manifest = load_manifest(path)
    manifest["project"] = collect_git_provenance(Path.cwd())
    manifest["training"] = _json_compatible(training)
    manifest.setdefault("artifacts", {}).setdefault("checkpoints", {})
    manifest.setdefault("evaluations", [])
    write_manifest(path, manifest)


def update_training_manifest(path: os.PathLike | str, updates: Mapping[str, Any]) -> None:
    manifest = load_manifest(path)
    training = manifest.setdefault("training", {})
    training.update(_json_compatible(updates))
    write_manifest(path, manifest)


def record_artifact(
    path: os.PathLike | str,
    artifact_path: os.PathLike | str,
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    manifest = load_manifest(path)
    artifact = Path(artifact_path).resolve()
    identity = file_identity(str(artifact))
    record = {
        **(identity or {}),
        "kind": kind,
        "recorded_utc": utc_now(),
        **_json_compatible(metadata or {}),
    }
    artifacts = manifest.setdefault("artifacts", {})
    if kind == "checkpoint":
        artifacts.setdefault("checkpoints", {})[artifact.name] = record
    else:
        artifacts[artifact.name] = record
    write_manifest(path, manifest)
    return record


def append_evaluation(path: os.PathLike | str, evaluation: Mapping[str, Any]) -> None:
    manifest = load_manifest(path)
    manifest.setdefault("project", collect_git_provenance(Path.cwd()))
    manifest.setdefault("evaluations", []).append(_json_compatible(evaluation))
    write_manifest(path, manifest)
