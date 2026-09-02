#!/usr/bin/env python3
"""Publish inference-only Motion-DynVLA checkpoints to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


CHECKPOINTS = {
    "libero-source": "dynvla_gr00t_n16_libero_source_joint_final",
    "libero-target20": "dynvla_gr00t_n16_libero_target20_final",
}

REQUIRED_FILES = (
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "embodiment_id.json",
    "statistics.json",
)

EXCLUDED_TRAINING_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
)


def call_with_retry(label: str, operation, attempts: int = 8):
    """Retry idempotent Hugging Face API operations after transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(5 * (2 ** (attempt - 1)), 90)
            print(
                f"{label} failed on attempt {attempt}/{attempts} "
                f"with {type(exc).__name__}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def checkpoint_files(
    project_root: Path,
) -> tuple[dict[str, list[tuple[str, Path]]], dict]:
    runs_root = project_root / "runs" / "gr00t_n16"
    checkpoint_entries: dict[str, list[tuple[str, Path]]] = {}
    manifest: dict = {"checkpoints": {}, "excluded_training_files": list(EXCLUDED_TRAINING_FILES)}
    for folder, run_name in CHECKPOINTS.items():
        checkpoint = (runs_root / run_name).resolve(strict=True)
        files = []
        entries = []
        for filename in REQUIRED_FILES:
            path = checkpoint / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
            entries.append((f"{folder}/{filename}", path))
            files.append({"name": filename, "bytes": path.stat().st_size})

        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        indexed_shards = sorted(set(index["weight_map"].values()))
        expected_shards = sorted(
            filename for filename in REQUIRED_FILES if filename.endswith(".safetensors")
        )
        if indexed_shards != expected_shards:
            raise ValueError(
                f"Weight index mismatch for {folder}: {indexed_shards} != {expected_shards}"
            )
        manifest["checkpoints"][folder] = {
            "source": str(checkpoint),
            "files": files,
            "total_bytes": sum(item["bytes"] for item in files),
        }
        checkpoint_entries[folder] = entries
    manifest["total_bytes"] = sum(
        value["total_bytes"] for value in manifest["checkpoints"].values()
    )
    return checkpoint_entries, manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_sha256(sibling: object) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return str(value) if value else None


def verify_uploaded_files(
    api: HfApi,
    repo_id: str,
    local_paths: dict[str, Path],
) -> dict:
    info = call_with_retry(
        "Inspect final repository",
        lambda: api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True),
    )
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    expected = set(local_paths) | {"README.md"}
    missing = sorted(expected - siblings.keys())
    if missing:
        raise RuntimeError(f"Upload completed but files are missing: {missing}")
    if getattr(info, "private", True):
        raise RuntimeError("Repository is not public")

    size_mismatches = []
    for path_in_repo, local_path in local_paths.items():
        remote_size = getattr(siblings[path_in_repo], "size", None)
        local_size = local_path.stat().st_size
        if remote_size != local_size:
            size_mismatches.append(
                {
                    "path": path_in_repo,
                    "local": local_size,
                    "remote": remote_size,
                }
            )
    if size_mismatches:
        raise RuntimeError(f"Remote size verification failed: {size_mismatches}")

    weight_paths = {
        path_in_repo: local_path
        for path_in_repo, local_path in local_paths.items()
        if path_in_repo.endswith(".safetensors")
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        digests = dict(
            zip(
                weight_paths,
                executor.map(sha256_file, weight_paths.values()),
                strict=True,
            )
        )
    hash_mismatches = []
    for path_in_repo, local_digest in digests.items():
        remote_digest = lfs_sha256(siblings[path_in_repo])
        if remote_digest != local_digest:
            hash_mismatches.append(
                {
                    "path": path_in_repo,
                    "local": local_digest,
                    "remote": remote_digest,
                }
            )
    if hash_mismatches:
        raise RuntimeError(f"Remote SHA256 verification failed: {hash_mismatches}")

    return {
        "repo_id": repo_id,
        "public": True,
        "files": len(expected),
        "files_size_verified": len(local_paths),
        "weights_sha256_verified": len(weight_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("MOTION_DYNVLA_ROOT", "/mnt/hdd/hesibo/motion_dynvla")),
    )
    parser.add_argument("--repo-id", default="trantor2nd/Motion-DynVLA")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    code_root = args.project_root / "code" / "DynVLA-GR00T"
    model_card = code_root / "docs" / "MODEL_CARD.md"
    if not model_card.is_file():
        raise FileNotFoundError(model_card)

    checkpoint_entries, manifest = checkpoint_files(args.project_root)
    local_paths = {
        path_in_repo: path
        for entries in checkpoint_entries.values()
        for path_in_repo, path in entries
    }
    manifest["repo_id"] = args.repo_id
    manifest["private"] = False
    manifest["model_card"] = str(model_card)
    print(json.dumps(manifest, indent=2))
    if not args.execute:
        print("Dry run complete. Pass --execute to publish.")
        return

    token = None
    if args.token_file is not None:
        token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f"Missing token in {args.token_file or 'environment variable ' + args.token_env}"
        )
    api = HfApi(token=token)
    call_with_retry(
        "Create or inspect repository",
        lambda: api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        ),
    )
    call_with_retry(
        "Upload model card",
        lambda: api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=model_card,
            path_in_repo="README.md",
            commit_message="Add Motion-DynVLA model card",
        ),
    )
    for folder, entries in checkpoint_entries.items():
        for path_in_repo, path in entries:
            info = call_with_retry(
                f"Inspect remote file {path_in_repo}",
                lambda: api.repo_info(
                    repo_id=args.repo_id,
                    repo_type="model",
                    files_metadata=True,
                ),
            )
            siblings = {sibling.rfilename: sibling for sibling in info.siblings}
            complete = (
                path_in_repo in siblings
                and getattr(siblings[path_in_repo], "size", None) == path.stat().st_size
            )
            if complete:
                print(f"Skipping complete file: {path_in_repo}", flush=True)
                continue
            print(f"Uploading file: {path_in_repo}", flush=True)
            call_with_retry(
                f"Upload {path_in_repo}",
                lambda: api.upload_file(
                    repo_id=args.repo_id,
                    repo_type="model",
                    path_or_fileobj=path,
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload {path_in_repo}",
                ),
            )
            print(f"Committed file: {path_in_repo}", flush=True)
    print(json.dumps(verify_uploaded_files(api, args.repo_id, local_paths), indent=2))


if __name__ == "__main__":
    main()
