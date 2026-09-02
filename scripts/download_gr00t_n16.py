"""Download one pinned GR00T N1.6 snapshot directly into the project model store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    files = sorted(
        (
            {
                "path": str(path.relative_to(args.local_dir)),
                "size_bytes": path.stat().st_size,
            }
            for path in args.local_dir.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        ),
        key=lambda item: item["path"],
    )
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(snapshot_path),
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }
    manifest_path = args.local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"download_complete={snapshot_path}")
    print(f"revision={args.revision}")
    print(f"files={len(files)}")
    print(f"total_size_bytes={manifest['total_size_bytes']}")


if __name__ == "__main__":
    main()
