"""Reproducible dataset ingestion from Kaggle with version pinning and SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _default_project_root() -> Path:
    """Return the repository root for editable installs, or fall back to cwd."""
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _default_project_root()
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

Downloader = Callable[[str, bool], str]


class DatasetVerificationError(RuntimeError):
    """Raised when dataset files are missing or fail size/checksum verification."""


@dataclass(frozen=True, slots=True)
class DatasetFileSpec:
    """Pinned metadata for a single file inside a Kaggle dataset archive."""

    archive_path: str
    local_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Pinned dataset version and file checksum manifest."""

    dataset: str
    source: str
    handle: str
    version: int
    url: str
    files: tuple[DatasetFileSpec, ...]

    @property
    def versioned_handle(self) -> str:
        """Return the KaggleHub handle pinned to the exact manifest version."""
        return f"{self.handle}/versions/{self.version}"


def _parse_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Manifest field '{field_name}' must be a positive integer, got {value!r}")
    return value


def _parse_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Manifest field '{field_name}' must be a non-negative integer, got {value!r}")
    return value


def _parse_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Manifest field 'sha256' must be a string, got {value!r}")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"Manifest field 'sha256' must be a 64-char hex digest, got {value!r}")
    return normalized


def _resolve_within(base_dir: Path, relative_path: str) -> Path:
    """Resolve a relative path and guard against directory traversal."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"Expected a non-empty relative path, got {relative_path!r}")
    base_resolved = base_dir.resolve()
    target = (base_resolved / relative_path).resolve()
    if not target.is_relative_to(base_resolved):
        raise ValueError(f"Path escapes base directory {base_dir}: {relative_path!r}")
    return target


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> DatasetManifest:
    """Load and validate a pinned dataset manifest JSON file."""
    raw_text = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest at {manifest_path} must be a JSON object")

    handle = str(payload.get("handle", "")).strip()
    parts = [part for part in handle.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"Manifest field 'handle' must look like 'owner/dataset', got {handle!r}")

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Manifest field 'files' must be a non-empty list")

    files: list[DatasetFileSpec] = []
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(f"Each manifest file entry must be an object, got {entry!r}")
        files.append(
            DatasetFileSpec(
                archive_path=str(entry["archive_path"]),
                local_path=str(entry["local_path"]),
                size_bytes=_parse_non_negative_int(entry["size_bytes"], "size_bytes"),
                sha256=_parse_sha256(entry["sha256"]),
            )
        )

    return DatasetManifest(
        dataset=str(payload["dataset"]),
        source=str(payload["source"]),
        handle=handle,
        version=_parse_positive_int(payload["version"], "version"),
        url=str(payload.get("url", "")),
        files=tuple(files),
    )


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(file_path: Path, spec: DatasetFileSpec) -> str | None:
    if not file_path.is_file():
        return f"Missing file: {file_path}"

    actual_size = file_path.stat().st_size
    if actual_size != spec.size_bytes:
        return f"Size mismatch for {file_path}: expected {spec.size_bytes} bytes, got {actual_size}"

    actual_sha256 = compute_sha256(file_path)
    if actual_sha256 != spec.sha256:
        return f"SHA-256 mismatch for {file_path}: expected {spec.sha256}, got {actual_sha256}"

    return None


def verify_dataset(
    target_dir: Path,
    manifest: DatasetManifest,
    *,
    path_attr: Literal["local_path", "archive_path"] = "local_path",
) -> list[Path]:
    """Verify all files in target_dir match the manifest sizes and SHA-256 checksums."""
    resolved_paths: list[Path] = []
    errors: list[str] = []

    for spec in manifest.files:
        rel_path = getattr(spec, path_attr)
        file_path = _resolve_within(target_dir, rel_path)
        resolved_paths.append(file_path)
        if error := _verify_file(file_path, spec):
            errors.append(error)

    if errors:
        raise DatasetVerificationError("\n".join(errors))

    return resolved_paths


def _copy_atomic(src: Path, dest: Path) -> None:
    """Copy src to dest atomically via a temporary file in dest's directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            with src.open("rb") as src_handle:
                shutil.copyfileobj(src_handle, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, dest)
    except BaseException:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _download_with_kagglehub(versioned_handle: str, force_download: bool) -> str:
    import kagglehub

    return kagglehub.dataset_download(versioned_handle, force_download=force_download)


def fetch_dataset(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_RAW_DIR,
    *,
    force: bool = False,
    verify_only: bool = False,
    downloader: Downloader | None = None,
) -> list[Path]:
    """Ensure the pinned dataset is present in output_dir and passes SHA-256 verification."""
    manifest = load_manifest(manifest_path)

    if verify_only:
        return verify_dataset(output_dir, manifest, path_attr="local_path")

    if not force:
        try:
            return verify_dataset(output_dir, manifest, path_attr="local_path")
        except DatasetVerificationError:
            pass

    download_fn = downloader or _download_with_kagglehub
    source_dir = Path(download_fn(manifest.versioned_handle, force))

    # Validate upstream files before touching output_dir so corrupted downloads never replace good data.
    verified_sources = verify_dataset(source_dir, manifest, path_attr="archive_path")

    for src_path, spec in zip(verified_sources, manifest.files, strict=True):
        dest_path = _resolve_within(output_dir, spec.local_path)
        _copy_atomic(src_path, dest_path)

    return verify_dataset(output_dir, manifest, path_attr="local_path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the pinned Kaggle dataset into data/raw/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to dataset manifest JSON (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Target directory for raw dataset files (default: {DEFAULT_RAW_DIR}).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite local files even if they already match the manifest.",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing local files against the manifest without downloading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for reproducible dataset fetching and verification."""
    args = _build_parser().parse_args(argv)
    try:
        paths = fetch_dataset(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            force=args.force,
            verify_only=args.verify_only,
        )
    except (DatasetVerificationError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Dataset ingestion failed: {exc}", file=sys.stderr)
        return 1

    action = "Verified" if args.verify_only else "Ready"
    for path in paths:
        print(f"{action}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
