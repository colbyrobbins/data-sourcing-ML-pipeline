"""Unit tests for reproducible dataset ingestion and manifest verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml_pipeline.data.ingest import (
    DEFAULT_MANIFEST_PATH,
    DatasetVerificationError,
    fetch_dataset,
    load_manifest,
    main,
    verify_dataset,
)


def _write_test_manifest(tmp_path: Path, files_payload: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "sample-dataset",
                "source": "kaggle",
                "handle": "owner/sample-dataset",
                "version": 1,
                "url": "https://www.kaggle.com/datasets/owner/sample-dataset/versions/1",
                "files": files_payload,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _create_sample_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "upstream"
    nested = source_dir / "Nested Folder"
    nested.mkdir(parents=True)

    csv_bytes = b"col_a,col_b\n1,2\n"
    csv_file = nested / "data.csv"
    csv_file.write_bytes(csv_bytes)

    manifest_path = _write_test_manifest(
        tmp_path,
        [
            {
                "archive_path": "Nested Folder/data.csv",
                "local_path": "data.csv",
                "size_bytes": len(csv_bytes),
                "sha256": hashlib.sha256(csv_bytes).hexdigest(),
            }
        ],
    )
    return manifest_path, source_dir


def test_repo_manifest_is_valid() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    assert manifest.handle == "mkhoirulmustaqim/student-depression-detection-dataset"
    assert manifest.version == 1
    assert manifest.versioned_handle == "mkhoirulmustaqim/student-depression-detection-dataset/versions/1"
    assert [spec.local_path for spec in manifest.files] == [
        "Student_Depression_Dataset.csv",
        "Student_Depression_Dataset.xlsx",
    ]


def test_fetch_dataset_downloads_verifies_and_skips_when_cached(tmp_path: Path) -> None:
    manifest_path, source_dir = _create_sample_fixture(tmp_path)
    output_dir = tmp_path / "raw"
    calls: list[tuple[str, bool]] = []

    def fake_downloader(handle: str, force: bool) -> str:
        calls.append((handle, force))
        return str(source_dir)

    paths = fetch_dataset(manifest_path, output_dir, downloader=fake_downloader)
    assert calls == [("owner/sample-dataset/versions/1", False)]
    assert paths == [output_dir.resolve() / "data.csv"]
    assert paths[0].read_bytes() == b"col_a,col_b\n1,2\n"

    def fail_if_called(handle: str, force: bool) -> str:
        raise AssertionError(f"Downloader should not be called when already verified: {handle}, {force}")

    second_paths = fetch_dataset(manifest_path, output_dir, downloader=fail_if_called)
    assert second_paths == paths


def test_fetch_dataset_force_redownloads(tmp_path: Path) -> None:
    manifest_path, source_dir = _create_sample_fixture(tmp_path)
    output_dir = tmp_path / "raw"
    calls: list[tuple[str, bool]] = []

    def fake_downloader(handle: str, force: bool) -> str:
        calls.append((handle, force))
        return str(source_dir)

    fetch_dataset(manifest_path, output_dir, downloader=fake_downloader)
    (output_dir / "data.csv").write_bytes(b"stale")

    paths = fetch_dataset(manifest_path, output_dir, force=True, downloader=fake_downloader)
    assert calls == [
        ("owner/sample-dataset/versions/1", False),
        ("owner/sample-dataset/versions/1", True),
    ]
    assert paths[0].read_bytes() == b"col_a,col_b\n1,2\n"


def test_fetch_dataset_rejects_bad_checksum_before_writing_output(tmp_path: Path) -> None:
    manifest_path, source_dir = _create_sample_fixture(tmp_path)
    (source_dir / "Nested Folder" / "data.csv").write_bytes(b"corrupted12345")
    output_dir = tmp_path / "raw"

    with pytest.raises(DatasetVerificationError):
        fetch_dataset(manifest_path, output_dir, downloader=lambda _h, _f: str(source_dir))

    assert not (output_dir / "data.csv").exists()


def test_verify_only_detects_missing_and_tampered_files(tmp_path: Path) -> None:
    manifest_path, source_dir = _create_sample_fixture(tmp_path)
    output_dir = tmp_path / "raw"

    with pytest.raises(DatasetVerificationError, match="Missing file"):
        fetch_dataset(manifest_path, output_dir, verify_only=True)

    fetch_dataset(manifest_path, output_dir, downloader=lambda _h, _f: str(source_dir))
    assert fetch_dataset(manifest_path, output_dir, verify_only=True) == [output_dir.resolve() / "data.csv"]

    (output_dir / "data.csv").write_bytes(b"col_a,col_b\n9,9\n")
    with pytest.raises(DatasetVerificationError, match="SHA-256 mismatch"):
        verify_dataset(output_dir, load_manifest(manifest_path))


def test_rejects_path_traversal_and_invalid_manifest(tmp_path: Path) -> None:
    manifest_path = _write_test_manifest(
        tmp_path,
        [
            {
                "archive_path": "data.csv",
                "local_path": "../escape.csv",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    )
    manifest = load_manifest(manifest_path)
    with pytest.raises(ValueError, match="escapes base directory"):
        verify_dataset(tmp_path / "raw", manifest)

    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps({"dataset": "x", "source": "kaggle", "handle": "invalid", "version": 1, "files": []}))
    with pytest.raises(ValueError):
        load_manifest(bad_manifest)


def test_cli_main_exit_codes(tmp_path: Path) -> None:
    manifest_path, source_dir = _create_sample_fixture(tmp_path)
    output_dir = tmp_path / "raw"

    assert main(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--verify-only"]) == 1

    fetch_dataset(manifest_path, output_dir, downloader=lambda _h, _f: str(source_dir))
    assert main(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--verify-only"]) == 0
