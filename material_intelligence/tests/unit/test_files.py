"""Upload safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from material_intelligence.app.utils.files import FileValidationError, safe_extract_zip, validate_extension


def test_reject_bad_extension():
    with pytest.raises(FileValidationError):
        validate_extension("malware.exe", {".jpg", ".png"})


def test_zip_path_traversal_blocked(tmp_path: Path):
    import io
    import zipfile

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    dest = tmp_path / "out"
    with pytest.raises(FileValidationError):
        safe_extract_zip(zip_path, dest)
