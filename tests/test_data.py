from pathlib import Path

from anima_rfpar.data import list_image_files


def test_list_image_files_empty(tmp_path: Path) -> None:
    assert list_image_files(tmp_path) == []


def test_list_image_files_filters_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("nope", encoding="utf-8")

    files = list_image_files(tmp_path)
    assert [p.name for p in files] == ["a.jpg", "b.png"]
