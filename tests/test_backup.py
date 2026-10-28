import io
import tarfile

from minerva import backup


def test_backup_excludes_secrets(tmp_path):
    (tmp_path / "trade").mkdir()
    (tmp_path / "trade" / "journal.jsonl").write_text("{}\n")
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "latest.json").write_text("{}")
    (tmp_path / "account" / "token.json").write_text("SECRET")
    (tmp_path / "kis.env").write_text("SECRET")
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "latest.json").write_text("{}")
    names = tarfile.open(fileobj=io.BytesIO(backup.build(tmp_path))).getnames()
    assert sorted(names) == ["account/latest.json", "daily/latest.json", "trade/journal.jsonl"]


def test_secret_markers():
    from pathlib import Path

    assert backup._is_secret(Path("view.token")) and backup._is_secret(Path("telegram.env"))
    assert not backup._is_secret(Path("positions.json"))
