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


def test_backup_skips_temp_files_and_vanished_files(tmp_path, monkeypatch):
    (tmp_path / "trade").mkdir()
    (tmp_path / "trade" / "positions.json").write_text("{}")
    (tmp_path / "trade" / ".positions.json.123.tmp").write_text("{")
    gone = tmp_path / "trade" / "NO_ENTRY"  # 목록을 만든 뒤 지워진 파일
    real_files = backup.files
    monkeypatch.setattr(backup, "files", lambda root: real_files(root) + [gone])
    names = tarfile.open(fileobj=io.BytesIO(backup.build(tmp_path))).getnames()
    assert names == ["trade/positions.json"]


def test_backup_main_writes_private_file(tmp_path, monkeypatch):
    (tmp_path / "trade").mkdir()
    (tmp_path / "trade" / "positions.json").write_text("{}")
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    monkeypatch.setattr(backup.notify, "send_document", lambda path, caption="": True)
    assert backup.main([]) == 0
    (out,) = (tmp_path / "backup").iterdir()
    assert out.stat().st_mode & 0o777 == 0o600
    assert tarfile.open(out).getnames() == ["trade/positions.json"]
