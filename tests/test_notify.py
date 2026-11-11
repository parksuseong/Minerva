import json

from minerva import main, notify


def test_load_config_from_env_file(tmp_path, monkeypatch):
    env = tmp_path / "telegram.env"
    env.write_text("TELEGRAM_BOT_TOKEN=abc:123\nTELEGRAM_CHAT_ID=42\n")
    monkeypatch.setattr(notify, "ENV_FILE", env)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.load_config() == ("abc:123", "42")


def test_send_skips_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.send("hi") is False


def test_send_posts_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    assert notify.send("기동") is True
    assert sent["url"].endswith("/botabc:123/sendMessage")
    assert sent["body"] == {"chat_id": "42", "text": "기동"}


def test_send_failure_returns_false(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def boom(req, timeout):
        raise OSError("network down")

    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.send("x") is False


def test_startup_message_mentions_minerva():
    assert "미네르바 기동" in main.startup_message()


def test_send_returns_false_when_config_unreadable(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    env = tmp_path / "telegram.env"
    env.mkdir()  # 읽으면 IsADirectoryError: 작업이 죽지 않고 False여야 한다
    monkeypatch.setattr(notify, "ENV_FILE", env)
    assert notify.send("hi") is False


def test_send_document_missing_file_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def no_network(*a, **k):
        raise AssertionError("should not send")

    monkeypatch.setattr(notify.urllib.request, "urlopen", no_network)
    assert notify.send_document(tmp_path / "gone.tar.gz") is False
