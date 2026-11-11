import json
import threading

from minerva.strategy import report


def _payload(close: float) -> dict:
    quote = {"open": [1.0], "high": [2.0], "low": [0.5], "close": [close], "volume": [10]}
    return {"chart": {"result": [{"timestamp": [1767225600], "indicators": {"quote": [quote]}}]}}


def test_write_cache_uses_private_tmp_per_thread(tmp_path, monkeypatch):
    # 같은 임시 파일 이름을 쓰면 동시에 받는 스레드가 서로의 반쯤 쓴 파일을 옮길 수 있다
    monkeypatch.setattr(report, "CACHE_DIR", tmp_path)
    path = report._cache_file("^KS11", "max")
    seen: list[str] = []
    real = report.Path.write_text
    together = threading.Barrier(4, timeout=5)  # 넷이 동시에 살아 있어야 스레드 id가 겹치지 않는다

    def spy(self, text, *a, **k):
        seen.append(self.name)
        together.wait()
        return real(self, text, *a, **k)

    monkeypatch.setattr(report.Path, "write_text", spy)
    threads = [
        threading.Thread(target=report._write_cache, args=(path, _payload(float(i))))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(seen)) == 4
    assert report.parse_yahoo(json.loads(path.read_text()))[0].close in (0.0, 1.0, 2.0, 3.0)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_cache_failure_is_silent_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "x.json"
    target.mkdir()  # 디렉터리라 replace가 실패한다
    report._write_cache(target, _payload(1.0))
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_fetch_yahoo_writes_cache_after_download(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "CACHE_DIR", tmp_path)

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps(_payload(1.5)).encode()

    monkeypatch.setattr(report.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert report.fetch_yahoo("SPY", "1y", ttl=0)[0].close == 1.5
    assert report._cache_file("SPY", "1y").exists()
