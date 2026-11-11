"""운용 상태 백업. 주문 기록·봇 보유·계좌·분석 스냅숏을 묶어 텔레그램으로 보낸다.

`~/.minerva/trade`는 git에 넣지 않는다(개인 금융 정보). EC2가 망가져도 남도록 서버 밖
(사용자 텔레그램 대화)에 하루 한 번 사본을 둔다. 키·토큰 파일은 절대 넣지 않는다.
실행: `python -m minerva.backup` (평일 16:10, minerva-backup.timer)
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path

from minerva import kis, notify

ROOT = Path(os.environ.get("MINERVA_HOME", "~/.minerva")).expanduser()
INCLUDE = ("trade", "account/latest.json", "daily/latest.json", "daily/history")
# 이름에 이게 들어가면 무조건 뺀다 (키·토큰)
SECRET_MARKERS = (".env", "token", "secret", "key")
MAX_BYTES = 45 * 1024 * 1024  # 텔레그램 봇 업로드 한도 50MB


def _is_secret(path: Path) -> bool:
    name = path.name.lower()
    return any(m in name for m in SECRET_MARKERS)


def files(root: Path = ROOT) -> list[Path]:
    out = []
    for rel in INCLUDE:
        p = root / rel
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        # 쓰는 중인 임시 파일(원자적 쓰기)은 곧 사라지고 내용도 반쪽이라 뺀다
        out += [
            f
            for f in candidates
            if f.is_file() and not _is_secret(f) and not f.name.endswith(".tmp")
        ]
    return out


def build(root: Path = ROOT) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files(root):
            try:
                tar.add(f, arcname=str(f.relative_to(root)))
            except FileNotFoundError:
                continue  # 목록을 만든 뒤 다른 작업이 지운 파일 (예: NO_ENTRY 해제)
    return buf.getvalue()


def _write_bytes_private(path: Path, data: bytes) -> None:
    """권한 600으로 원자적으로 쓴다. 쓰다 끊겨도 반쪽 백업이 남지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    now = datetime.now(kis.KST)
    data = build(ROOT)
    if len(data) > MAX_BYTES:
        notify.send(f"⚠️ 백업이 {len(data) / 1e6:.0f}MB라 텔레그램 한도를 넘어요. 정리가 필요해요")
        return 1
    out = ROOT / "backup" / f"minerva-state-{now:%Y%m%d}.tar.gz"
    _write_bytes_private(out, data)
    ok = notify.send_document(
        out, f"🗄 미네르바 운용 백업 {now:%Y-%m-%d %H:%M} ({len(data):,}바이트)"
    )
    for old in sorted((ROOT / "backup").glob("minerva-state-*.tar.gz"))[:-7]:
        old.unlink()  # 서버에는 최근 7개만 둔다
    print("sent" if ok else "send failed", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
