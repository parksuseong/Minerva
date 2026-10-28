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
        out += [f for f in candidates if f.is_file() and not _is_secret(f)]
    return out


def build(root: Path = ROOT) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files(root):
            tar.add(f, arcname=str(f.relative_to(root)))
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    now = datetime.now(kis.KST)
    data = build()
    if len(data) > MAX_BYTES:
        notify.send(f"⚠️ 백업이 {len(data) / 1e6:.0f}MB라 텔레그램 한도를 넘어요. 정리가 필요해요")
        return 1
    out = ROOT / "backup" / f"minerva-state-{now:%Y%m%d}.tar.gz"
    kis._write_private(out, "")  # 권한 600으로 먼저 만든다
    out.write_bytes(data)
    ok = notify.send_document(
        out, f"🗄 미네르바 운용 백업 {now:%Y-%m-%d %H:%M} ({len(data):,}바이트)"
    )
    for old in sorted((ROOT / "backup").glob("minerva-state-*.tar.gz"))[:-7]:
        old.unlink()  # 서버에는 최근 7개만 둔다
    print("sent" if ok else "send failed", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
