import socket
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from minerva import __version__, notify
from minerva.office import router as office_router
from minerva.routes import router as data_router

STATIC_DIR = Path(__file__).parent / "static"

KST = timezone(timedelta(hours=9))


def startup_message() -> str:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    return (
        f"🦉 미네르바 기동 완료\n"
        f"• 시각: {now} (KST)\n"
        f"• 버전: v{__version__} · 호스트 {socket.gethostname()}\n"
        f"• 대시보드: https://minerva-ai.duckdns.org/"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 알림이 느리거나 실패해도 서버 기동은 기다리지 않는다
    threading.Thread(target=notify.send, args=(startup_message(),), daemon=True).start()
    yield


app = FastAPI(title="Minerva", version=__version__, lifespan=lifespan)
app.include_router(office_router)
app.include_router(data_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }
