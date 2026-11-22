import socket
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from minerva import __version__, admin, notify, routes
from minerva.kis import KST
from minerva.office import router as office_router
from minerva.routes import router as data_router

STATIC_DIR = Path(__file__).parent / "static"


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


# API 설명서(/docs·/openapi.json)는 끈다: 바깥 스캐너에게 경로 목록을 보여 줄 이유가 없다
app = FastAPI(
    title="Minerva",
    version=__version__,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(office_router)
app.include_router(data_router)
app.include_router(admin.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate(request, call_next):
    # 캐시 헤더가 없으면 브라우저가 옛 CSS·JS를 한동안 그대로 써서 화면 수정이 안 보인다.
    # no-cache는 매번 ETag로 확인만 하므로 바뀌지 않았으면 본문을 다시 받지 않는다.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    response.headers.update(SECURITY_HEADERS)
    return response


# 검색엔진에 안 나오게(사용자 2026-11-23), 다른 사이트 iframe에 못 넣게, 형식 추측·리퍼러 노출 막기
SECURITY_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Strict-Transport-Security": "max-age=31536000",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
}


@app.get("/robots.txt", include_in_schema=False)
def robots() -> PlainTextResponse:
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


@app.middleware("http")
async def access_log(request: Request, call_next):
    """어드민 탭 접속 기록: 페이지 열기와 토큰 시도만 남긴다(30초 갱신 API·정적 파일은 빼서
    기록이 묻히지 않게). 서로 다른 틀린 토큰을 15분에 10개 보낸 IP는 잠시 토큰 요청을 막는다."""
    path = request.url.path
    token = request.headers.get("x-minerva-token", "")
    ip = admin.request_ip(request)
    ua = request.headers.get("user-agent")
    ref = request.headers.get("referer")
    if token and path.startswith("/api/"):
        area = "admin" if path.startswith("/api/admin") else "account"
        if admin.blocked(ip):
            admin.log_access(admin.access_entry(ip, ua, path, ref, "auth", "blocked"))
            if admin.auth_alert_kind(ip, False, True):
                admin.send_auth_alert("blocked", ip, ua, path)
            return JSONResponse({"detail": "too many attempts"}, status_code=429)
        ok = admin.admin_token_ok(token) if area == "admin" else routes._token_ok(token)
        if not ok:
            admin.note_fail(ip, token)
        kind = admin.auth_alert_kind(ip, ok, False)
        if kind:
            admin.send_auth_alert(kind, ip, ua, path)
        explicit = request.headers.get("x-minerva-login") == "1"
        if admin.should_log_auth(ip, ua or "", area, ok, explicit, token):
            admin.log_access(admin.access_entry(ip, ua, path, ref, "auth", "ok" if ok else "fail"))
    elif admin.is_page(request.method, path):
        admin.log_access(admin.access_entry(ip, ua, path, ref, "page", None))
    return await call_next(request)


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
