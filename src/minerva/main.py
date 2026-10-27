from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from minerva import __version__
from minerva.office import router as office_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Minerva", version=__version__)
app.include_router(office_router)
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
