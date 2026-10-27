from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from minerva import __version__

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Minerva", version=__version__)


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
