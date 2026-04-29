import os
import httpx
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="Geo API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
    expose_headers=["*"]
)

AI_URL     = os.getenv("AI_SERVICE_URL",     "http://ai_service:8001")
TILING_URL = os.getenv("TILING_SERVICE_URL", "http://tiling_service:8002")
GEE_URL    = os.getenv("GEE_SERVICE_URL",    "http://gee_service:8003")
EXPORT_URL = os.getenv("EXPORT_SERVICE_URL", "http://export_service:8004")

TIMEOUT = httpx.Timeout(600.0)   


# ── AI / U-Net ───────────────────────────────────────────────────────────────
@app.post("/predict_area/")
async def predict_area(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{AI_URL}/predict_area/",
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()


# ── Tiling ───────────────────────────────────────────────────────────────────
@app.post("/upload_orthomosaic/")
async def upload_orthomosaic(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{TILING_URL}/upload_orthomosaic/",
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()

@app.post("/process_orthomosaic/")
async def process_orthomosaic(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{TILING_URL}/process_orthomosaic/",
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()


@app.get("/tiling_status/{job_id}")
async def tiling_status(job_id: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{TILING_URL}/tiling_status/{job_id}")
    return resp.json()


# ── GEE ──────────────────────────────────────────────────────────────────────
@app.post("/search_recent_image/")
async def search_recent_image(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{GEE_URL}/search_recent_image/", json=body)
    return resp.json()


@app.post("/confirm_export/")
async def confirm_export(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{GEE_URL}/confirm_export/", json=body)
    return resp.json()


# ── Export / File serving ────────────────────────────────────────────────────
@app.post("/export_vector/{filename}")
async def export_vector(filename: str, request: Request):
    params = dict(request.query_params)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{EXPORT_URL}/export_vector/{filename}", params=params
        )
    return resp.json()


@app.get("/download/{filename}")
async def download_file(filename: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{EXPORT_URL}/download/{filename}")
    return Response(
        status_code=resp.status_code,
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
    )


@app.get("/legend/{filename}")
async def download_legend(filename: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{EXPORT_URL}/legend/{filename}")
    return Response(content=resp.content, media_type="image/png")
@app.get("/tiles_outputs/{file_id}/{z}/{x}/{y}.png")
async def proxy_tile(file_id: str, z: int, x: int, y: int):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{TILING_URL}/tiles_outputs/{file_id}/{z}/{x}/{y}.png"
        )
    return Response(content=resp.content, media_type="image/png")

@app.get("/temp_outputs/{filename}")
async def proxy_temp_file(filename: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{TILING_URL}/temp_outputs/{filename}")
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream")
    )