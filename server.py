#!/usr/bin/env python3
"""Serve the x-nav browser viewer, OpenAPI and a single inference worker."""

import argparse
import asyncio
import contextlib
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from loaders.xnav_loader import processing_frame_id
from utils.xnav_service import (
    REPO,
    FrameData,
    FramePage,
    ImportRequest,
    InferenceInfo,
    InferenceOptions,
    JobInfo,
    JobManager,
    JobRequest,
    MapDirectories,
    MapInfo,
    MapRequest,
    SceneInfo,
    list_map_directories,
)
from utils.xnav_web_scene import cloud_bytes, render_observation


class ErrorResponse(BaseModel):
    detail: str | list[dict]


class HealthInfo(BaseModel):
    status: str
    cuda_available: bool
    device: str
    active_job_id: str | None
    inference_status: Literal["loading", "ready", "failed"]
    inference_error: str | None = None


class LogInfo(BaseModel):
    text: str


ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 409, 422, 500, 503)}


def create_app(
    output_dir="output/server", *, manager=None, web_dir=None, map_root="/nav_map"
):
    manager = manager or JobManager(output_dir)
    web_dir = Path(web_dir) if web_dir is not None else REPO / "web" / "dist"

    @asynccontextmanager
    async def lifespan(app):
        import torch

        available = torch.cuda.is_available()
        app.state.hardware = {
            "cuda_available": available,
            "device": torch.cuda.get_device_name(0) if available else "cpu",
        }
        await asyncio.to_thread(manager.open)

        async def monitor():
            while True:
                await asyncio.to_thread(manager.poll)
                await asyncio.sleep(0.3)

        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(manager.close)

    app = FastAPI(
        title="Boxer x-nav API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        description="Local/LAN map inference and committed-result replay. Nanosecond timestamps are strings. Cloud responses contain little-endian float32 xyzrgb vertices (24 bytes per point).",
    )
    app.state.manager = manager
    app.state.stopping = False

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    def input_error(request, error):
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.get("/api/health", response_model=HealthInfo, responses=ERRORS)
    def health():
        inference = manager.inference_info()
        return {
            "status": "ok",
            **app.state.hardware,
            "active_job_id": manager.active_id,
            "inference_status": inference["status"],
            "inference_error": inference["error"],
        }

    @app.get("/api/inference", response_model=InferenceInfo, responses=ERRORS)
    def inference_info():
        return manager.inference_info()

    @app.put(
        "/api/inference",
        response_model=InferenceInfo,
        status_code=202,
        responses=ERRORS,
    )
    def apply_inference(body: InferenceOptions):
        return manager.configure(body)

    @app.post("/api/maps", response_model=MapInfo, responses=ERRORS)
    def add_map(body: MapRequest):
        return manager.add_map(body.path)

    @app.get(
        "/api/maps/directories",
        response_model=MapDirectories,
        responses=ERRORS,
        description=(
            "Recursively list x-nav map packages under /nav_map, sorted by relative "
            "path. Directory symlinks are not followed. Discovery does not perform "
            "full map validation; POST /api/maps still accepts full server-local "
            "paths outside this root."
        ),
    )
    def map_directories():
        return list_map_directories(map_root)

    @app.get("/api/maps/{map_id}", response_model=MapInfo, responses=ERRORS)
    def map_info(map_id: str):
        return manager.map_info(map_id)

    @app.get("/api/maps/{map_id}/scene", response_model=SceneInfo, responses=ERRORS)
    def scene_info(map_id: str):
        return manager.get_map(map_id).scene()

    def binary(data, request, media_type):
        tag = '"' + hashlib.sha256(data).hexdigest() + '"'
        headers = {"ETag": tag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=headers)
        return Response(data, media_type=media_type, headers=headers)

    cloud_response = {
        200: {
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        }
    }
    image_response = {
        200: {
            "content": {
                "image/jpeg": {"schema": {"type": "string", "format": "binary"}}
            }
        }
    }

    @app.get(
        "/api/maps/{map_id}/cloud",
        response_class=Response,
        responses={**ERRORS, **cloud_response},
    )
    def map_cloud(map_id: str, request: Request):
        scene = manager.get_map(map_id)
        scene.scene()
        return binary(scene.cloud, request, "application/octet-stream")

    @app.get("/api/jobs", response_model=list[JobInfo], responses=ERRORS)
    def list_jobs():
        with manager.lock:
            return [manager.info(key) for key in reversed(manager.jobs)]

    @app.post("/api/jobs", response_model=JobInfo, status_code=202, responses=ERRORS)
    def start_job(body: JobRequest):
        return manager.start(body)

    # Literal routes precede the dynamic job-ID route.
    @app.post("/api/jobs/import", response_model=JobInfo, responses=ERRORS)
    def import_job(body: ImportRequest):
        return manager.import_results(body)

    @app.get("/api/jobs/{job_id}", response_model=JobInfo, responses=ERRORS)
    def job_info(job_id: str):
        return manager.info(job_id)

    @app.post(
        "/api/jobs/{job_id}/stop",
        response_model=JobInfo,
        status_code=202,
        responses=ERRORS,
    )
    def stop_job(job_id: str):
        return manager.stop(job_id)

    @app.get("/api/jobs/{job_id}/log", response_model=LogInfo, responses=ERRORS)
    def job_log(job_id: str):
        return {"text": manager.log(job_id)}

    @app.get(
        "/api/jobs/{job_id}/events",
        response_class=StreamingResponse,
        responses={
            **ERRORS,
            200: {
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
                "description": "Initial snapshot, then changed job summaries; reconnect always sends the latest snapshot.",
            },
        },
    )
    async def events(job_id: str, request: Request):
        manager.info(job_id)

        async def stream():
            previous = None
            heartbeat = 0
            while not app.state.stopping and not await request.is_disconnected():
                data = json.dumps(manager.info(job_id), ensure_ascii=False)
                if data != previous:
                    yield f"event: job\ndata: {data}\n\n"
                    previous = data
                elif heartbeat >= 30:
                    yield ": heartbeat\n\n"
                    heartbeat = 0
                heartbeat += 1
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/frames", response_model=FramePage, responses=ERRORS)
    def frames(
        job_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        return manager.frame_page(job_id, offset, limit)

    @app.get(
        "/api/jobs/{job_id}/frames/{index}", response_model=FrameData, responses=ERRORS
    )
    def frame(job_id: str, index: int):
        return manager.frame(job_id, index)

    def frame_context(job_id, index):
        with manager.lock:
            frame = manager.frame(job_id, index)
            replay = manager.replay(job_id)
            scene = manager.get_map(manager.info(job_id)["map_id"])
            return frame, replay.results["parameters"], scene

    @app.get(
        "/api/jobs/{job_id}/frames/{index}/images/{kind}",
        response_class=Response,
        responses={**ERRORS, **image_response},
    )
    def image(
        job_id: str,
        index: int,
        kind: Literal["2d", "3d"],
        request: Request,
        camera_id: int | None = None,
        view_id: str | None = None,
        tracked: bool = True,
        tentative: bool = False,
    ):
        frame, parameters, scene = frame_context(job_id, index)
        observation = next(
            (
                o
                for o in frame["observations"]
                if (camera_id is None or o["camera_id"] == camera_id)
                and (view_id is None or o["view_id"] == view_id)
            ),
            None,
        )
        if observation is None:
            raise HTTPException(404, "No image observation for this frame/view")

        def make():
            loader = scene.result_loader(parameters)
            panels = render_observation(loader, frame, observation, tracked, tentative)
            ok, encoded = cv2.imencode(
                ".jpg", panels[0 if kind == "2d" else 1], [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
            if not ok:
                raise RuntimeError("Could not encode observation image")
            return encoded.tobytes()

        key = (
            job_id,
            index,
            kind,
            observation["camera_id"],
            observation["view_id"],
            tracked,
            tentative,
        )
        return binary(scene.cached(key, make), request, "image/jpeg")

    @app.get(
        "/api/jobs/{job_id}/frames/{index}/cloud",
        response_class=Response,
        responses={**ERRORS, **cloud_response},
    )
    def frame_cloud(job_id: str, index: int, request: Request):
        frame, parameters, scene = frame_context(job_id, index)

        def make():
            points, rgb = scene.result_loader(parameters).world_points(
                processing_frame_id(frame)
            )
            return cloud_bytes(
                points, rgb / 255.0 if rgb is not None else None, limit=50000
            )[0]

        return binary(
            scene.cached((job_id, index, "cloud"), make),
            request,
            "application/octet-stream",
        )

    @app.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Boxer x-nav API",
            swagger_js_url="/docs-assets/swagger-ui-bundle.js",
            swagger_css_url="/docs-assets/swagger-ui.css",
            swagger_favicon_url="/favicon.svg",
        )

    @app.get("/", include_in_schema=False)
    def index():
        path = web_dir / "index.html"
        if not path.is_file():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Build the Web UI first: cd web && npm ci && npm run build"
                },
            )
        return FileResponse(path)

    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir), name="web")
    return app


def main(*, manager_factory=JobManager):
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--output_dir", default=str(REPO / "output" / "server"))
    args = parser.parse_args()
    app = create_app(manager=manager_factory(args.output_dir))

    class LocalServer(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # End SSE connections before Uvicorn waits for HTTP requests to drain.
            app.state.stopping = True
            for identity in list(app.state.manager.processes):
                app.state.manager.stop(identity)
            super().handle_exit(sig, frame)

    LocalServer(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            timeout_graceful_shutdown=75,
            access_log=False,
        )
    ).run()


if __name__ == "__main__":
    main()
