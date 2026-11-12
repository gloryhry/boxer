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
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from loaders.xnav_loader import processing_frame_id
from utils.potential_field_service import (
    LogInfo,
    RoomSegmentationDispatcher,
    VoxelConnectivityJobs,
)
from utils.xnav_service import (
    REPO,
    Box2DPage,
    Box3DPage,
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
    RoomBindingPage,
    RoomBindingResult,
    RoomPolygonDocument,
    RoomResult,
    SceneInfo,
    list_map_directories,
)
from utils.xnav_upload import (
    Interrupted,
    UploadInfo,
    UploadList,
    UploadManager,
    UploadRequest,
    storage_error,
)
from utils.xnav_web_scene import cloud_bytes, render_observation
from utils.xnav_workflow import MapWorkflows


class ErrorResponse(BaseModel):
    detail: str | list[dict]


class HealthInfo(BaseModel):
    status: str
    cuda_available: bool
    device: str
    active_job_id: str | None
    active_workflow_id: str | None = None
    inference_status: Literal["loading", "ready", "failed"]
    inference_error: str | None = None


ERRORS = {
    code: {"model": ErrorResponse}
    for code in (400, 404, 409, 413, 415, 422, 500, 503, 507)
}


def create_app(
    output_dir="output/server",
    *,
    manager=None,
    web_dir=None,
    map_root="/nav_map",
    upload_limits=None,
    voxel_connectivity_jobs=None,
):
    manager = manager or JobManager(output_dir)
    uploads = UploadManager(map_root, manager, limits=upload_limits)
    room_output = getattr(manager, "root", output_dir)
    if voxel_connectivity_jobs is None:
        voxel_connectivity_jobs = VoxelConnectivityJobs(room_output)
    room_dispatcher = RoomSegmentationDispatcher(
        manager, voxel_connectivity_jobs
    )
    workflows = MapWorkflows(
        manager, room_dispatcher, voxel_connectivity_jobs, room_output
    )
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
        await asyncio.to_thread(uploads.open)

        async def monitor():
            while True:
                await asyncio.to_thread(manager.poll)
                await asyncio.to_thread(voxel_connectivity_jobs.poll)
                await asyncio.to_thread(workflows.poll)
                await asyncio.sleep(0.3)

        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            finally:
                try:
                    await asyncio.to_thread(workflows.close)
                finally:
                    try:
                        await asyncio.to_thread(voxel_connectivity_jobs.close)
                    finally:
                        await asyncio.to_thread(uploads.close)
                        await asyncio.to_thread(manager.close)
                        await asyncio.to_thread(workflows.poll)

    app = FastAPI(
        title="Boxer x-nav API",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        description=(
            "Local/LAN map inference and committed-result replay. "
            "Nanosecond timestamps are strings. Cloud responses contain "
            "little-endian float32 xyzrgb vertices (24 bytes per point)."
            " POST /api/jobs is the only computation entrypoint: Boxer and "
            "voxel room segmentation run concurrently, followed by volume-based "
            "object binding; "
            "workflow.status describes the complete workflow, status describes Boxer."
        ),
    )
    app.state.manager = manager
    app.state.uploads = uploads
    app.state.voxel_connectivity_jobs = voxel_connectivity_jobs
    app.state.workflows = workflows
    app.state.stopping = False

    @app.exception_handler(Interrupted)
    def upload_interrupted(request, error):
        return JSONResponse(
            status_code=409, content={"detail": "Upload cancelled or service stopping"}
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    def input_error(request, error):
        code = storage_error(error).status_code if isinstance(error, OSError) else 400
        return JSONResponse(status_code=code, content={"detail": str(error)})

    @app.get("/api/health", response_model=HealthInfo, responses=ERRORS)
    def health():
        inference = manager.inference_info()
        return {
            "status": "ok",
            **app.state.hardware,
            "active_job_id": manager.active_id,
            "active_workflow_id": workflows.active_id,
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

    @app.post(
        "/api/maps/uploads",
        response_model=UploadInfo,
        status_code=201,
        responses=ERRORS,
    )
    def create_upload(body: UploadRequest):
        return uploads.create(body)

    @app.get("/api/maps/uploads", response_model=UploadList, responses=ERRORS)
    def list_uploads():
        return uploads.list()

    @app.get(
        "/api/maps/uploads/{upload_id}", response_model=UploadInfo, responses=ERRORS
    )
    def upload_info(upload_id: str):
        return uploads.get(upload_id)

    @app.put(
        "/api/maps/uploads/{upload_id}/chunks/{index}",
        response_model=UploadInfo,
        responses=ERRORS,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    },
                },
            }
        },
        description="Stream one exact-length chunk. Identical retries are idempotent. "
        "Only durably committed chunks appear in received_chunks.",
    )
    async def upload_chunk(
        upload_id: str,
        index: int,
        request: Request,
        x_chunk_sha256: str = Header(pattern=r"^[a-fA-F0-9]{64}$"),
        content_length: int | None = Header(default=None, ge=0),
    ):
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/octet-stream"
        ):
            raise HTTPException(415, "Use application/octet-stream for chunks")
        writer = uploads.begin_chunk(upload_id, index, x_chunk_sha256, content_length)
        try:
            async for data in request.stream():
                if data:
                    await asyncio.to_thread(writer.write, data)
            return await asyncio.to_thread(writer.finish)
        except ClientDisconnect as error:
            raise HTTPException(400, "Upload interrupted; retry this chunk") from error
        except Interrupted as error:
            raise HTTPException(409, "Upload cancelled or service stopping") from error
        finally:
            await asyncio.shield(asyncio.to_thread(writer.abort))

    @app.post(
        "/api/maps/uploads/{upload_id}/complete",
        response_model=UploadInfo,
        status_code=202,
        responses=ERRORS,
    )
    def complete_upload(upload_id: str):
        return uploads.complete(upload_id)

    @app.delete(
        "/api/maps/uploads/{upload_id}", response_model=UploadInfo, responses=ERRORS
    )
    def cancel_upload(upload_id: str):
        return uploads.cancel(upload_id)

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
        return workflows.list()

    @app.post("/api/jobs", response_model=JobInfo, status_code=202, responses=ERRORS)
    def start_job(body: JobRequest):
        if app.state.stopping:
            raise HTTPException(503, "Workflow service is shutting down")
        return workflows.start(body)

    # Literal routes precede the dynamic job-ID route.
    @app.post("/api/jobs/import", response_model=JobInfo, responses=ERRORS)
    def import_job(body: ImportRequest):
        return manager.import_results(body)

    @app.get("/api/jobs/{job_id}", response_model=JobInfo, responses=ERRORS)
    def job_info(job_id: str):
        return workflows.info(job_id)

    @app.post(
        "/api/jobs/{job_id}/stop",
        response_model=JobInfo,
        status_code=202,
        responses=ERRORS,
    )
    def stop_job(job_id: str):
        return workflows.stop(job_id)

    @app.get("/api/jobs/{job_id}/rooms", response_model=RoomResult, responses=ERRORS)
    def job_rooms(job_id: str):
        """Completed room segmentation, available independently of Boxer/binding."""
        return workflows.room_result(job_id)

    @app.get(
        "/api/jobs/{job_id}/room-polygons.json", response_class=FileResponse,
        responses={
            **ERRORS,
            200: {"model": RoomPolygonDocument},
        },
    )
    def download_room_polygons(job_id: str):
        """Immutable Polygon/MultiPolygon footprints with metre-scale Z bounds."""
        return FileResponse(
            workflows.room_polygons_path(job_id), media_type="application/json",
            filename=f"room-polygons-{job_id}.json",
        )

    @app.get(
        "/api/jobs/{job_id}/room-bindings", response_model=RoomBindingPage,
        responses=ERRORS,
    )
    def job_room_bindings(
        job_id: str, offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
        room_id: int | None = Query(None, ge=-1),
    ):
        """Final OBB/prism overlaps; room_id=-1 selects unbound objects."""
        return workflows.binding_page(job_id, offset, limit, room_id)

    @app.get(
        "/api/jobs/{job_id}/room-bindings.json", response_class=FileResponse,
        responses={
            **ERRORS,
            200: {"model": RoomBindingResult},
        },
    )
    def download_room_bindings(job_id: str):
        """Immutable full result with provenance, thresholds and volume evidence."""
        return FileResponse(
            workflows.binding_path(job_id), media_type="application/json",
            filename=f"room-bindings-{job_id}.json",
        )

    @app.get("/api/jobs/{job_id}/log", response_model=LogInfo, responses=ERRORS)
    def job_log(
        job_id: str,
        stage: Literal["boxer", "room_segmentation", "binding"] = Query("boxer"),
    ):
        """Read the selected stage's log using the parent task ID."""
        return {"text": workflows.log(job_id, stage)}

    @app.get(
        "/api/jobs/{job_id}/events",
        response_class=StreamingResponse,
        responses={
            **ERRORS,
            200: {
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
                "description": (
                    "Initial snapshot, then changed job summaries; "
                    "reconnect always sends the latest snapshot."
                ),
            },
        },
    )
    async def events(job_id: str, request: Request):
        workflows.info(job_id)

        async def stream():
            previous = None
            heartbeat = 0
            while not app.state.stopping and not await request.is_disconnected():
                data = json.dumps(workflows.info(job_id), ensure_ascii=False)
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

    @app.get(
        "/api/jobs/{job_id}/3dbbox.zip",
        response_class=FileResponse,
        responses={
            **ERRORS,
            200: {
                "description": "Immutable 3dbbox directory snapshot for this job.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
                "headers": {
                    "Content-Disposition": {
                        "description": "Attachment named 3dbbox-{job_id}.zip",
                        "schema": {"type": "string"},
                    }
                },
            },
        },
        description=(
            "Download the immutable 3dbbox snapshot. New workflows archive after "
            "room results are saved; workflow.status must be complete. Includes "
            "detection CSVs, room segmentation, object bindings and accumulated "
            "DINO features under 3dbbox/. Returns 409 for jobs that "
            "have not completed successfully or used no_csv=true; 404 for unknown "
            "jobs, imported replays, or historical jobs without an archive."
        ),
    )
    def download_job_archive(job_id: str):
        return FileResponse(
            workflows.archive_path(job_id),
            media_type="application/zip",
            filename=f"3dbbox-{job_id}.zip",
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

    @app.get(
        "/api/jobs/{job_id}/3dbbox_tracked",
        response_model=Box3DPage,
        responses=ERRORS,
        description=(
            "Current confirmed global objects from the latest committed frame; "
            "uses final objects after inference completes. CSV column names, "
            "time_ns is the earliest associated image time for new results. "
            "observations_2d contains every associated 2D detection "
            "and source image path. "
            "Paginated by boxes; live pages may change."
        ),
    )
    def tracked_boxes(
        job_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        return manager.box_page(job_id, "3dbbox_tracked", offset, limit)

    @app.get(
        "/api/jobs/{job_id}/3dbbox_tracks",
        response_model=Box3DPage,
        responses=ERRORS,
        description=(
            "Confirmed tracking boxes from all committed frames in frame order. "
            "Excludes TENTATIVE tracks and preserves historical instance IDs. "
            "CSV column names; paginated by boxes, including during inference."
        ),
    )
    def track_boxes(
        job_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        return manager.box_page(job_id, "3dbbox_tracks", offset, limit)

    @app.get(
        "/api/jobs/{job_id}/3dbbox",
        response_model=Box3DPage,
        responses=ERRORS,
        description=(
            "Fused 3D detections from all committed frames in frame order. "
            "CSV column names; paginated by boxes, including during inference."
        ),
    )
    def detection_boxes(
        job_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        return manager.box_page(job_id, "3dbbox", offset, limit)

    @app.get(
        "/api/jobs/{job_id}/owl_2dbbox",
        response_model=Box2DPage,
        responses=ERRORS,
        description=(
            "OWL 2D detections from all committed frames, cameras and views in "
            "saved order. CSV column names and image timestamps; coordinates "
            "refer to preprocessed views. Paginated by boxes during inference."
        ),
    )
    def owl_boxes(
        job_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
        object_id: int | None = Query(
            None, ge=0, description="Filter by canonical object ID."
        ),
    ):
        return manager.box_page(
            job_id, "owl_2dbbox", offset, limit, object_id=object_id
        )

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
                    "detail": (
                        "Build the Web UI first: cd web && npm ci && npm run build"
                    )
                },
            )
        return FileResponse(path)

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
        include_in_schema=False,
    )
    def unknown_api(path: str):
        raise HTTPException(404, "API route not found")

    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir), name="web")
    return app


def main(*, manager_factory=JobManager):
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--output_dir", default=str(REPO / "output" / "server"))
    parser.add_argument("--map_root", default="/nav_map")
    args = parser.parse_args()
    app = create_app(manager=manager_factory(args.output_dir), map_root=args.map_root)

    class LocalServer(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # End SSE connections before Uvicorn waits for HTTP requests to drain.
            app.state.stopping = True
            app.state.uploads.stop.set()
            identity = app.state.workflows.active_id or app.state.manager.active_id
            if identity is not None:
                app.state.workflows.stop(identity)
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
