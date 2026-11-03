"""Snapshot replay and live inference on the OpenGL/ImGui orbit viewer."""

import colorsys
import time

import cv2
import moderngl
import numpy as np

import utils.imgui_compat as imgui
from loaders.xnav_loader import XNavMapLoader, processing_frame_id, read_pcd
from utils.image import put_text
from utils.orbit_camera import OrbitCamera
from utils.tw.obb import BB3D_LINE_ORDERS
from utils.viewer_3d import OrbitViewer, _perspective_projection, launch_viewer
from utils.xnav_pipeline import XNavReplay, empty_obbs, source_signature
from utils.xnav_web_scene import render_observation


def _color(identity):
    return colorsys.hsv_to_rgb((max(0, identity) * 0.61803398875) % 1, 0.8, 0.95)


def launch_xnav_viewer(args):
    live = getattr(args, "live", False)
    if args.start_n < 0 or args.skip_n < 1 or args.max_n < 0:
        raise ValueError("Invalid replay frame range")
    loader = XNavMapLoader(args.input, camera_mode=args.xnav_camera_mode)
    replay = XNavReplay(
        loader.result_file(args.output_dir), args.xnav_camera_mode, live=live
    )
    signature = source_signature(loader)

    def select_frames():
        if replay.results["source_signature"] != signature:
            raise ValueError(
                "Map changed since inference; replay would use mismatched geometry"
            )
        parameters = replay.results["parameters"]
        loader.size = parameters["image_size"]
        loader.fov = parameters["fov_degrees"]
        replay.frames = replay.results["frames"][args.start_n :: args.skip_n]
        if args.max_n > 0:
            replay.frames = replay.frames[: args.max_n]

    select_frames()
    if not replay.frames and not live:
        raise ValueError("No frames in selected replay range")

    class Viewer(OrbitViewer):
        title = (
            "Boxer — x-nav live inference"
            if live
            else "Boxer — x-nav static object replay"
        )
        window_size = (args.window_w or 1600, args.window_h or 900)

        def __init__(self, **kwargs):
            self.index = 0
            self.playing = args.autoplay
            self.follow_latest = live
            self.last_refresh = 0.0
            self.live_error = None
            self.fps = 5.0
            self.last_step = time.monotonic()
            self.view_index = 0
            self.show_tracks = replay.results["tracking"]["enabled"]
            self.show_tentative = args.xnav_show_tentative
            self.show_map = True
            self.show_obs = args.init_show_obs
            self.texture = None
            self.image_key = None
            self.buffers = {}
            self.current_boxes = empty_obbs()
            self.ui_panel_width = 470
            self.orbit = None
            super().__init__(**kwargs)
            imgui.get_style().font_scale_main = 1.2 * self.wnd.pixel_ratio
            self.bg_color_index = 0
            self.orbit = OrbitCamera(
                self.scene_center, self.scene_radius,
                follow=getattr(args, "init_follow", False),
            )
            self.orbit.overview(self._viewport_aspect())
            self.orbit.follow = getattr(args, "init_follow", False)
            self._camera_pose(snap=True)

        def _viewport_aspect(self):
            width, height = self.wnd.size
            return max(width - self.ui_panel_width, 1) / max(height, 1)

        def _camera_pose(self, snap=False):
            if self.orbit is None:
                return
            observations = replay.frame(self.index)["observations"] if replay.frames else []
            pose = observations[self.view_index].get("T_map_view") if observations else None
            self.orbit.set_pose(pose, snap=snap)

        def get_camera_matrices(self):
            if self.orbit is not None:
                self.camera_target = self.orbit.target.astype(np.float32)
                self.camera_distance = self.orbit.distance
                self.camera_azimuth = self.orbit.azimuth
                self.camera_elevation = self.orbit.elevation
            _, view, _ = super().get_camera_matrices()
            near, far = self.orbit.clipping if self.orbit is not None else (0.05, 1000)
            projection = _perspective_projection(
                45.0, self._viewport_aspect(), near, far,
            )
            return projection, view, view @ projection

        def _geometry(self, name, positions, colors):
            if name in self.buffers:
                vao, vbo = self.buffers.pop(name)
                vao.release()
                vbo.release()
            if not len(positions):
                return
            colors = np.broadcast_to(colors, (len(positions), 3))
            data = np.column_stack([positions, colors]).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.program, [(vbo, "3f 3f", "in_position", "in_color")]
            )
            self.buffers[name] = vao, vbo

        def init_scene(self):
            self.program = self.ctx.program(
                vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_position;
                in vec3 in_color;
                out vec3 color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = 2.0;
                    color = in_color;
                }
            """,
                fragment_shader="""
                #version 330
                in vec3 color;
                out vec4 fragColor;
                void main() { fragColor = vec4(color, 1.0); }
            """,
            )
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            cloud_path = loader.root / "global_map.pcd"
            if cloud_path.exists():
                points, _ = read_pcd(cloud_path)
            else:
                key = (
                    processing_frame_id(replay.frames[0])
                    if replay.frames
                    else loader.frame_ids[0]
                )
                points, _ = loader.world_points(key)
            if len(points) > 500000:
                points = points[np.linspace(0, len(points) - 1, 500000, dtype=int)]
            trajectory = np.asarray(
                [np.array(f["T_map_base"])[:3, 3] for f in replay.frames]
            ).reshape(-1, 3)
            self.scene_center = (
                np.median(points, axis=0)
                if len(points)
                else loader.poses[loader.frame_ids[0]][:3, 3]
            )
            self.scene_radius = max(
                5.0,
                float(
                    np.linalg.norm(
                        np.quantile(points, 0.95, axis=0)
                        - np.quantile(points, 0.05, axis=0)
                    )
                )
                if len(points)
                else 5.0,
            )
            self._geometry("map", points, (0.5, 0.55, 0.6))
            self._geometry("trajectory", trajectory, (0.15, 0.55, 0.85))
            self._select_frame(len(replay.frames) - 1 if live else 0)

        def _select_frame(self, index, snap=True):
            if not replay.frames:
                self.current_boxes = empty_obbs()
                self._geometry("boxes", [], (1, 1, 1))
                self._geometry("observed", [], (1, 1, 1))
                if self.texture is not None:
                    self.texture.release()
                    self.texture = None
                self.image_key = None
                self.index = 0
                self._camera_pose(snap=snap)
                return
            self.index = max(0, min(index, len(replay.frames) - 1))
            self.current_boxes = replay.boxes(
                self.index,
                tracked=self.show_tracks,
                confirmed_only=not self.show_tentative,
            )
            corners = self.current_boxes.bb3corners_world.numpy()
            edges = np.asarray(BB3D_LINE_ORDERS)
            positions = corners[:, edges].reshape(-1, 3)
            colors = np.repeat(
                [
                    _color(int(o.inst_id.item()))
                    if self.show_tracks
                    else (0.95, 0.4, 0.1)
                    for o in self.current_boxes
                ],
                24,
                axis=0,
            ).reshape(-1, 3)
            self._geometry("boxes", positions, colors)
            key = processing_frame_id(replay.frame(self.index))
            if self.show_obs:
                xyz, rgb = loader.world_points(key)
                step = max(1, len(xyz) // 50000)
                self._geometry(
                    "observed",
                    xyz[::step],
                    rgb[::step] / 255.0 if rgb is not None else (0.2, 0.6, 0.3),
                )
            self.image_key = None
            self._update_image(snap=snap)

        def _update_image(self, snap=True):
            frame = replay.frame(self.index)
            observations = frame["observations"]
            self.view_index = min(self.view_index, max(0, len(observations) - 1))
            self._camera_pose(snap=snap)
            key = self.index, self.view_index, self.show_tracks, self.show_tentative
            if key == self.image_key:
                return
            self.image_key = key
            if self.texture is not None:
                self.texture.release()
                self.texture = None
            if not observations:
                return
            observation = observations[self.view_index]
            top, bottom = render_observation(
                loader, frame, observation, self.show_tracks, self.show_tentative,
            )
            put_text(top, "2D detections", line=0)
            put_text(
                bottom, "Saved tracks" if self.show_tracks else "3D detections", line=0
            )
            panel = cv2.cvtColor(np.vstack([top, bottom]), cv2.COLOR_BGR2RGB)
            self.texture = self.ctx.texture(
                (panel.shape[1], panel.shape[0]), 3, panel.tobytes()
            )
            self.texture.filter = moderngl.LINEAR, moderngl.LINEAR

        def _refresh_live(self, now):
            if not live or self.live_error or now - self.last_refresh < 0.3:
                return
            self.last_refresh = now
            try:
                if not replay.refresh():
                    return
                select_frames()
                self._geometry(
                    "trajectory",
                    np.asarray(
                        [np.array(f["T_map_base"])[:3, 3] for f in replay.frames]
                    ).reshape(-1, 3),
                    (0.15, 0.55, 0.85),
                )
                self._select_frame(
                    len(replay.frames) - 1 if self.follow_latest else self.index,
                    snap=False,
                )
            except (OSError, ValueError, RuntimeError) as error:
                self.live_error = str(error)
                self.follow_latest = False
                print(f"Live result update stopped: {error}", flush=True)

        def render_3d(self, time_value, frame_time):
            now = time.monotonic()
            self._refresh_live(now)
            if self.playing and now - self.last_step >= 1 / max(self.fps, 0.1):
                if self.index + 1 < len(replay.frames):
                    self._select_frame(self.index + 1, snap=False)
                else:
                    self.playing = False
                self.last_step = now
            self.orbit.update(frame_time)
            width, height = self.wnd.fbo.size
            panel = int(self.ui_panel_width * width / max(self.wnd.size[0], 1))
            self.ctx.viewport = (panel, 0, max(width - panel, 1), height)
            _, _, self.mvp = self.get_camera_matrices()
            self.program["mvp"].write(self.mvp.astype("f4").tobytes())
            for name, mode in (
                ("map", moderngl.POINTS),
                ("trajectory", moderngl.LINE_STRIP),
                ("observed", moderngl.POINTS),
                ("boxes", moderngl.LINES),
            ):
                if (
                    name in self.buffers
                    and (name != "map" or self.show_map)
                    and (name != "observed" or self.show_obs)
                ):
                    self.buffers[name][0].render(mode=mode)
            self.ctx.viewport = (0, 0, width, height)

        def on_mouse_press_event(self, x, y, button):
            self.imgui.mouse_press_event(x, y, button)
            self.mouse_dragging = self.mouse_panning = False
            if x >= self.ui_panel_width and not imgui.get_io().want_capture_mouse:
                self.mouse_dragging = button == 1
                self.mouse_panning = button == 2

        def on_mouse_drag_event(self, x, y, dx, dy):
            self.imgui.mouse_drag_event(x, y, dx, dy)
            if x < self.ui_panel_width or imgui.get_io().want_capture_mouse:
                return
            if self.mouse_dragging:
                self.orbit.rotate(dx, dy, self.wnd.size[1])
            elif self.mouse_panning:
                self.orbit.pan(dx, dy, self.wnd.size[1])

        def on_mouse_release_event(self, x, y, button):
            self.imgui.mouse_release_event(x, y, button)
            self.mouse_dragging = self.mouse_panning = False

        def on_mouse_scroll_event(self, x_offset, y_offset):
            io = imgui.get_io()
            io.add_mouse_wheel_event(float(x_offset), float(y_offset))
            mouse_x = io.mouse_pos.x if hasattr(io.mouse_pos, "x") else io.mouse_pos[0]
            if mouse_x >= self.ui_panel_width and not io.want_capture_mouse:
                self.orbit.zoom(y_offset)

        def render_ui(self):
            w, h = self.wnd.size
            imgui.set_next_window_position(0, 0, imgui.ALWAYS)
            imgui.set_next_window_size(self.ui_panel_width, h, imgui.ALWAYS)
            imgui.begin(
                "x-nav live inference" if live else "x-nav replay",
                flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE,
            )
            changed, self.orbit.follow = imgui.checkbox(
                "Follow camera position", self.orbit.follow
            )
            if changed:
                self._camera_pose(snap=True)
            if imgui.button("Focus camera"):
                self.orbit.focus()
            imgui.same_line()
            if imgui.button("Fit map"):
                self.orbit.overview(self._viewport_aspect())
            imgui.text_wrapped("Left: orbit | Right: pan | Wheel: zoom")
            if self.orbit.follow and self.orbit.message:
                imgui.text_wrapped(self.orbit.message)
            if live:
                status = replay.results.get("status", "running")
                imgui.text(
                    f"Inference: {status} | {len(replay.frames)} processed frames"
                )
                if self.live_error or replay.results.get("error"):
                    imgui.text_wrapped(self.live_error or replay.results["error"])
                changed, self.follow_latest = imgui.checkbox(
                    "Follow latest result", self.follow_latest
                )
                if changed and self.follow_latest:
                    self.playing = False
                    self._select_frame(len(replay.frames) - 1)
                imgui.text_wrapped("Closing this window leaves inference running.")
            if not replay.frames:
                imgui.text("Waiting for the first completed frame...")
                imgui.end()
                return
            frame = replay.frame(self.index)
            frame_number = processing_frame_id(frame)
            metadata = loader.frame_metadata(frame_number)
            label = (
                f"Image group {frame_number} | LiDAR {metadata['lidar_keyframe_id']}"
                if metadata["image_frame_id"] is not None
                else f"LiDAR keyframe {frame_number}"
            )
            imgui.text(f"{label} | {self.index + 1}/{len(replay.frames)}")
            imgui.text(f"Map time: {frame['time_ns']} ns")
            changed, index = imgui.slider_int(
                "Frame", self.index, 0, len(replay.frames) - 1
            )
            if changed:
                self.follow_latest = False
                self._select_frame(index)
            changed, self.playing = imgui.checkbox("Play", self.playing)
            if changed and self.playing:
                self.follow_latest = False
            _, self.fps = imgui.slider_float("Speed", self.fps, 0.2, 20.0)
            changed, self.show_tracks = imgui.checkbox("Saved tracks", self.show_tracks)
            tent_changed, self.show_tentative = imgui.checkbox(
                "Include tentative candidates", self.show_tentative
            )
            _, self.show_map = imgui.checkbox("Map point cloud", self.show_map)
            obs_changed, self.show_obs = imgui.checkbox(
                "Current colored cloud", self.show_obs
            )
            if changed or tent_changed or obs_changed:
                self._select_frame(self.index)
            choices = [
                f"cam{o['camera_id']} / {o['view_id']}" for o in frame["observations"]
            ]
            if choices:
                changed, self.view_index = imgui.combo("View", self.view_index, choices)
                if changed:
                    self._update_image()
            if self.texture is not None:
                width = min(
                    self.ui_panel_width - 36,
                    max(100, (h - imgui.get_cursor_pos_y() - 65) / 2),
                )
                imgui.image(self.texture.glo, width, width * 2)
            else:
                imgui.text("No image observation in this frame")
            imgui.text(f"{len(self.current_boxes)} displayed 3D objects")
            imgui.text_wrapped("IDs come from saved snapshots; seeking never retracks.")
            imgui.end()
            draw = imgui.get_foreground_draw_list()
            occupied = []
            for obb in self.current_boxes:
                p = np.r_[obb.T_world_object.t.numpy().reshape(3), 1.0] @ self.mvp
                if p[3] <= 0:
                    continue
                x, y = (p[:2] / p[3] + 1) / 2
                x, y = self.ui_panel_width + x * max(w - self.ui_panel_width, 1), (1 - y) * h
                if self.ui_panel_width < x < w and 0 < y < h:
                    name = obb.reshape(1, -1).text_string()[0]
                    label = (
                        f"#{int(obb.inst_id.item())} {name}"
                        if self.show_tracks
                        else name
                    )
                    if any(
                        abs(x - px) < 130 and abs(y - py) < 22 for px, py in occupied
                    ):
                        continue
                    occupied.append((x, y))
                    draw.add_text(
                        x, y, imgui.get_color_u32_rgba(0.1, 0.15, 0.2, 1.0), label
                    )

    launch_viewer(Viewer)
