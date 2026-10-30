"""Prompt selection and invalidation helpers, independent of OpenGL."""

import numpy as np


def screen_to_image_coords(sx, sy, rect, width, height):
    if rect is None or width <= 0 or height <= 0:
        return None
    x, y, w, h = rect
    if w <= 0 or h <= 0 or not x <= sx <= x + w or not y <= sy <= y + h:
        return None
    return (sx - x) / w * width, (sy - y) / h * height


class XNavPromptMixin:
    """Synchronize pending interaction with a selected x-nav observation.

    Accepted map-space OBB lists deliberately do not belong to this state.
    The host viewer supplies rendering, navigation and inference methods.
    """

    def _xnav_context_key(self):
        source = getattr(self, "_xnav_loader", None)
        if source is None or self.total_frames == 0:
            return None
        return source.context_key(self.current_frame_idx)

    def _xnav_can_prompt(self):
        source = getattr(self, "_xnav_loader", None)
        return source is None or (
            self.total_frames > 0 and source.has_image(self.current_frame_idx)
        )

    def _reset_pending_prompt(self, clear_image_rect=True):
        self._drawing = False
        self._draw_start = self._draw_end = None
        self._draw_start_screen = self._draw_end_screen = None
        self._owl_stage = 0
        self._owl_cached_datum = None
        self._owl_context_key = None
        self._owl_2d_boxes = []
        self._owl_2d_scores = []
        self._owl_2d_colors = []
        self._owl_3d_confs = []
        self._flash_time = 0.0
        self._flash_screen_pos = None
        if clear_image_rect:
            self._img_screen_rect = None

    def _xnav_box_mask(self, boxes):
        boxes = (
            boxes.detach().cpu().numpy()
            if hasattr(boxes, "detach")
            else np.asarray(boxes)
        )
        boxes = boxes.reshape(-1, 4)
        keep = np.zeros(len(boxes), dtype=bool)
        if not self._xnav_can_prompt():
            return keep
        source = getattr(self, "_xnav_loader", None)
        if source is None:
            return np.ones(len(boxes), dtype=bool)
        datum = source.load(self.current_frame_idx)
        finite = np.isfinite(boxes).all(axis=1)
        keep[finite] = datum["view_geometry"].box_is_valid(boxes[finite])
        return keep

    def _pause_for_xnav_prompt(self):
        if getattr(self, "_xnav_loader", None) is not None:
            self.is_playing = False
            self.follow_view = False

    def _select_xnav_view(self, camera_id, view_id=None):
        self._xnav_loader.select_view(camera_id, view_id)
        self._pause_for_xnav_prompt()
        self._reset_pending_prompt()
        self._rgb_lru_cache.clear()
        self._step_to_frame(self.current_frame_idx)

    def _clear_missing_xnav_image(self):
        if self._xnav_can_prompt():
            return
        self._reset_pending_prompt()
        if self._rgb_texture is not None:
            self.imgui.remove_texture(self._rgb_texture)
            self._rgb_texture.release()
            self._rgb_texture = None
        self._rgb_tex_size = (0, 0)
        self._rgb_vrs_w = self._rgb_vrs_h = 0
        self._rgb_img_scale = 1.0
        self.frustum_instance_count = 0
        self.outline_instance_count = 0
        self._prompted_rgb_lines = []
        self._prompted_rgb_labels = []

    def _xnav_pending_is_current(self):
        return getattr(self, "_xnav_loader", None) is None or (
            self._xnav_can_prompt()
            and self._owl_context_key == self._xnav_context_key()
        )
