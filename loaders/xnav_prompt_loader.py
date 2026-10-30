"""Random-access x-nav observations for the interactive prompt viewer."""

import numpy as np
import torch

from loaders.xnav_loader import XNavMapLoader, read_pcd
from utils.tw.pose import PoseTW


class XNavPromptLoader:
    """One selected native or virtual observation over a keyframe timeline.

    Viewer frame indices and start_frame are zero-based. Image timestamps
    remain in each datum; the navigation timeline uses exact keyframe times.
    The map reader owns calibration and geometry, including V1 conversion.
    """

    def __init__(
        self,
        root,
        cameras="all",
        start_frame=0,
        skip_frames=1,
        max_frames=0,
        size=960,
        camera_mode="native",
    ):
        if start_frame < 0 or max_frames < 0:
            raise ValueError("Prompt frame range must be nonnegative")
        self.map = XNavMapLoader(
            root,
            cameras=cameras,
            start_n=start_frame + 1,
            skip_n=skip_frames,
            max_n=max_frames or None,
            size=size,
            camera_mode=camera_mode,
        )
        self.camera_mode = camera_mode
        self.keyframe_ids = self.map.keyframe_ids
        self.camera_ids = self.map.camera_ids
        self.camera_id = self.camera_ids[0]
        self._view_id = None
        self.timestamps_ns = np.asarray(
            [self.map.pose_times_ns[k] for k in self.keyframe_ids], dtype=np.int64
        )
        self._time_to_index = {int(t): i for i, t in enumerate(self.timestamps_ns)}
        self._cached_key = None
        self._cached_datum = None
        self._background = None

    def __len__(self):
        return len(self.keyframe_ids)

    @property
    def image_size(self):
        return self.map.size

    @image_size.setter
    def image_size(self, value):
        if not isinstance(value, int) or value < 2:
            raise ValueError("Prompt image size must be an integer >= 2")
        self.map.size = value
        self._cached_key = None
        self._cached_datum = None

    def available_views(self, camera_id=None):
        camera_id = self.camera_id if camera_id is None else camera_id
        if camera_id not in self.camera_ids:
            raise ValueError(f"Camera {camera_id} is outside the selected camera set")
        return [view.view_id for view in self.map.views(camera_id)]

    @property
    def view_id(self):
        views = self.available_views()
        if self._view_id not in views:
            self._view_id = "front" if "front" in views else views[0]
        return self._view_id

    def select_view(self, camera_id, view_id=None):
        views = self.available_views(camera_id)
        if view_id is None:
            view_id = (
                self._view_id
                if self._view_id in views
                else ("front" if "front" in views else views[0])
            )
        if view_id not in views:
            raise ValueError(f"Camera {camera_id} has no valid view {view_id}")
        self.camera_id, self._view_id = camera_id, view_id
        self._cached_key = None
        self._cached_datum = None

    def frame_index(self, time_ns):
        try:
            return self._time_to_index[int(time_ns)]
        except KeyError as error:
            raise ValueError(
                f"No x-nav keyframe at navigation time {time_ns}"
            ) from error

    def context_key(self, frame_index):
        if not 0 <= frame_index < len(self):
            raise IndexError("x-nav prompt frame index out of range")
        return (
            self.keyframe_ids[frame_index],
            self.camera_id,
            self.view_id,
            self.image_size,
            self.camera_mode,
        )

    def has_image(self, frame_index):
        key = self.keyframe_ids[frame_index]
        return self.map.image_index[key, self.camera_id]["status"] == "saved"

    def load(self, frame_index):
        key = self.context_key(frame_index)
        if key != self._cached_key:
            self._cached_datum = self.map.load_view(key[0], key[1], key[2])
            if self._cached_datum is not None:
                self._cached_datum["valid_mask"] = self._cached_datum[
                    "view_geometry"
                ].valid_mask
                self._cached_datum["keyframe_time_ns"] = int(
                    self.timestamps_ns[frame_index]
                )
            self._cached_key = key
        return self._cached_datum

    def load_at_time(self, time_ns):
        return self.load(self.frame_index(time_ns))

    def rgb_at_time(self, time_ns):
        datum = self.load_at_time(time_ns)
        if datum is None:
            return None
        image = datum["img0"][0].permute(1, 2, 0).numpy()
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    def camera_and_pose(self, time_ns):
        datum = self.load_at_time(time_ns)
        if datum is None:
            return None, None
        return datum["cam0"], datum["T_world_rig0"]

    def validate(self):
        summary = self.map.validate()
        self.background_points()
        return summary

    def background_points(self):
        """Display geometry only; never used as a prompt's model input."""
        if self._background is None:
            path = self.map.root / "global_map.pcd"
            if path.exists():
                self._background = read_pcd(path)[0].astype(np.float32)
            else:
                self._background = self.map.world_points(self.keyframe_ids[0])[0]
        return self._background

    def sequence_context(self):
        selected = next(
            v for v in self.map.views(self.camera_id) if v.view_id == self.view_id
        )
        cam = selected.cam.clone()
        return {
            "source": "xnav",
            "loader": self,
            "rgb_num_frames": len(self),
            "rgb_timestamps": self.timestamps_ns,
            "rgb_images": None,
            "is_nebula": True,
            "traj": [
                PoseTW.from_matrix(torch.from_numpy(self.map.poses[k]).float())
                for k in self.keyframe_ids
            ],
            "pose_ts": self.timestamps_ns,
            "calibs": [cam for _ in self.keyframe_ids],
            "calib_ts": self.timestamps_ns,
        }
