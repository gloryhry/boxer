export interface MapDirectory { name: string; path: string }
export interface MapDirectories { root: string; directories: MapDirectory[] }
export interface MapInfo {
  id: string; path: string;
  cameras: { id: number; model: string }[];
  validation: { frames: number; keyframes: number; frame_unit: string; map_id: string; snapshot_id: string };
}
export interface Job {
  id: string; map_id: string; status: string; readonly: boolean;
  created_at: string; output_dir: string; results_path: string;
  options: Record<string, unknown>; processed_frames: number; total_frames: number;
  elapsed_seconds: number; tracking: boolean; device: string; error: string | null;
}
export interface Box {
  center: number[]; quaternion_wxyz: number[]; size: number[];
  label: string; track_id: number; state?: string; prob: number;
}
export interface Observation {
  camera_id: number; view_id: string; image_stamp_ns: string; T_map_view: number[][];
}
export interface Frame {
  index: number; frame_id: number; image_frame_id: number | null; lidar_keyframe_id: number;
  time_ns: string; observations: Observation[]; detections: Box[]; tracks: Box[];
}
export interface SceneInfo { center: number[]; radius: number; point_count: number }
export interface FrameSummary { index: number; position: number[] }
