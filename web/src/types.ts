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
  workflow?: Workflow | null;
}
export interface WorkflowStage {
  status: string; job_id: string | null; error: string | null;
  reason: string | null; artifacts: Record<string, string>;
}
export interface Workflow {
  status: string; room_segmentation: WorkflowStage; binding: WorkflowStage;
  error: string | null; finished_at: string | null;
  map_export?: { status: string; relative_path: string; error: string | null } | null;
}
export interface Room {
  id: number; name: string; point_count: number; color: number[];
  bounds: number[][]; center: number[]; area_m2: number | null;
}
export interface RoomResult {
  job_id: string; map_id: string; segmentation_job_id: string;
  rooms: Room[]; summary: { room_count: number; assigned_ratio: number };
  artifacts: Record<string, string>;
}
export interface RoomBinding {
  tx_world_object: number; ty_world_object: number; tz_world_object: number;
  qw_world_object: number; qx_world_object: number; qy_world_object: number; qz_world_object: number;
  scale_x: number; scale_y: number; scale_z: number;
  instance: number; name: string; sem_id: number; prob: number;
  room_id: number | null; room_name: string | null; reason: string | null;
  point_count: number; labeled_point_count: number; support_points: number; vote_ratio: number;
}
export interface RoomBindingPage {
  total: number; items: RoomBinding[];
  summary: { object_count: number; assigned_count: number; unassigned_count: number; reason: string | null };
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
