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
  schema_version: 2 | null;
  status: string; room_segmentation: WorkflowStage; binding: WorkflowStage;
  error: string | null; finished_at: string | null;
  map_export?: { status: string; relative_path: string; error: string | null } | null;
}
export type RoomGeometry =
  | { type: 'Polygon'; coordinates: [number, number][][] }
  | { type: 'MultiPolygon'; coordinates: [number, number][][][] };
export interface Room {
  id: number; name: string; area_m2: number;
  geometry: RoomGeometry; z_bounds_m: [number, number];
}
export interface RoomResult {
  schema_version: 2; job_id: string; map_id: string; segmentation_job_id: string;
  coordinate_frame: 'scene_xy'; units: 'm'; volume_type: 'vertical_prism'; resolution_m: number;
  rooms: Room[]; summary: { room_count: number; area_m2: number; volume_m3: number };
  artifacts: Record<string, string>;
}
export interface RoomBinding {
  tx_world_object: number; ty_world_object: number; tz_world_object: number;
  qw_world_object: number; qx_world_object: number; qy_world_object: number; qz_world_object: number;
  scale_x: number; scale_y: number; scale_z: number;
  instance: number; name: string; sem_id: number; prob: number;
  room_id: number | null; room_name: string | null; reason: string | null;
  object_volume_m3: number | null; intersection_volume_m3: number; overlap_ratio: number;
  overlaps: { room_id: number; intersection_volume_m3: number; overlap_ratio: number }[];
}
export interface RoomBindingPage {
  schema_version: 2; binding_method: 'obb_room_volume_v1';
  job_id: string;
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
