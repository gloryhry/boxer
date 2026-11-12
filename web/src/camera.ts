import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { buildRoomOutlines } from './room-geometry';
import type { Box, Observation, SceneInfo } from './types';

/** Follow is a translation only. Manual pans accumulate in world coordinates. */
export class FollowTarget {
  enabled = false;
  offset = new THREE.Vector3();
  pose: THREE.Vector3 | null = null;

  setPose(matrix?: number[][]) {
    this.pose = matrix?.length === 4 && matrix.every(row => row.length === 4 && row.every(Number.isFinite))
      ? new THREE.Vector3(matrix[0][3], matrix[1][3], matrix[2][3]) : null;
  }

  pan(delta: THREE.Vector3) { if (this.enabled) this.offset.add(delta); }

  delta(target: THREE.Vector3, seconds: number, snap = false) {
    if (!this.enabled || !this.pose) return new THREE.Vector3();
    return this.pose.clone().add(this.offset).sub(target)
      .multiplyScalar(snap ? 1 : 1 - Math.exp(-Math.max(seconds, 0) / 0.12));
  }
}

const edges = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
const OVERLAY_LINE_WIDTH = 2; // CSS pixels; twice the previous 1px WebGL lines.

function release(object: THREE.Object3D) {
  object.traverse(child => {
    if (child instanceof THREE.Line || child instanceof THREE.Points || child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach(material => material.dispose());
    }
  });
  object.removeFromParent();
}

export class SceneView {
  readonly follow = new FollowTarget();
  readonly camera = new THREE.PerspectiveCamera(45, 1, 0.005, 1000);
  readonly controls: OrbitControls;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly mapGroup = new THREE.Group();
  private readonly roomGroup = new THREE.Group();
  private readonly boxGroup = new THREE.Group();
  private readonly labels: { point: THREE.Vector3; element: HTMLSpanElement }[] = [];
  private readonly cameraMarker = new THREE.Group();
  private map?: THREE.Points;
  private mapVisible = true;
  private observed?: THREE.Points;
  private trajectory?: THREE.Line;
  private center = new THREE.Vector3();
  private radius = 5;
  private roomView = false;
  private roomBounds?: THREE.Sphere;
  private boxBounds?: THREE.Box3;
  private savedMapView?: { target: THREE.Vector3; position: THREE.Vector3; maxDistance: number };
  private lastTarget = new THREE.Vector3();
  private internalChange = false;
  private lastTime = performance.now();

  constructor(private readonly host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor('#f3f6f8');
    this.renderer.domElement.setAttribute('aria-label', '三维点云：左键旋转，右键平移，滚轮缩放');
    this.renderer.domElement.tabIndex = 0;
    host.prepend(this.renderer.domElement);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.controls.minPolarAngle = 0.018;
    this.controls.maxPolarAngle = Math.PI - 0.018;
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 1000;
    this.controls.addEventListener('change', () => {
      if (!this.internalChange && !this.roomView) this.follow.pan(this.controls.target.clone().sub(this.lastTarget));
      this.lastTarget.copy(this.controls.target);
    });
    const grid = new THREE.GridHelper(20, 20, '#c2d1da', '#e1e8ed');
    grid.rotateX(Math.PI / 2);
    grid.position.z = -0.02;
    const sourceAxes = new THREE.AxesHelper(1);
    const axes = this.lineSegments(
      Array.from(sourceAxes.geometry.getAttribute('position').array),
      0xffffff,
      Array.from(sourceAxes.geometry.getAttribute('color').array),
    );
    axes.material.toneMapped = false;
    sourceAxes.dispose();
    this.mapGroup.add(grid, axes, this.cameraMarker);
    this.roomGroup.visible = false;
    this.scene.add(this.mapGroup, this.roomGroup, this.boxGroup);
    this.fit();
    new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      this.renderer.setSize(Math.max(width, 1), Math.max(height, 1));
      this.camera.aspect = width / Math.max(height, 1);
      this.camera.updateProjectionMatrix();
    }).observe(host);
    this.renderer.setAnimationLoop(() => this.render());
  }

  private moveTarget(delta: THREE.Vector3) {
    this.internalChange = true;
    this.camera.position.add(delta);
    this.controls.target.add(delta);
    this.lastTarget.copy(this.controls.target);
    this.internalChange = false;
  }

  private lineSegments(positions: number[], color: THREE.ColorRepresentation, colors?: number[]) {
    const geometry = new LineSegmentsGeometry().setPositions(positions);
    if (colors) geometry.setColors(colors);
    const material = new LineMaterial({
      color, vertexColors: !!colors, worldUnits: false, alphaToCoverage: true,
    });
    material.linewidth = OVERLAY_LINE_WIDTH;
    material.resolution.set(Math.max(this.host.clientWidth, 1), Math.max(this.host.clientHeight, 1));
    // LineSegments2 refreshes resolution before rendering, including after resize.
    return new LineSegments2(geometry, material);
  }

  private cloud(data: ArrayBuffer, size: number) {
    if (data.byteLength % 24 !== 0) throw new Error('点云数据长度无效');
    const values = new Float32Array(data);
    const color = new THREE.Color();
    // The API carries image/sRGB colors; Three's vertex attributes are linear.
    for (let i = 3; i < values.length; i += 6) {
      color.setRGB(values[i], values[i + 1], values[i + 2], THREE.SRGBColorSpace);
      values[i] = color.r; values[i + 1] = color.g; values[i + 2] = color.b;
    }
    const buffer = new THREE.InterleavedBuffer(values, 6);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
    geometry.setAttribute('color', new THREE.InterleavedBufferAttribute(buffer, 3, 3));
    return new THREE.Points(geometry, new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: true }));
  }

  setMap(info: SceneInfo, data: ArrayBuffer) {
    this.setRoomView(false);
    this.clearRoomOutlines();
    if (this.map) release(this.map);
    this.clearFrame();
    this.setTrajectory([]);
    this.map = this.cloud(data, 1.8);
    this.map.visible = this.mapVisible;
    this.mapGroup.add(this.map);
    this.center.fromArray(info.center);
    this.radius = info.radius;
    this.savedMapView = undefined;
    if (!this.roomView) this.fit();
  }

  setObserved(data?: ArrayBuffer) {
    if (this.observed) release(this.observed);
    this.observed = data ? this.cloud(data, 2.5) : undefined;
    if (this.observed) this.mapGroup.add(this.observed);
  }

  showMap(show: boolean) {
    this.mapVisible = show;
    if (this.map) this.map.visible = show;
  }

  setTrajectory(positions: number[][]) {
    if (this.trajectory) release(this.trajectory);
    const geometry = new THREE.BufferGeometry().setFromPoints(positions.map(p => new THREE.Vector3().fromArray(p)));
    this.trajectory = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#2c8ba5' }));
    this.mapGroup.add(this.trajectory);
  }

  setBoxes(boxes: Box[], tracked: boolean) {
    for (const child of [...this.boxGroup.children]) release(child);
    this.labels.splice(0).forEach(label => label.element.remove());
    const positions: number[] = [], colors: number[] = [];
    for (const box of boxes) {
      const q = box.quaternion_wxyz;
      const rotation = new THREE.Quaternion(q[1], q[2], q[3], q[0]);
      const center = new THREE.Vector3().fromArray(box.center);
      const corners = Array.from({ length: 8 }, (_, index) => new THREE.Vector3(
        (index & 1 ? 0.5 : -0.5) * box.size[0],
        (index & 2 ? 0.5 : -0.5) * box.size[1],
        (index & 4 ? 0.5 : -0.5) * box.size[2],
      ).applyQuaternion(rotation).add(center));
      const color = tracked ? new THREE.Color().setHSL((Math.max(0, box.track_id) * 0.61803398875) % 1, 0.7, 0.45) : new THREE.Color('#e47b3b');
      for (const edge of edges) for (const index of edge) {
        positions.push(...corners[index].toArray()); colors.push(color.r, color.g, color.b);
      }
      const element = document.createElement('span');
      element.className = 'object-label';
      element.hidden = true;
      element.textContent = `${tracked ? `#${box.track_id} ` : ''}${box.label}`;
      element.style.borderColor = `#${color.getHexString()}`;
      this.host.append(element);
      this.labels.push({ point: center, element });
    }
    this.boxBounds = positions.length ? new THREE.Box3().setFromArray(positions) : undefined;
    this.boxGroup.add(this.lineSegments(positions, 0xffffff, colors));
  }

  setRoomOutlines(payload: unknown): number {
    const { rooms, bounds } = buildRoomOutlines(payload);
    this.clearRoomOutlines();
    for (const { id, positions } of rooms) {
      const color = new THREE.Color().setHSL((id * 0.61803398875) % 1, 0.7, 0.4);
      this.roomGroup.add(this.lineSegments(positions, color));
    }
    if (bounds) {
      this.roomBounds = new THREE.Box3(
        new THREE.Vector3(...bounds.min), new THREE.Vector3(...bounds.max),
      ).getBoundingSphere(new THREE.Sphere());
      if (this.roomView) this.fit();
    }
    return rooms.length;
  }

  clearRoomOutlines() {
    for (const child of [...this.roomGroup.children]) release(child);
    this.roomBounds = undefined;
  }

  setRoomView(enabled: boolean) {
    if (enabled === this.roomView) return;
    this.roomView = enabled;
    this.roomGroup.visible = enabled;
    this.mapGroup.visible = !enabled;
    this.renderer.domElement.setAttribute('aria-label', `${enabled ? '三维房间框' : '三维点云'}：左键旋转，右键平移，滚轮缩放`);
    if (enabled) {
      this.savedMapView = {
        target: this.controls.target.clone(),
        position: this.camera.position.clone(),
        maxDistance: this.controls.maxDistance,
      };
      for (const label of this.labels) label.element.hidden = true;
      this.fit();
    } else if (this.savedMapView) {
      this.controls.maxDistance = this.savedMapView.maxDistance;
      this.positionCamera(this.savedMapView.target, this.savedMapView.position);
      this.savedMapView = undefined;
    } else {
      this.fit();
    }
  }

  setObservation(observation?: Observation, snap = false) {
    this.follow.setPose(observation?.T_map_view);
    for (const child of [...this.cameraMarker.children]) release(child);
    if (this.follow.pose && observation) {
      const points = [[0, 0, 0], [-0.25, -0.2, 0.45], [0.25, -0.2, 0.45], [0, 0, 0], [0.25, 0.2, 0.45], [-0.25, 0.2, 0.45], [0, 0, 0]];
      const transform = new THREE.Matrix4().fromArray(observation.T_map_view.flat()).transpose();
      const vertices = points.map(p => new THREE.Vector3().fromArray(p).applyMatrix4(transform));
      const positions: number[] = [];
      for (let i = 1; i < vertices.length; i++) {
        positions.push(...vertices[i - 1].toArray(), ...vertices[i].toArray());
      }
      this.cameraMarker.add(this.lineSegments(positions, '#167d90'));
    }
    if (snap && !this.roomView) this.moveTarget(this.follow.delta(this.controls.target, 0, true));
  }

  setFollow(enabled: boolean) {
    this.follow.enabled = enabled;
    if (enabled && !this.roomView) this.moveTarget(this.follow.delta(this.controls.target, 0, true));
  }

  focus() {
    if (this.roomView) return;
    this.follow.offset.set(0, 0, 0);
    if (this.follow.pose) this.moveTarget(this.follow.pose.clone().sub(this.controls.target));
  }

  fit() {
    if (!this.roomView) {
      this.follow.enabled = false;
      this.follow.offset.set(0, 0, 0);
    }
    let bounds = this.roomView ? this.roomBounds?.clone() : undefined;
    if (this.roomView && this.boxBounds) {
      const objects = this.boxBounds.getBoundingSphere(new THREE.Sphere());
      if (bounds) bounds.union(objects);
      else bounds = objects;
    }
    const center = bounds?.center ?? this.center;
    const radius = Math.max(0.1, bounds?.radius ?? this.radius);
    const distance = radius * 2.8 / Math.min(1, Math.max(this.camera.aspect, 0.1));
    this.controls.maxDistance = Math.max(100, radius * 20, distance);
    this.positionCamera(center, center.clone().add(new THREE.Vector3(0.6124, 0.6124, 0.5).multiplyScalar(distance)));
  }

  private positionCamera(target: THREE.Vector3, position: THREE.Vector3) {
    this.internalChange = true;
    // Flush pending damping so an old drag cannot move a newly fitted scene.
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.target.copy(target);
    this.camera.position.copy(position);
    this.controls.update();
    this.controls.enableDamping = damping;
    this.lastTarget.copy(this.controls.target);
    this.internalChange = false;
  }

  clearFrame() { this.setBoxes([], false); this.setObserved(); this.setObservation(); }

  private render() {
    const now = performance.now(), dt = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    this.controls.update();
    if (!this.roomView) this.moveTarget(this.follow.delta(this.controls.target, dt));
    const distance = this.camera.position.distanceTo(this.controls.target);
    this.camera.near = Math.max(0.005, distance / 10000);
    this.camera.far = Math.max(1000, distance + (this.roomView ? this.roomBounds?.radius ?? this.radius : this.radius) * 4);
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    const occupied: [number, number][] = [];
    for (const label of this.labels) {
      const p = label.point.clone().project(this.camera);
      const x = (p.x + 1) * this.host.clientWidth / 2, y = (1 - p.y) * this.host.clientHeight / 2;
      const hidden = p.z < -1 || p.z > 1 || Math.abs(p.x) > 1 || Math.abs(p.y) > 1 || occupied.some(([px, py]) => Math.abs(px - x) < 110 && Math.abs(py - y) < 22);
      label.element.hidden = hidden;
      if (!hidden) { label.element.style.transform = `translate(${x}px, ${y}px)`; occupied.push([x, y]); }
    }
  }
}
