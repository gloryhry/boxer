type XY = [number, number];
type XYZ = [number, number, number];

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function finitePair(value: unknown): value is XY {
  return Array.isArray(value) && value.length === 2 && value.every(Number.isFinite);
}

function invalidResult(): never {
  throw new Error('房间轮廓数据格式无效');
}

function equalPoint(a: XY, b: XY) { return a[0] === b[0] && a[1] === b[1]; }

function rings(part: unknown): XY[][] {
  if (!Array.isArray(part) || !part.length) return invalidResult();
  return part.map(ring => {
    if (!Array.isArray(ring) || ring.length < 4 || !ring.every(finitePair)
        || !equalPoint(ring[0], ring[ring.length - 1])) return invalidResult();
    return ring.slice(0, -1);
  });
}

export function buildRoomOutlines(payload: unknown) {
  if (!record(payload) || (payload.schema_version !== 1 && payload.schema_version !== 2) || payload.coordinate_frame !== 'scene_xy'
      || payload.units !== 'm' || payload.volume_type !== 'vertical_prism' || !Array.isArray(payload.rooms)) {
    return invalidResult();
  }
  const min: XYZ = [Infinity, Infinity, Infinity], max: XYZ = [-Infinity, -Infinity, -Infinity];
  const ids = new Set<number>();
  const rooms = payload.rooms.map(room => {
    if (!record(room) || typeof room.id !== 'number' || !Number.isSafeInteger(room.id) || room.id < 0
        || ids.has(room.id) || !record(room.geometry) || !finitePair(room.z_bounds_m)) {
      return invalidResult();
    }
    const { id, geometry, z_bounds_m: [bottom, top] } = room;
    const parts = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
    if ((geometry.type !== 'Polygon' && geometry.type !== 'MultiPolygon') || !Array.isArray(parts)
        || !parts.length || top < bottom) return invalidResult();
    ids.add(id);

    const positions: number[] = [];
    min[2] = Math.min(min[2], bottom);
    max[2] = Math.max(max[2], top);
    for (const ring of parts.flatMap(rings)) {
      // Dense raster contours need only a few uprights to show the extrusion.
      const uprightStep = Math.ceil(ring.length / 12);
      for (let index = 0; index < ring.length; index++) {
        const [x, y] = ring[index], [nextX, nextY] = ring[(index + 1) % ring.length];
        min[0] = Math.min(min[0], x); min[1] = Math.min(min[1], y);
        max[0] = Math.max(max[0], x); max[1] = Math.max(max[1], y);
        if (x !== nextX || y !== nextY) {
          positions.push(x, y, bottom, nextX, nextY, bottom);
          if (top > bottom) positions.push(x, y, top, nextX, nextY, top);
        }
        if (top > bottom && index % uprightStep === 0) positions.push(x, y, bottom, x, y, top);
      }
    }
    return { id, positions };
  });
  return { rooms, bounds: rooms.length ? { min, max } : undefined };
}
