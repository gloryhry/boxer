import type { SceneView } from './camera';
import type { Box, Job, RoomBinding, RoomBindingPage, RoomResult } from './types';

const reasons: Record<string, string> = {
  invalid_bbox: '包围框无效', no_rooms: '未分割出房间', no_overlap: '与房间没有体积相交',
  insufficient_overlap: '最高体积覆盖率不足 60%', ambiguous_overlap: '多个房间的相交体积并列',
  tracking_disabled: '本次未开启跟踪', no_tracked_objects: '没有最终确认物品',
};

export function bindingBox(item: RoomBinding): Box {
  return {
    center: [item.tx_world_object, item.ty_world_object, item.tz_world_object],
    quaternion_wxyz: [item.qw_world_object, item.qx_world_object, item.qy_world_object, item.qz_world_object],
    size: [item.scale_x, item.scale_y, item.scale_z],
    label: item.name, track_id: item.instance, prob: item.prob,
  };
}

export class RoomResults {
  private job: Job | null = null;
  private metadata?: RoomResult;
  private bindings?: RoomBindingPage;
  private generation = 0;
  private requests = new Set<AbortController>();
  private loadingRooms = false;
  private loadingBindings = false;
  private outlineKey = '';
  private readonly mode: HTMLSelectElement;
  private readonly filter: HTMLSelectElement;
  private readonly content: HTMLElement;
  private readonly availability: HTMLElement;
  private readonly note: HTMLElement;
  private readonly list: HTMLElement;
  private readonly downloads: HTMLElement;

  constructor(host: HTMLElement, private readonly scene: SceneView | undefined,
    private readonly fetchOk: (url: string, options?: RequestInit) => Promise<Response>,
    private readonly onModeChange: () => void, private readonly onError: (error: string) => void,
    private readonly onCount: (count: number) => void) {
    host.innerHTML = `<label for="result-mode">结果视图</label><select id="result-mode"><option value="frames">逐帧回放</option><option value="rooms" disabled>房间归属</option></select>
      <p id="room-availability" class="hint" role="status" aria-live="polite"></p>
      <div id="room-content" hidden><label for="room-filter">房间 / 物品归属</label><select id="room-filter"><option value="">全部房间</option></select><p id="room-note" class="hint" role="status" aria-live="polite"></p><div id="room-downloads" class="room-downloads"></div><ul id="room-objects" class="room-objects" aria-label="最终跟踪物品"></ul></div>`;
    this.mode = host.querySelector<HTMLSelectElement>('#result-mode')!;
    this.filter = host.querySelector<HTMLSelectElement>('#room-filter')!;
    this.content = host.querySelector('#room-content')!;
    this.availability = host.querySelector('#room-availability')!;
    this.note = host.querySelector('#room-note')!;
    this.list = host.querySelector('#room-objects')!;
    this.downloads = host.querySelector('#room-downloads')!;
    this.mode.onchange = () => {
      this.content.hidden = !this.active;
      this.scene?.setRoomView(this.active);
      this.onModeChange();
      this.paint();
    };
    this.filter.onchange = () => this.paint();
  }

  get active() { return this.mode.value === 'rooms'; }
  get roomCount() {
    return this.filter.value !== '' && this.filter.value !== '-1' ? 1 : this.metadata?.rooms.length || 0;
  }

  update(job: Job | null) {
    if (job?.id !== this.job?.id) {
      const wasActive = this.active;
      ++this.generation;
      this.requests.forEach(request => request.abort()); this.requests.clear();
      this.metadata = undefined; this.bindings = undefined;
      this.loadingRooms = this.loadingBindings = false;
      this.outlineKey = ''; this.mode.value = 'frames'; this.content.hidden = true;
      this.filter.replaceChildren(new Option('全部房间', ''));
      this.list.replaceChildren(); this.downloads.replaceChildren();
      this.scene?.setRoomView(false); this.scene?.clearRoomOutlines(); this.scene?.setBoxes([], true);
      if (wasActive) this.onModeChange();
    }
    this.job = job;
    const supported = job?.workflow?.schema_version === 2;
    const roomReady = supported && job.workflow!.room_segmentation.status === 'complete';
    this.mode.options[1].disabled = !roomReady;
    this.availability.textContent = job && !supported ? '该任务为旧格式，房间归属需重新运行任务。'
      : !job ? '运行任务后可查看房间轮廓与物品归属。'
      : roomReady ? '房间轮廓已就绪；归属结果将随任务进度更新。' : '等待当前任务完成房间分割。';
    if (roomReady && !this.metadata && !this.loadingRooms) void this.loadRooms();
    if (supported && job.workflow!.binding.status === 'complete' && !this.bindings && !this.loadingBindings) void this.loadBindings();
    this.renderDownloads(); this.paint();
  }

  private async loadRooms() {
    const identity = this.job!.id, generation = this.generation;
    const request = new AbortController(); this.requests.add(request); this.loadingRooms = true;
    try {
      const result: RoomResult = await (await this.fetchOk(`/api/jobs/${identity}/rooms`, { signal: request.signal })).json();
      if (generation !== this.generation || request.signal.aborted) return;
      if (result.schema_version !== 2 || result.job_id !== identity) throw new Error('房间结果与当前任务不匹配');
      this.metadata = result;
      this.filter.replaceChildren(new Option('全部房间', ''), new Option('未归属物品', '-1'),
        ...result.rooms.map(room => new Option(room.name, String(room.id))));
      this.renderDownloads(); this.paint();
    } catch (error) {
      if (generation === this.generation && !request.signal.aborted) this.onError(`加载房间结果失败：${String(error)}`);
    } finally {
      this.requests.delete(request);
      if (generation === this.generation) this.loadingRooms = false;
    }
  }

  private async loadBindings() {
    const identity = this.job!.id, generation = this.generation;
    const request = new AbortController(); this.requests.add(request); this.loadingBindings = true;
    try {
      const items: RoomBinding[] = [];
      let result: RoomBindingPage;
      do {
        result = await (await this.fetchOk(`/api/jobs/${identity}/room-bindings?offset=${items.length}&limit=1000`, { signal: request.signal })).json();
        if (generation !== this.generation || request.signal.aborted) return;
        if (result.schema_version !== 2 || result.binding_method !== 'obb_room_volume_v1'
            || result.job_id !== identity) throw new Error('物品归属结果格式或任务关联无效');
        items.push(...result.items);
        if (!result.items.length) break;
      } while (items.length < result.total);
      this.bindings = { ...result, items };
      this.outlineKey = '';
      this.renderDownloads(); this.paint();
    } catch (error) {
      if (generation === this.generation && !request.signal.aborted) this.onError(`加载物品归属失败：${String(error)}`);
    } finally {
      this.requests.delete(request);
      if (generation === this.generation) this.loadingBindings = false;
    }
  }

  private renderDownloads() {
    this.downloads.replaceChildren();
    const artifacts = { ...this.metadata?.artifacts, ...this.job?.workflow?.binding.artifacts };
    if (this.job?.workflow?.schema_version !== 2) return;
    if (this.job.workflow.map_export && this.job.workflow.status === 'complete' && !this.job.options.no_csv) artifacts['3dbbox.zip'] = `/api/jobs/${this.job.id}/3dbbox.zip`;
    const names: Record<string, string> = { 'room_polygons.json': '房间几何 JSON', 'room-bindings.json': '物品归属 JSON', '3dbbox.zip': '完整结果 ZIP' };
    for (const [name, url] of Object.entries(artifacts)) {
      if (!names[name]) continue;
      const link = document.createElement('a'); link.href = url;
      link.textContent = names[name]; link.download = name;
      this.downloads.append(link);
    }
  }

  private paint() {
    if (!this.active) return;
    const value = this.filter.value;
    const selected = value === '' ? undefined : Number(value);
    const items = (this.bindings?.items || []).filter(item => selected === undefined || item.room_id === (selected === -1 ? null : selected));
    this.scene?.setBoxes(items.filter(item => item.reason !== 'invalid_bbox').map(bindingBox), true);
    const outlineKey = `${this.job?.id}/${value}`;
    if (this.metadata && outlineKey !== this.outlineKey) {
      const rooms = this.metadata.rooms.filter(room => selected === undefined || selected === -1 || room.id === selected);
      this.scene?.setRoomOutlines({ ...this.metadata, rooms });
      this.outlineKey = outlineKey;
    }
    this.onCount(items.length);
    this.list.replaceChildren(...items.map(item => {
      const row = document.createElement('li');
      const title = document.createElement('strong'); title.textContent = `#${item.instance} ${item.name}`;
      const detail = document.createElement('span');
      const volumes = item.object_volume_m3 === null ? '体积不可用'
        : `${item.intersection_volume_m3.toFixed(3)} / ${item.object_volume_m3.toFixed(3)} m³`;
      detail.textContent = `${item.room_name || '未归属'} · ${volumes} · 体积覆盖率 ${(item.overlap_ratio * 100).toFixed(1)}%${item.reason ? ` · ${reasons[item.reason] || item.reason}` : ''}`;
      row.append(title, detail); return row;
    }));
    const stats = this.bindings?.summary;
    const binding = this.job?.workflow?.binding;
    this.note.textContent = !this.metadata ? '正在加载房间轮廓…'
      : !stats ? `${this.metadata.rooms.length} 个房间；${binding?.status === 'skipped' ? '上游任务未成功，未生成归属。' : binding?.status === 'failed' ? '归属计算失败，可下载房间几何。' : binding?.status === 'interrupted' ? '归属计算已中断。' : '等待物品归属完成…'}`
      : `${this.metadata.rooms.length} 个房间 · ${stats.assigned_count} 个物品已归属 · ${stats.unassigned_count} 个未归属${stats.reason ? ` · ${reasons[stats.reason] || stats.reason}` : ''}`;
  }
}
