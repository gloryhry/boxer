import type { SceneView } from './camera';
import type { Box, Job, RoomBinding, RoomBindingPage, RoomResult } from './types';

const reasons: Record<string, string> = {
  invalid_bbox: '包围框无效', no_rooms: '未分割出房间', no_points: '框内没有点云',
  no_labeled_points: '框内点云没有房间标签', insufficient_support: '支持点不足 10 个',
  ambiguous_room: '房间票数占比低于 60%', tracking_disabled: '本次未开启跟踪',
  no_tracked_objects: '没有最终确认物体',
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
  private cloudGeneration = 0;
  private requests = new Set<AbortController>();
  private cloudRequest?: AbortController;
  private loadingRooms = false;
  private loadingBindings = false;
  private cloudKey = '';
  private readonly mode: HTMLSelectElement;
  private readonly filter: HTMLSelectElement;
  private readonly content: HTMLElement;
  private readonly note: HTMLElement;
  private readonly list: HTMLElement;
  private readonly downloads: HTMLElement;

  constructor(host: HTMLElement, private readonly scene: SceneView | undefined,
    private readonly fetchOk: (url: string, options?: RequestInit) => Promise<Response>,
    private readonly onModeChange: () => void, private readonly onError: (error: string) => void,
    private readonly onCount: (count: number) => void) {
    host.innerHTML = `<label for="result-mode">结果视图</label><select id="result-mode"><option value="frames">逐帧回放</option><option value="rooms" disabled>房间结果</option></select>
      <div id="room-content" hidden><label for="room-filter">房间 / 物体归属</label><select id="room-filter"><option value="">全部房间</option></select><p id="room-note" class="hint" role="status" aria-live="polite"></p><div id="room-downloads" class="room-downloads"></div><ul id="room-objects" class="room-objects" aria-label="最终跟踪物体"></ul></div>`;
    this.mode = host.querySelector<HTMLSelectElement>('#result-mode')!;
    this.filter = host.querySelector<HTMLSelectElement>('#room-filter')!;
    this.content = host.querySelector('#room-content')!;
    this.note = host.querySelector('#room-note')!;
    this.list = host.querySelector('#room-objects')!;
    this.downloads = host.querySelector('#room-downloads')!;
    this.mode.onchange = () => {
      this.cloudKey = '';
      ++this.cloudGeneration;
      this.cloudRequest?.abort();
      this.content.hidden = !this.active;
      this.scene?.setRoomsMode(this.active);
      this.onModeChange();
      this.paint();
    };
    this.filter.onchange = () => this.paint();
  }

  get active() { return this.mode.value === 'rooms'; }

  update(job: Job | null) {
    if (job?.id !== this.job?.id) {
      ++this.generation; ++this.cloudGeneration;
      this.requests.forEach(request => request.abort()); this.requests.clear();
      this.metadata = undefined; this.bindings = undefined;
      this.loadingRooms = this.loadingBindings = false;
      this.cloudKey = ''; this.mode.value = 'frames'; this.content.hidden = true;
      this.filter.replaceChildren(new Option('全部房间', ''));
      this.list.replaceChildren(); this.downloads.replaceChildren();
      this.scene?.setRoomsMode(false); this.scene?.setRoomCloud();
    }
    this.job = job;
    if (this.metadata) this.renderDownloads();
    const roomReady = job?.workflow?.room_segmentation.status === 'complete';
    this.mode.options[1].disabled = !roomReady;
    if (roomReady && !this.metadata && !this.loadingRooms) void this.loadRooms();
    if (job?.workflow?.binding.status === 'complete' && !this.bindings && !this.loadingBindings) void this.loadBindings();
    this.paint();
  }

  private async loadRooms() {
    const identity = this.job!.id, generation = this.generation;
    const request = new AbortController(); this.requests.add(request); this.loadingRooms = true;
    try {
      const result: RoomResult = await (await this.fetchOk(`/api/jobs/${identity}/rooms`, { signal: request.signal })).json();
      if (generation !== this.generation) return;
      this.metadata = result;
      this.filter.replaceChildren(new Option('全部房间', ''), new Option('未绑定物体', '-1'),
        ...result.rooms.map(room => new Option(room.name, String(room.id))));
      this.renderDownloads(); this.paint();
    } catch (error) {
      if (!request.signal.aborted) this.onError(`加载房间结果失败：${String(error)}`);
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
        if (generation !== this.generation) return;
        items.push(...result.items);
        if (!result.items.length) break;
      } while (items.length < result.total);
      this.bindings = { ...result, items };
      this.renderDownloads(); this.paint();
    } catch (error) {
      if (!request.signal.aborted) this.onError(`加载物体绑定失败：${String(error)}`);
    } finally {
      this.requests.delete(request);
      if (generation === this.generation) this.loadingBindings = false;
    }
  }

  private renderDownloads() {
    this.downloads.replaceChildren();
    const artifacts = { ...this.metadata?.artifacts, ...this.job?.workflow?.binding.artifacts };
    if (this.job?.workflow?.map_export && this.job.workflow.status === 'complete' && !this.job.options.no_csv) artifacts['3dbbox.zip'] = `/api/jobs/${this.job.id}/3dbbox.zip`;
    const names: Record<string, string> = { 'result_colored.ply': '房间点云 PLY', 'rooms.json': '房间 JSON', 'summary.json': '分割统计', 'room-bindings.json': '物体绑定 JSON', '3dbbox.zip': '完整结果 ZIP' };
    for (const [name, url] of Object.entries(artifacts)) {
      const link = document.createElement('a'); link.href = url;
      link.textContent = names[name] || name; link.download = name;
      this.downloads.append(link);
    }
  }

  private paint() {
    if (!this.active) return;
    const value = this.filter.value;
    const selected = value === '' ? undefined : Number(value);
    const items = (this.bindings?.items || []).filter(item => selected === undefined || item.room_id === (selected === -1 ? null : selected));
    this.scene?.setBoxes(items.map(bindingBox), true); this.onCount(items.length);
    this.list.replaceChildren(...items.map(item => {
      const row = document.createElement('li');
      const title = document.createElement('strong'); title.textContent = `#${item.instance} ${item.name}`;
      const detail = document.createElement('span');
      detail.textContent = `${item.room_name || '未绑定'} · ${item.support_points}/${item.point_count} 点 · ${(item.vote_ratio * 100).toFixed(1)}%${item.reason ? ` · ${reasons[item.reason] || item.reason}` : ''}`;
      row.append(title, detail); return row;
    }));
    const stats = this.bindings?.summary;
    this.note.textContent = !this.metadata ? '正在加载房间结果…'
      : !stats ? `${this.metadata.rooms.length} 个房间；${this.job?.workflow?.binding.status === 'skipped' ? '上游任务未成功，未生成绑定。' : this.job?.workflow?.binding.status === 'failed' ? '绑定失败，可下载分割结果。' : this.job?.workflow?.binding.status === 'interrupted' ? '绑定已中断。' : '等待物体绑定完成…'}`
      : `${this.metadata.rooms.length} 个房间 · ${stats.assigned_count} 个物体已绑定 · ${stats.unassigned_count} 个未绑定${stats.reason ? ` · ${reasons[stats.reason] || stats.reason}` : ''}`;
    if (this.metadata) void this.loadCloud(selected !== undefined && selected >= 0 ? selected : undefined);
  }

  private async loadCloud(roomId?: number) {
    const identity = this.job!.id;
    const key = `${identity}/${roomId ?? 'all'}`;
    if (key === this.cloudKey) return;
    this.cloudKey = key; this.cloudRequest?.abort();
    const request = new AbortController(); this.cloudRequest = request; this.requests.add(request);
    const generation = ++this.cloudGeneration;
    this.scene?.setRoomCloud();
    try {
      const cloud = await (await this.fetchOk(`/api/jobs/${identity}/rooms/cloud${roomId === undefined ? '' : `?room_id=${roomId}`}`, { signal: request.signal })).arrayBuffer();
      if (generation !== this.cloudGeneration || !this.active) return;
      this.scene?.setRoomCloud(cloud);
    } catch (error) {
      if (!request.signal.aborted) { this.cloudKey = ''; this.onError(`加载房间点云失败：${String(error)}`); }
    } finally { this.requests.delete(request); }
  }
}
