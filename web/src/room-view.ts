import type { SceneView } from './camera';

interface RoomJob {
  id: string;
  map_id: string;
  status: 'running' | 'stopping' | 'complete' | 'failed' | 'interrupted';
  artifacts: Record<string, string>;
  error?: string | null;
}

const jobsPath = '/api/voxel-connectivity/jobs';
const artifactName = 'room_polygons.json';
const active = (job: RoomJob) => job.status === 'running' || job.status === 'stopping';

interface MapContext {
  id: string;
  abort: AbortController;
}

export function initializeRoomView(
  panel: HTMLElement,
  view: SceneView | undefined,
  api: <T>(url: string, options?: RequestInit) => Promise<T>,
  onVisibility: (visible: boolean, roomCount: number) => void,
) {
  const start = panel.querySelector<HTMLButtonElement>('#room-start')!;
  const toggle = panel.querySelector<HTMLButtonElement>('#room-toggle')!;
  const status = panel.querySelector<HTMLElement>('#room-status')!;
  let context: MapContext | undefined;
  let running: RoomJob | undefined, result: RoomJob | undefined;
  let timer: number | undefined, outlineRequest: AbortController | undefined;
  let pending = false, visible = false, roomCount = 0, renderedJob = '';

  function updateButtons() {
    start.disabled = !context || pending || !!running || !!outlineRequest;
    toggle.disabled = !view || !result || !!outlineRequest;
    toggle.setAttribute('aria-pressed', String(visible));
    toggle.textContent = visible ? '返回地图视图' : '房间框视图';
  }

  function setVisible(show: boolean) {
    visible = show;
    view?.setRoomView(show);
    onVisibility(show, roomCount);
    updateButtons();
  }

  function fail(error: unknown) {
    status.textContent = error instanceof Error ? error.message : String(error);
    status.dataset.state = 'failed';
  }

  function message(value: string) {
    status.textContent = value;
    delete status.dataset.state;
  }

  async function openResult(map: MapContext, job: RoomJob) {
    if (context !== map) return;
    if (!view) { message('房间分割已完成，当前浏览器无法显示三维视图。'); return; }
    if (renderedJob === job.id) { setVisible(true); return; }
    outlineRequest?.abort();
    const request = new AbortController();
    outlineRequest = request;
    updateButtons();
    message('正在读取房间轮廓…');
    try {
      const payload = await api<unknown>(job.artifacts[artifactName], { signal: request.signal });
      if (context !== map || request.signal.aborted) return;
      roomCount = view.setRoomOutlines(payload);
      renderedJob = job.id;
      setVisible(true);
      message(roomCount ? `${roomCount} 个房间` : '未保留房间');
    } catch (error) {
      if (context === map && !request.signal.aborted) fail(error);
    } finally {
      if (outlineRequest === request) {
        outlineRequest = undefined;
        updateButtons();
      }
    }
  }

  function poll(map: MapContext) {
    const jobId = running?.id;
    if (!jobId) return;
    timer = window.setTimeout(async () => {
      if (context !== map || running?.id !== jobId) return;
      try {
        const job = await api<RoomJob>(`${jobsPath}/${encodeURIComponent(jobId)}`, { signal: map.abort.signal });
        if (context !== map) return;
        if (active(job)) {
          running = job;
          message(job.status === 'stopping' ? '正在停止房间分割…' : '正在分割房间…');
        } else {
          running = undefined;
          if (job.status === 'complete' && job.artifacts[artifactName]) {
            result = job;
            await openResult(map, job);
          } else fail(job.error || (job.status === 'interrupted' ? '房间分割已中断' : '房间分割未生成轮廓'));
        }
      } catch (error) {
        if (context === map) fail(error);
      } finally {
        if (context === map) {
          updateButtons();
          if (running?.id === jobId) poll(map);
        }
      }
    }, 1500);
  }

  start.onclick = async () => {
    const map = context;
    if (!map || running || pending || outlineRequest) return;
    pending = true;
    updateButtons();
    message('正在提交房间分割…');
    try {
      const job = await api<RoomJob>(jobsPath, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ map_id: map.id }), signal: map.abort.signal,
      });
      if (context !== map) return;
      if (active(job)) {
        running = job;
        message('正在分割房间…');
        poll(map);
      } else fail(job.error || '房间分割未启动');
    } catch (error) {
      if (context === map) fail(error);
    } finally {
      if (context === map) { pending = false; updateButtons(); }
    }
  };

  toggle.onclick = () => {
    if (visible) setVisible(false);
    else if (context && result) void openResult(context, result);
  };

  return {
    async setMap(id: string) {
      if (context?.id === id) return;
      context?.abort.abort();
      outlineRequest?.abort();
      outlineRequest = undefined;
      window.clearTimeout(timer);
      const map = { id, abort: new AbortController() };
      context = map;
      pending = true;
      running = result = undefined;
      renderedJob = '';
      roomCount = 0;
      view?.clearRoomOutlines();
      setVisible(false);
      message('正在读取房间任务…');
      try {
        const jobs = await api<RoomJob[]>(jobsPath, { signal: map.abort.signal });
        if (context !== map) return;
        const matching = jobs.filter(job => job.map_id === id);
        result = matching.find(job => job.status === 'complete' && job.artifacts[artifactName]);
        running = matching.find(active);
        if (running) {
          message(running.status === 'stopping' ? '正在停止房间分割…' : '正在分割房间…');
          poll(map);
        } else message(result ? '已有房间框结果' : '尚无房间框结果');
      } catch (error) {
        if (context === map) fail(error);
      } finally {
        if (context === map) { pending = false; updateButtons(); }
      }
    },
  };
}
