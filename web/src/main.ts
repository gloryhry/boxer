import './style.css';
import { SceneView } from './camera';
import { initializeWorkspaceLayout } from './layout';
import { inferenceControls, shouldSyncOptions } from './inference';
import { initializeUploadPanel } from './upload-ui';
import { RoomResults } from './rooms';
import { initializeRoomView } from './room-view';
import type { InferenceState } from './inference';
import type { Frame, FrameSummary, Job, MapDirectories, MapDirectory, MapInfo, Observation, SceneInfo } from './types';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<header class="app-header">
  <a class="brand" href="/" aria-label="Boxer 首页"><span class="brand-mark">B<span>•</span></span><span><strong>BOXER <i>/ x-nav</i></strong><small>空间推理工作台</small></span></a>
  <div class="header-right"><span id="health" class="status"><span class="dot"></span>连接服务中</span><div class="header-tools"><button id="reset-layout" class="layout-reset" type="button">恢复布局</button><a href="/docs" target="_blank" rel="noopener">API 文档 ↗</a></div></div>
</header>
<p id="layout-help" class="sr-only">宽屏下可拖动图像标题栏移动窗口，拖动边缘或角落缩放。标题栏方向键移动窗口，右下角方向键调整大小，每次 10 像素，按住 Shift 每次 50 像素。双击标题或使用收起按钮可折叠图像。</p>
<main id="workspace" class="workspace">
  <section id="parameter-panel" class="panel controls" aria-label="地图与推理控制">
    <div class="section-heading"><h1>推理控制</h1><span class="eyebrow">LOCAL / LAN</span></div>
    <div id="error" role="alert" class="error" hidden></div>
    <div class="map-heading"><label for="map-path">x-nav 地图包</label><button id="refresh-maps" class="text-button" type="button" disabled>刷新列表</button></div>
    <div class="input-action"><select id="map-path" aria-describedby="map-directory-note" disabled><option value="">正在读取地图列表…</option></select><button id="load-map" class="secondary" disabled>加载地图</button></div>
    <p id="map-directory-note" class="hint" role="status" aria-live="polite">正在读取 /nav_map 下的地图包…</p>
    <p id="map-info" class="hint">选择地图后加载点云与相机信息。</p>
    <div id="map-upload"></div>
    <section id="room-panel" aria-labelledby="room-heading">
      <h2 id="room-heading">房间分割</h2>
      <div class="actions"><button id="room-start" class="secondary" type="button" disabled>分割房间</button><button id="room-toggle" class="secondary" type="button" aria-pressed="false" disabled>房间框视图</button></div>
      <p id="room-status" class="hint" role="status" aria-live="polite">加载地图后可分割房间。</p>
    </section>
    <form id="options">
      <details class="settings"><summary>推理参数 <span>词表 · 相机 · 阈值</span></summary>
        <label for="labels">词表或自定义标签</label><input id="labels" value="indoor" list="vocabularies" placeholder="indoor 或 chair,table,lamp"><datalist id="vocabularies"><option value="lvisplus"><option value="indoor"><option value="small"></datalist>
        <div class="fields"><div><label for="thresh2d">2D 阈值</label><input id="thresh2d" type="number" min="0" max="1" step="0.01" value="0.25"></div><div><label for="thresh3d">3D 阈值</label><input id="thresh3d" type="number" min="0" max="1" step="0.01" value="0.5"></div></div>
        <div class="fields"><div><label for="cameras">相机</label><input id="cameras" value="all" list="camera-list" placeholder="all 或 0,1"><datalist id="camera-list"></datalist></div><div><label for="camera-mode">图像模式</label><select id="camera-mode"><option value="native">原生相机</option><option value="virtual_pinhole">虚拟针孔</option></select></div></div>
        <label class="check"><input id="track" type="checkbox" checked>启用全局目标跟踪</label>
        <details class="advanced"><summary>高级参数</summary>
          <div class="fields three"><div><label for="start_n">起始帧（从 1 起）</label><input id="start_n" type="number" min="1" value="1"></div><div><label for="skip_n">帧步长</label><input id="skip_n" type="number" min="1" value="1"></div><div><label for="max_n">最多帧数</label><input id="max_n" type="number" min="1" value="99999"></div></div>
          <div class="fields"><div><label for="detector_hw">检测尺寸</label><input id="detector_hw" type="number" min="2" value="960"></div><div><label for="force_precision">推理精度</label><select id="force_precision"><option value="">自动</option><option value="float32">float32</option><option value="bfloat16">bfloat16</option></select></div></div>
          <div class="fields"><div><label for="xnav_execution">执行模式</label><select id="xnav_execution"><option value="auto">自动流水线</option><option value="serial">串行</option></select></div><div><label for="xnav_detector_batch_size">检测批大小</label><select id="xnav_detector_batch_size"><option value="1">1</option><option value="2">2</option></select></div></div>
          <div><label for="xnav_prepare_workers">数据准备并发</label><select id="xnav_prepare_workers"><option value="2">2</option><option value="1">1</option></select></div>
          <label for="xnav_dedup">静态地图去重</label><select id="xnav_dedup"><option value="conservative">保守去重（推荐）</option><option value="legacy">旧版策略（用于对比）</option></select>
          <div class="fields"><div><label for="track_min_hits">跟踪确认帧数</label><input id="track_min_hits" type="number" min="1" placeholder="自动"></div><div><label for="ckpt">模型检查点</label><input id="ckpt" placeholder="使用默认模型"></div></div>
          <div class="checks"><label class="check"><input id="force_cpu" type="checkbox">强制 CPU</label><label class="check"><input id="profile" type="checkbox">性能记录</label><label class="check"><input id="no_sdp" type="checkbox">关闭 SDP</label><label class="check"><input id="no_csv" type="checkbox">不写 CSV</label></div>
        </details>
      </details>
    </form>
    <div class="parameter-apply"><button id="apply-options" class="secondary" disabled>应用参数</button><p id="inference-note" class="hint" role="status" aria-live="polite">正在连接模型服务…</p></div>
    <div class="actions"><button id="start" class="primary" disabled>▶ 开始推理</button><button id="stop" class="danger" disabled>■ 停止推理</button></div>
    <p id="busy-note" class="hint" hidden>已有其他推理任务运行中，可在历史任务中查看。</p>
    <div class="job-card"><div class="job-top"><span id="job-status" class="status" data-state="idle"><span class="dot"></span>尚未开始</span><span id="progress-text" class="mono">0 / 0</span></div><progress id="progress" max="1" value="0" aria-label="Boxer 已提交帧进度"></progress><p id="workflow-stages" class="hint" role="status" aria-live="polite"></p><p id="job-detail" class="hint">加载地图后开始推理，同时运行 Boxer 和房间分割。</p><details id="output-details"><summary>输出与日志</summary><p id="output" class="path"></p><button id="refresh-log" class="text-button">刷新日志</button><pre id="log"></pre></details></div>
    <section id="room-results" class="room-results" aria-label="房间分割与物体绑定"></section>
    <details class="settings"><summary>已有结果与历史任务</summary><label for="results-path">results.json 路径</label><div class="input-action"><input id="results-path" placeholder="/path/to/results.json" spellcheck="false"><button id="import" class="secondary" disabled>加载结果</button></div><label for="history">历史任务</label><div class="input-action"><select id="history"><option value="">选择任务</option></select><button id="refresh-history" class="secondary" aria-label="刷新历史任务">刷新</button></div></details>
    <section class="playback" aria-label="结果回放"><div class="section-heading"><h2>结果回放</h2><span id="stream-state" class="hint">等待结果</span></div>
      <div class="timeline"><button id="play" class="secondary" disabled aria-label="播放回放">▶</button><input id="frame" type="range" min="0" max="0" value="0" disabled aria-label="回放帧"><span id="frame-number" class="mono">0 / 0</span></div>
      <div class="fields"><label class="check"><input id="follow-latest" type="checkbox" checked>跟随最新结果</label><label class="speed">速度 <select id="speed" aria-label="回放速度"><option value="0.2">0.2 fps</option><option value="1">1 fps</option><option value="5" selected>5 fps</option><option value="10">10 fps</option><option value="20">20 fps</option></select></label></div>
      <div class="checks"><label class="check"><input id="show-tracks" type="checkbox" checked>轨迹框</label><label class="check"><input id="tentative" type="checkbox">候选框</label><label class="check"><input id="show-map" type="checkbox" checked>地图点云</label><label class="check"><input id="show-observed" type="checkbox">当前彩色点云</label></div>
      <label for="view">相机 / 视图</label><select id="view" disabled><option>等待图像观测</option></select><p id="frame-info" class="hint mono">尚无已提交帧</p>
    </section>
  </section>
  <div class="sidebar-separator" role="separator" aria-label="调整参数栏宽度" aria-orientation="vertical" aria-controls="parameter-panel" tabindex="0"></div>
  <section class="panel viewport" aria-label="三维地图">
    <div class="viewport-toolbar"><div><h2><span id="scene-title">地图视图</span> <span class="eyebrow">3D</span></h2><p id="scene-info" class="hint">Z 轴向上 · 米</p></div><div class="view-actions"><button id="follow-camera" class="secondary" aria-pressed="false">◎ 相机中心跟随</button><button id="focus" class="secondary">定位相机</button><button id="fit" class="secondary">全图视角</button></div></div>
    <div id="canvas-host"><div id="empty-scene" class="empty-scene"><span class="empty-symbol">⌖</span><strong>从一张地图开始</strong><p>加载 x-nav 地图包，查看点云、相机轨迹与推理结果。</p></div></div>
    <div class="viewport-footer"><span id="camera-hint">左键旋转 · 右键平移 · 滚轮缩放</span><span id="room-camera-hint" hidden>左键旋转 · 右键平移 · 滚轮缩放</span><span id="object-count" class="mono">0 个目标</span><span id="room-count" class="mono" hidden></span></div>
  </section>
  <section class="image-stack" aria-label="相机图像">
    <figure id="window-2d" class="panel image-panel" data-floating-window data-window-name="2D 检测" aria-label="2D 检测图像窗口"><figcaption data-window-drag tabindex="0" aria-label="移动 2D 检测窗口" aria-describedby="layout-help"><span class="window-name"><i class="image-dot orange"></i>2D 检测</span><span id="image-view-2d" class="hint">CAMERA</span><button type="button" data-window-collapse aria-expanded="true" aria-controls="image-content-2d" aria-label="收起 2D 检测">▴</button></figcaption><div id="image-content-2d" class="image-well"><img id="image-2d" alt="当前视图的二维检测框" draggable="false" hidden><span class="image-empty">等待图像观测</span></div></figure>
    <figure id="window-3d" class="panel image-panel" data-floating-window data-window-name="3D 投影" aria-label="3D 投影图像窗口"><figcaption data-window-drag tabindex="0" aria-label="移动 3D 投影窗口" aria-describedby="layout-help"><span class="window-name"><i class="image-dot teal"></i><span id="image-title-3d">3D 轨迹投影</span></span><span id="image-view-3d" class="hint">CAMERA</span><button type="button" data-window-collapse aria-expanded="true" aria-controls="image-content-3d" aria-label="收起 3D 投影">▴</button></figcaption><div id="image-content-3d" class="image-well"><img id="image-3d" alt="当前视图的三维目标投影" draggable="false" hidden><span class="image-empty">等待图像观测</span></div></figure>
  </section>
</main>`;

function element<T extends HTMLElement = HTMLElement>(id: string) { return document.getElementById(id)! as T; }
function input(id: string) { return element<HTMLInputElement>(id); }
function checked(id: string) { return input(id).checked; }
initializeWorkspaceLayout(element('workspace'), element('reset-layout'));
const labels: Record<string, string> = { starting: '准备任务', running: '运行中', stopping: '正在停止', complete: '已完成', failed: '失败', interrupted: '已中断', pending: '等待中', skipped: '已跳过' };
const active = (job: Job | null) => job && ['starting', 'running', 'stopping'].includes(job.workflow?.status || job.status);
let scene: SceneView | undefined;
let map: MapInfo | null = null, job: Job | null = null, frame: Frame | null = null;
let summaries: FrameSummary[] = [], jobs: Job[] = [];
let index = 0, viewKey = '', busy = false, playing = false, loadingFrame = false;
let activeJobId: string | null = null, mapGeneration = 0;
let inference: InferenceState | null = null, connected = false, draftEdited = false;
let serviceGeneration = 0;
let events: EventSource | undefined, frameRequest: AbortController | undefined;
let imageUrls: string[] = [];
let mapDirectories: MapDirectory[] = [], loadingDirectories = false;
let preferredMapPath = localStorage.getItem('boxer-map-path') || '';
try { scene = new SceneView(element('canvas-host')); }
catch { showError('浏览器无法初始化 WebGL。请启用硬件加速或使用支持 WebGL 的浏览器；接口和图像回放仍可使用。'); }
const roomResults = new RoomResults(element('room-results'), scene, fetchOk, () => {
  if (roomResults.active) { setPlaying(false); scene?.setFollow(false); syncFollow(); }
  else if (job?.processed_frames) void selectFrame(index, false);
  buttons();
}, showError, count => { element('object-count').textContent = `${count} 个最终目标`; });

function showError(message: string) { element('error').textContent = message; element('error').hidden = false; }
function clearError() { element('error').hidden = true; }
async function fetchOk(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof data.detail === 'string' ? data.detail : data.detail.map((item: { msg: string }) => item.msg).join('；'));
  }
  return response;
}
async function api<T>(url: string, options?: RequestInit): Promise<T> { return (await fetchOk(url, options)).json(); }
function post<T>(url: string, data: unknown) { return api<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }); }
const roomView = initializeRoomView(element('room-panel'), scene, api, (visible, count) => {
  element('workspace').toggleAttribute('data-room-view', visible);
  element('scene-title').textContent = visible ? '房间框' : '地图视图';
  element('fit').textContent = visible ? '查看全部房间框' : '全图视角';
  for (const id of ['scene-info', 'object-count', 'focus', 'follow-camera', 'camera-hint']) element(id).hidden = visible;
  element('room-camera-hint').hidden = !visible;
  const roomCount = element('room-count');
  roomCount.hidden = !visible;
  roomCount.textContent = `${count} 个房间`;
  syncFollow();
});
function buttons() {
  const state = inferenceControls(inference, options(), connected, activeJobId);
  element<HTMLButtonElement>('start').disabled = busy || !map || !state.canStart;
  element<HTMLButtonElement>('apply-options').disabled = busy || !state.canApply;
  element('inference-note').textContent = state.message;
  element<HTMLButtonElement>('stop').disabled = busy || !active(job) || !!job?.readonly || (job?.workflow?.status || job?.status) === 'stopping';
  const selectedMapPath = element<HTMLSelectElement>('map-path').value;
  element<HTMLButtonElement>('load-map').disabled = busy || loadingDirectories || !mapDirectories.some(item => item.path === selectedMapPath);
  element<HTMLSelectElement>('map-path').disabled = busy || loadingDirectories || !mapDirectories.length;
  element<HTMLButtonElement>('refresh-maps').disabled = busy || loadingDirectories;
  element<HTMLButtonElement>('import').disabled = busy || !map;
  element<HTMLButtonElement>('play').disabled = !job?.processed_frames || roomResults.active;
  input('frame').disabled = !job?.processed_frames || roomResults.active;
  element<HTMLButtonElement>('focus').disabled = roomResults.active || !scene?.follow.pose;
  element<HTMLButtonElement>('follow-camera').disabled = roomResults.active;
  element('busy-note').hidden = !activeJobId || job?.id === activeJobId;
}
async function action(work: () => Promise<void>) {
  if (busy) return;
  busy = true; clearError(); buttons();
  try { await work(); }
  catch (error) { showError(error instanceof Error ? error.message : String(error)); }
  finally { busy = false; buttons(); }
}
function clearImages() {
  imageUrls.forEach(URL.revokeObjectURL); imageUrls = [];
  for (const kind of ['2d', '3d']) {
    const image = element<HTMLImageElement>(`image-${kind}`);
    image.hidden = true; image.removeAttribute('src');
    image.nextElementSibling!.removeAttribute('hidden');
  }
}
function setPlaying(value: boolean) {
  playing = value;
  element('play').textContent = value ? 'Ⅱ' : '▶';
  element('play').setAttribute('aria-label', value ? '暂停回放' : '播放回放');
}
function syncFollow() { element('follow-camera').setAttribute('aria-pressed', String(!!scene?.follow.enabled)); }
function detachJob() {
  events?.close(); frameRequest?.abort(); setPlaying(false);
  frame = null; job = null; summaries = []; index = 0; viewKey = '';
  roomResults.update(null);
  clearImages(); scene?.clearFrame(); scene?.setTrajectory([]);
  element('frame-number').textContent = '0 / 0'; input('frame').value = '0'; input('frame').max = '0';
  element('object-count').textContent = '0 个目标';
  element('frame-info').textContent = '尚无已提交帧';
  element('view').replaceChildren(new Option('等待图像观测', ''));
  element<HTMLSelectElement>('view').disabled = true;
  element('job-status').innerHTML = '<span class="dot"></span>尚未开始';
  element('job-status').dataset.state = 'idle';
  element('workflow-stages').textContent = '';
  element('progress-text').textContent = '0 / 0';
  element<HTMLProgressElement>('progress').value = 0;
  element('job-detail').textContent = '地图已加载，开始推理后同时运行 Boxer 和房间分割。';
  element('output').textContent = ''; element('log').textContent = '';
  element('stream-state').textContent = '等待结果';
  buttons();
}

async function activateMap(info: MapInfo) {
  const generation = ++mapGeneration;
  const [metadata, cloud] = await Promise.all([
    api<SceneInfo>(`/api/maps/${info.id}/scene`),
    fetchOk(`/api/maps/${info.id}/cloud`).then(response => response.arrayBuffer()),
  ]);
  if (generation !== mapGeneration) return;
  map = info;
  const selector = element<HTMLSelectElement>('map-path');
  selector.value = mapDirectories.some(item => item.path === info.path) ? info.path : '';
  if (loadingDirectories) preferredMapPath = info.path;
  else rememberMapPath(selector.value);
  element('map-info').textContent = `当前地图：${info.path}\n${info.validation.frames} 个${info.validation.frame_unit === 'image' ? '图像组' : '关键帧'} · ${info.validation.keyframes} 个 LiDAR 锚点 · ${info.cameras.length} 个相机`;
  element('scene-info').textContent = `${metadata.point_count.toLocaleString()} 点 · Z ↑ · 米`;
  scene?.setMap(metadata, cloud); scene?.showMap(checked('show-map'));
  void roomView.setMap(info.id);
  element('empty-scene').hidden = true; syncFollow();
  const cameraList = element<HTMLDataListElement>('camera-list');
  cameraList.replaceChildren(new Option('全部相机', 'all'), ...info.cameras.map(camera => new Option(`cam${camera.id} · ${camera.model}`, String(camera.id))));
  buttons();
}

function options() {
  const result: Record<string, unknown> = {
    labels: input('labels').value.split(',').map(label => label.trim()).filter(Boolean).join(','), xnav_cameras: input('cameras').value.trim(),
    xnav_camera_mode: input('camera-mode').value, force_precision: input('force_precision').value || null,
    xnav_execution: input('xnav_execution').value, ckpt: input('ckpt').value.trim() || null,
    xnav_dedup: input('xnav_dedup').value,
    track_min_hits: input('track_min_hits').value ? input('track_min_hits').valueAsNumber : null,
  };
  for (const key of ['thresh2d', 'thresh3d', 'start_n', 'skip_n', 'max_n', 'detector_hw']) result[key] = input(key).valueAsNumber;
  result.xnav_detector_batch_size = Number(input('xnav_detector_batch_size').value);
  result.xnav_prepare_workers = Number(input('xnav_prepare_workers').value);
  for (const key of ['track', 'force_cpu', 'no_sdp', 'no_csv', 'profile']) result[key] = checked(key);
  return result;
}

function setInference(value: InferenceState) {
  const draft = options();
  if ((!inference && !draftEdited) || shouldSyncOptions(inference, value, draft)) {
    for (const [key, item] of Object.entries(value.options)) {
      const id = ({ xnav_cameras: 'cameras', xnav_camera_mode: 'camera-mode' } as Record<string, string>)[key] || key;
      const field = input(id);
      if (typeof item === 'boolean') field.checked = item;
      else field.value = item === null ? '' : String(item);
    }
  }
  inference = value;
  buttons();
}

function showJob(value: Job) {
  job = value;
  const overallStatus = value.workflow?.status || value.status;
  const status = element('job-status');
  status.replaceChildren(Object.assign(document.createElement('span'), { className: 'dot' }), document.createTextNode(`${value.readonly ? '回放 · ' : ''}${labels[overallStatus] || overallStatus}`));
  status.dataset.state = overallStatus;
  element('workflow-stages').textContent = value.workflow ? `Boxer：${labels[value.status]} · 房间分割：${labels[value.workflow.room_segmentation.status]} · 物体绑定：${labels[value.workflow.binding.status]}${value.workflow.map_export ? ` · 地图保存：${labels[value.workflow.map_export.status]}` : ''}` : '';
  roomResults.update(value);
  element('progress-text').textContent = `${value.processed_frames} / ${value.total_frames}`;
  const progress = element<HTMLProgressElement>('progress');
  progress.max = Math.max(1, value.total_frames); progress.value = value.processed_frames;
  element('job-detail').textContent = `${value.device} · ${value.elapsed_seconds.toFixed(1)} 秒 · ${value.options.xnav_dedup === 'conservative' ? '保守去重' : '旧版去重'}${value.readonly ? ' · 只读结果' : ''}`;
  element('output').textContent = `任务输出：${value.output_dir}${!value.readonly && !value.options.no_csv && value.status === 'complete' && map ? `\n地图 CSV：${map.path}/3dbbox` : ''}${value.workflow?.map_export && map ? `\n房间结果位置：${map.path}/${value.workflow.map_export.relative_path}` : ''}`;
  input('frame').max = String(Math.max(0, value.processed_frames - 1));
  element('frame-number').textContent = `${value.processed_frames ? index + 1 : 0} / ${value.processed_frames}`;
  input('show-tracks').disabled = !value.tracking;
  if (!value.tracking) input('show-tracks').checked = false;
  if (value.error) showError(value.error);
  if (value.workflow?.error) showError(value.workflow.error);
  if (!value.readonly) activeJobId = active(value) ? value.id : activeJobId === value.id ? null : activeJobId;
  buttons();
}

let syncing: Promise<void> = Promise.resolve();
async function syncFrames(identity: string) {
  if (job?.id !== identity || !job.processed_frames) return;
  while (job?.id === identity && summaries.length < job.processed_frames) {
    const page = await api<{ frames: FrameSummary[] }>(`/api/jobs/${identity}/frames?offset=${summaries.length}&limit=1000`);
    if (job?.id !== identity) return;
    if (!page.frames.length) break;
    summaries.push(...page.frames);
  }
  if (job?.id !== identity) return;
  scene?.setTrajectory(summaries.map(summary => summary.position));
  if (checked('follow-latest')) await selectFrame(job.processed_frames - 1, false);
  else if (!frame) await selectFrame(0, true);
}

async function selectJob(value: Job) {
  detachJob();
  await activateMap(await api<MapInfo>(`/api/maps/${value.map_id}`));
  showJob(value); input('history').value = value.id; input('show-tracks').checked = value.tracking;
  input('results-path').value = value.results_path;
  localStorage.setItem('boxer-job-id', value.id);
  input('follow-latest').checked = !value.readonly && !!active(value);
  // Finish the initial load before subscribing, so the first SSE snapshot
  // cannot append the same page concurrently with initial replay loading.
  await syncFrames(value.id);
  events = new EventSource(`/api/jobs/${value.id}/events`);
  events.addEventListener('job', message => {
    if (job?.id !== value.id) return;
    const previousCount = job.processed_frames;
    const update: Job = JSON.parse((message as MessageEvent).data);
    showJob(update);
    if (previousCount !== update.processed_frames || !frame) {
      syncing = syncing.then(() => syncFrames(value.id)).catch(error => showError(String(error)));
    }
  });
  events.onopen = () => { element('stream-state').textContent = '已连接'; };
  events.onerror = () => { element('stream-state').textContent = '连接恢复中…'; };
}

function observationFor(value: Frame): Observation | undefined {
  return value.observations.find(o => `${o.camera_id}/${o.view_id}` === viewKey) || value.observations[0];
}

async function selectFrame(next: number, snap: boolean) {
  if (!job || !job.processed_frames) return;
  frameRequest?.abort();
  const request = new AbortController(); frameRequest = request; loadingFrame = true;
  const identity = job.id;
  index = Math.max(0, Math.min(next, job.processed_frames - 1));
  input('frame').value = String(index);
  element('frame-number').textContent = `${index + 1} / ${job.processed_frames}`;
  element('frame-info').textContent = '正在加载所选帧…';
  try {
    const value = await api<Frame>(`/api/jobs/${identity}/frames/${index}`, { signal: request.signal });
    const observation = observationFor(value);
    const query = new URLSearchParams({ tracked: String(checked('show-tracks')), tentative: String(checked('tentative')) });
    if (observation) { query.set('camera_id', String(observation.camera_id)); query.set('view_id', observation.view_id); }
    const base = `/api/jobs/${identity}/frames/${value.index}`;
    const [top, bottom, cloud] = await Promise.all([
      observation ? fetchOk(`${base}/images/2d?${query}`, { signal: request.signal }).then(r => r.blob()) : undefined,
      observation ? fetchOk(`${base}/images/3d?${query}`, { signal: request.signal }).then(r => r.blob()) : undefined,
      checked('show-observed') ? fetchOk(`${base}/cloud`, { signal: request.signal }).then(r => r.arrayBuffer()) : undefined,
    ]);
    if (request.signal.aborted || job?.id !== identity) return;
    frame = value;
    clearImages();
    [top, bottom].forEach((blob, position) => {
      if (!blob) return;
      const image = element<HTMLImageElement>(`image-${position === 0 ? '2d' : '3d'}`);
      const url = URL.createObjectURL(blob); imageUrls.push(url);
      image.src = url; image.hidden = false; image.nextElementSibling!.setAttribute('hidden', '');
    });
    const tracked = checked('show-tracks');
    const boxes = (tracked ? value.tracks : value.detections).filter(box => !tracked || checked('tentative') || box.state !== 'TENTATIVE');
    if (!roomResults.active) scene?.setBoxes(boxes, tracked);
    scene?.setObserved(cloud); scene?.setObservation(observation, snap && !roomResults.active);
    if (!roomResults.active) element('object-count').textContent = `${boxes.length} 个目标`;
    element('image-title-3d').textContent = tracked ? '3D 轨迹投影' : '3D 检测投影';
    const views = element<HTMLSelectElement>('view');
    views.replaceChildren(...value.observations.map(o => new Option(`cam${o.camera_id} / ${o.view_id}`, `${o.camera_id}/${o.view_id}`)));
    views.disabled = !observation;
    viewKey = observation ? `${observation.camera_id}/${observation.view_id}` : '';
    if (viewKey) views.value = viewKey;
    else views.replaceChildren(new Option('此帧没有图像观测', ''));
    for (const kind of ['2d', '3d']) element(`image-view-${kind}`).textContent = observation ? `cam${observation.camera_id} / ${observation.view_id}` : '无图像';
    element('frame-info').textContent = `${value.image_frame_id === null ? '关键帧' : '图像组'} ${value.frame_id} · LiDAR ${value.lidar_keyframe_id}\n${value.time_ns} ns`;
    element('camera-hint').textContent = scene?.follow.pose ? '左键旋转 · 右键平移 · 滚轮缩放' : '当前帧无有效相机位姿，保持观察位置';
    buttons();
  } catch (error) {
    if (!request.signal.aborted) showError(error instanceof Error ? error.message : String(error));
  } finally { if (frameRequest === request) loadingFrame = false; }
}

function rememberMapPath(path: string) {
  preferredMapPath = path;
  if (path) localStorage.setItem('boxer-map-path', path);
  else localStorage.removeItem('boxer-map-path');
}
async function refreshMapDirectories() {
  if (loadingDirectories) return;
  loadingDirectories = true; buttons();
  const selector = element<HTMLSelectElement>('map-path');
  const note = element('map-directory-note');
  note.textContent = '正在读取 /nav_map 下的地图包…';
  delete note.dataset.state;
  try {
    const value = await api<MapDirectories>('/api/maps/directories');
    mapDirectories = value.directories;
    const placeholder = new Option(mapDirectories.length ? '请选择地图' : '未找到地图包', '');
    placeholder.disabled = true;
    selector.replaceChildren(placeholder, ...mapDirectories.map(item => new Option(item.name, item.path)));
    selector.value = mapDirectories.some(item => item.path === preferredMapPath) ? preferredMapPath : '';
    rememberMapPath(selector.value);
    note.textContent = mapDirectories.length ? `${value.root} 下共 ${mapDirectories.length} 个地图包（包含子文件夹）。` : `${value.root} 下未找到 x-nav 地图包，请放入地图后刷新列表。`;
  } catch (error) {
    mapDirectories = [];
    selector.replaceChildren(new Option('无法读取地图列表', ''));
    note.dataset.state = 'failed';
    note.textContent = `无法读取 /nav_map 地图列表：${error instanceof Error ? error.message : String(error)}。请检查目录和读取权限后刷新列表。`;
  } finally { loadingDirectories = false; buttons(); }
}

async function refreshHistory() {
  jobs = await api<Job[]>('/api/jobs');
  const history = element<HTMLSelectElement>('history');
  history.replaceChildren(new Option('选择任务', ''), ...jobs.map(item => new Option(`${item.readonly ? '回放' : labels[item.workflow?.status || item.status]} · ${new Date(item.created_at).toLocaleString()} · ${item.processed_frames} 帧`, item.id)));
  if (job) history.value = job.id;
}
async function health() {
  const generation = serviceGeneration;
  try {
    const [value, models] = await Promise.all([
      api<{ cuda_available: boolean; device: string; active_job_id: string | null; active_workflow_id?: string | null }>('/api/health'),
      api<InferenceState>('/api/inference'),
    ]);
    if (generation !== serviceGeneration) return;
    activeJobId = value.active_workflow_id || value.active_job_id;
    connected = true; setInference(models);
    const device = models.status === 'ready' ? models.device : value.device;
    const state = models.status === 'loading' ? '模型加载中' : models.status === 'failed' ? '模型加载失败' : activeJobId ? '任务运行中' : '模型就绪';
    element('health').replaceChildren(Object.assign(document.createElement('span'), { className: 'dot' }), document.createTextNode(`${device} / ${state}`));
  } catch (error) {
    if (generation !== serviceGeneration) return;
    connected = false; element('health').textContent = '服务连接中断'; buttons(); throw error;
  }
}

element('load-map').onclick = () => void action(async () => {
  const path = element<HTMLSelectElement>('map-path').value;
  if (loadingDirectories || !mapDirectories.some(item => item.path === path)) return;
  const info = await post<MapInfo>('/api/maps', { path });
  detachJob(); localStorage.removeItem('boxer-job-id'); await activateMap(info);
});
element('map-path').onchange = () => { rememberMapPath(element<HTMLSelectElement>('map-path').value); buttons(); };
element('refresh-maps').onclick = () => void refreshMapDirectories();
element('start').onclick = () => void action(async () => {
  if (!element<HTMLFormElement>('options').reportValidity() || !map) return;
  if (!inferenceControls(inference, options(), connected, activeJobId).canStart) return;
  ++serviceGeneration;
  const value = await post<Job>('/api/jobs', { map_id: map.id, options: options() });
  activeJobId = value.id; await selectJob(value); await refreshHistory();
});
element('apply-options').onclick = () => void action(async () => {
  if (!element<HTMLFormElement>('options').reportValidity()) return;
  ++serviceGeneration;
  setInference(await api<InferenceState>('/api/inference', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options()),
  }));
});
element('stop').onclick = () => void action(async () => { if (job) { ++serviceGeneration; showJob(await post<Job>(`/api/jobs/${job.id}/stop`, {})); } });
element('import').onclick = () => void action(async () => {
  if (!map) return;
  await selectJob(await post<Job>('/api/jobs/import', { map_id: map.id, results_path: input('results-path').value.trim() }));
  await refreshHistory();
});
element('history').onchange = () => void action(async () => {
  const value = jobs.find(item => item.id === input('history').value); if (value) await selectJob(await api<Job>(`/api/jobs/${value.id}`));
});
element('refresh-history').onclick = () => void action(refreshHistory);
element('refresh-log').onclick = () => void action(async () => { if (job) element('log').textContent = (await api<{ text: string }>(`/api/jobs/${job.id}/log`)).text || '暂无日志'; });
input('frame').oninput = () => { input('follow-latest').checked = false; setPlaying(false); void selectFrame(Number(input('frame').value), true); };
element('play').onclick = () => { input('follow-latest').checked = false; if (!playing && job && index + 1 >= job.processed_frames) void selectFrame(0, true); setPlaying(!playing); };
input('follow-latest').onchange = () => { setPlaying(false); if (job && checked('follow-latest')) void selectFrame(job.processed_frames - 1, true); };
element('view').onchange = () => { viewKey = input('view').value; void selectFrame(index, true); };
for (const key of ['show-tracks', 'tentative', 'show-observed']) input(key).onchange = () => { void selectFrame(index, true); };
input('show-map').onchange = () => scene?.showMap(checked('show-map'));
element('follow-camera').onclick = () => { scene?.setFollow(!scene.follow.enabled); syncFollow(); };
element('focus').onclick = () => scene?.focus();
element('fit').onclick = () => { scene?.fit(); syncFollow(); };
element('options').onsubmit = event => event.preventDefault();
element('options').addEventListener('input', () => { draftEdited = true; buttons(); });
element('options').addEventListener('change', () => { draftEdited = true; buttons(); });

let lastStep = performance.now();
setInterval(() => {
  if (playing && !loadingFrame && job && performance.now() - lastStep > 1000 / Number(input('speed').value)) {
    lastStep = performance.now();
    if (index + 1 < job.processed_frames) void selectFrame(index + 1, false);
    else setPlaying(false);
  }
}, 40);
setInterval(() => { void health().catch(() => {}); }, 1500);
void refreshMapDirectories();
void action(async () => {
  await Promise.all([health(), refreshHistory()]);
  const saved = localStorage.getItem('boxer-job-id');
  const value = jobs.find(item => item.id === saved) || jobs.find(item => item.id === activeJobId);
  if (value) await selectJob(value);
});
initializeUploadPanel(element('map-upload'), async info => {
  while (busy) await new Promise(resolve => setTimeout(resolve, 50));
  await action(async () => {
    rememberMapPath(info.path);
    await refreshMapDirectories();
    detachJob(); localStorage.removeItem('boxer-job-id');
    await activateMap(info);
  });
});
