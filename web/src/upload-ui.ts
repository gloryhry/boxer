import { MapUpload, uploadRequest } from './upload';
import type { FileHashes } from './upload';
import type { MapInfo } from './types';

export function hashFile(file: File, chunkSize: number, signal: AbortSignal, progress: (value: number) => void): Promise<FileHashes> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./upload-hash.worker.ts', import.meta.url), { type: 'module' });
    const close = () => { worker.terminate(); signal.removeEventListener('abort', abort); };
    const abort = () => { close(); reject(new DOMException('Paused', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { abort(); return; }
    worker.onmessage = event => {
      if (event.data.result) { close(); resolve(event.data.result); }
      else if (event.data.error) { close(); reject(new Error(event.data.error)); }
      else progress(event.data.progress);
    };
    worker.onerror = event => { close(); reject(new Error(event.message || '文件校验失败')); };
    worker.postMessage({ file, chunkSize });
  });
}

export function initializeUploadPanel(host: HTMLElement, onComplete: (map: MapInfo) => Promise<void>) {
  host.innerHTML = `<details class="upload-panel"><summary>上传 ZIP 地图包</summary>
    <label for="upload-session">上传记录</label><select id="upload-session"><option value="">新建上传</option></select>
    <label for="upload-file">ZIP 文件</label><input id="upload-file" type="file" accept=".zip,application/zip">
    <p id="upload-limit" class="hint"></p><p id="upload-file-note" class="hint"></p>
    <div class="upload-actions"><button id="upload-start" type="button" class="secondary">开始上传</button><button id="upload-pause" type="button" class="secondary" disabled>暂停</button><button id="upload-cancel" type="button" class="danger" disabled>取消上传</button></div>
    <progress id="upload-progress" max="1" value="0" aria-label="地图包上传进度"></progress>
    <p id="upload-status" class="hint" role="status" aria-live="polite">选择 ZIP 文件开始上传。</p>
    <p id="upload-error" class="error" role="alert" hidden></p>
  </details>`;
  const get = <T extends HTMLElement>(id: string) => host.querySelector<T>(`#${id}`)!;
  const selector = get<HTMLSelectElement>('upload-session'), file = get<HTMLInputElement>('upload-file');
  const start = get<HTMLButtonElement>('upload-start'), pause = get<HTMLButtonElement>('upload-pause'), cancel = get<HTMLButtonElement>('upload-cancel');
  const stages: Record<string, string> = { queued: '等待处理', merging: '合并并校验 ZIP', extracting: '解压地图', validating: '校验图像与点云', publishing: '注册地图' };
  const bytes = (value: number) => value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(2)} GiB` : `${(value / 2 ** 20).toFixed(1)} MiB`;
  const upload = new MapUpload({ request: uploadRequest, hash: hashFile, storage: localStorage,
    onComplete, onChange: () => render() });

  function render() {
    const session = upload.session;
    if (!upload.file && upload.phase === 'complete') file.value = '';
    selector.replaceChildren(new Option('新建上传', ''), ...upload.sessions.map(item =>
      new Option(`${item.filename} · ${item.status === 'complete' ? '已完成' : item.status === 'cancelled' ? '已取消' : '可恢复'} · ${item.id.slice(0, 8)}`, item.id)));
    selector.value = session?.id || ''; selector.disabled = upload.running;
    file.disabled = upload.running;
    start.disabled = upload.running || (!upload.file && session?.status !== 'processing');
    start.textContent = session && session.status !== 'cancelled' && session.status !== 'complete' ? '继续 / 重试' : '开始上传';
    pause.disabled = !upload.running || !['hashing', 'uploading'].includes(upload.phase);
    cancel.disabled = !session || ['complete', 'cancelled'].includes(session.status);
    const limit = upload.limits;
    get('upload-limit').textContent = limit ? `上限 ${bytes(limit.max_zip_bytes)} · 分块 ${bytes(limit.chunk_size)} · 闲置 ${limit.ttl_seconds / 3600} 小时后清理` : '正在读取上传配置…';
    get('upload-file-note').textContent = upload.file ? `${upload.file.name} · ${bytes(upload.file.size)}`
      : session && !['complete', 'cancelled', 'processing'].includes(session.status) ? `继续上传请重新选择 ${session.filename}；会先核对文件，再补传缺失分块。` : '';
    const progress = upload.phase === 'hashing' ? upload.hashProgress : session?.progress || 0;
    get<HTMLProgressElement>('upload-progress').value = progress;
    const text: Record<string, string> = {
      idle: '选择 ZIP 文件开始上传。', hashing: `校验本地 ZIP：${(progress * 100).toFixed(1)}%`,
      uploading: `已保存 ${bytes(session?.received_bytes || 0)} / ${bytes(session?.size || 0)}（${(progress * 100).toFixed(1)}%）`,
      processing: `${stages[session?.stage || ''] || '处理地图'}：${(progress * 100).toFixed(1)}%`,
      paused: `上传已暂停，可继续。${session ? `已保存 ${session.received_chunks.length} / ${session.total_chunks} 块。` : ''}`,
      complete: '上传完成，地图已注册。', cancelled: '上传已取消，临时文件将被清理。',
    };
    get('upload-status').textContent = text[upload.phase];
    get('upload-error').textContent = upload.error || session?.error || '';
    get('upload-error').hidden = !get('upload-error').textContent;
  }
  file.onchange = () => upload.setFile(file.files?.[0] || null);
  selector.onchange = () => { file.value = ''; upload.select(selector.value); if (upload.session?.status === 'processing' || upload.session?.status === 'complete') void upload.start(); };
  start.onclick = () => void upload.start();
  pause.onclick = () => upload.pause();
  cancel.onclick = () => void upload.cancel();
  render(); void upload.initialize();
  return upload;
}
