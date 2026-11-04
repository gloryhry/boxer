import type { MapInfo } from './types';

export interface UploadLimits {
  chunk_size: number; max_zip_bytes: number; max_expanded_bytes: number;
  max_files: number; ttl_seconds: number;
}
export interface UploadInfo {
  id: string; filename: string; size: number; sha256: string; chunk_size: number;
  total_chunks: number; received_chunks: number[]; received_bytes: number;
  status: 'uploading' | 'processing' | 'complete' | 'failed' | 'cancelled';
  stage: string; progress: number; created_at: string; expires_at: string;
  error: string | null; error_code: number | null; map: MapInfo | null;
}
export interface UploadList { limits: UploadLimits; uploads: UploadInfo[] }
export interface FileHashes { sha256: string; chunks: string[] }
export type UploadPhase = 'idle' | 'hashing' | 'uploading' | 'paused' | 'processing' | 'complete' | 'cancelled';
export class UploadError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('Paused', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, ms);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}

export async function retryUpload<T>(work: () => Promise<T>, signal: AbortSignal, wait = sleep): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    signal.throwIfAborted();
    try { const value = await work(); signal.throwIfAborted(); return value; }
    catch (error) {
      const temporary = error instanceof UploadError ? error.status >= 500 || error.status === 429
        : error instanceof TypeError;
      if (signal.aborted || !temporary || attempt === 3) throw error;
      await wait(1000 * 2 ** attempt, signal);
    }
  }
}

export async function uploadRequest<T>(path: string, options: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const timeout = new AbortController();
  const abort = () => timeout.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) timeout.abort();
  const timer = setTimeout(() => timeout.abort(), 30000);
  try {
    const response = await fetch(`/api/maps/uploads${path}`, { ...options, signal: timeout.signal });
    const value = await response.json().catch(() => ({ detail: response.statusText || '请求失败' }));
    if (!response.ok) {
      const detail = typeof value.detail === 'string' ? value.detail
        : Array.isArray(value.detail) ? value.detail.map((item: { msg: string }) => item.msg).join('；') : response.statusText;
      throw new UploadError(response.status, detail);
    }
    return value as T;
  } catch (error) {
    if (timeout.signal.aborted && !signal?.aborted) throw new UploadError(503, '上传请求超时');
    throw error;
  } finally {
    clearTimeout(timer); signal?.removeEventListener('abort', abort);
  }
}

interface Dependencies {
  request: typeof uploadRequest;
  hash: (file: File, chunkSize: number, signal: AbortSignal, progress: (value: number) => void) => Promise<FileHashes>;
  storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
  onChange: () => void;
  onComplete: (map: MapInfo) => Promise<void>;
  wait?: typeof sleep;
}

const STORAGE = 'boxer-map-upload';

export class MapUpload {
  limits: UploadLimits | null = null;
  sessions: UploadInfo[] = [];
  session: UploadInfo | null = null;
  file: File | null = null;
  phase: UploadPhase = 'idle';
  hashProgress = 0;
  error = '';
  running = false;
  private controller: AbortController | null = null;
  private activated = '';

  constructor(private deps: Dependencies) {}

  private notify() { this.deps.onChange(); }
  private processing() { return this.session?.status === 'processing'; }
  private accept(value: UploadInfo) {
    this.session = value;
    this.sessions = [value, ...this.sessions.filter(item => item.id !== value.id)];
    this.deps.storage.setItem(STORAGE, JSON.stringify({ id: value.id, filename: value.filename,
      size: value.size, sha256: value.sha256, activated: this.activated }));
    this.notify();
  }

  async initialize() {
    try {
      const value = await this.deps.request<UploadList>('');
      this.limits = value.limits; this.sessions = value.uploads;
      let saved: { id?: string; activated?: string } = {};
      try { saved = JSON.parse(this.deps.storage.getItem(STORAGE) || '{}'); } catch { /* Ignore stale browser state. */ }
      this.activated = saved.activated || '';
      const session = value.uploads.find(item => item.id === saved.id);
      if (session) {
        this.accept(session);
        this.phase = session.status === 'processing' ? 'processing' : session.status === 'complete' ? 'complete'
          : session.status === 'cancelled' ? 'cancelled' : 'paused';
      } else if (saved.id) {
        this.deps.storage.removeItem(STORAGE);
        this.error = '上次上传已过期或被清理，请重新上传。';
      }
      this.notify();
      if (session?.status === 'processing' || session?.status === 'complete') await this.start();
    } catch (error) { this.error = error instanceof Error ? error.message : String(error); this.notify(); }
  }

  select(identity: string) {
    if (this.running) return;
    this.file = null; this.error = '';
    const session = this.sessions.find(item => item.id === identity);
    if (session) {
      this.accept(session);
      this.phase = session.status === 'complete' ? 'complete' : session.status === 'processing' ? 'processing'
        : session.status === 'cancelled' ? 'cancelled' : 'paused';
    } else {
      this.session = null; this.phase = 'idle'; this.deps.storage.removeItem(STORAGE);
    }
    this.notify();
  }

  setFile(file: File | null) { if (!this.running) { this.file = file; this.error = ''; this.notify(); } }

  pause() {
    if (!this.running) return;
    this.controller?.abort(); this.phase = 'paused'; this.notify();
  }

  async cancel() {
    this.controller?.abort();
    if (!this.session) { this.phase = 'cancelled'; this.notify(); return; }
    try {
      this.accept(await this.deps.request<UploadInfo>(`/${this.session.id}`, { method: 'DELETE' }));
      this.phase = 'cancelled'; this.error = ''; this.notify();
    } catch (error) { this.error = error instanceof Error ? error.message : String(error); this.notify(); }
  }

  async start() {
    if (this.running) return;
    const controller = new AbortController();
    this.controller = controller; this.running = true; this.error = ''; this.notify();
    const signal = controller.signal;
    const request = <T>(path: string, options?: RequestInit) => retryUpload(
      () => this.deps.request<T>(path, options, signal), signal, this.deps.wait,
    );
    try {
      // A complete/processing restored session can finish without selecting the ZIP.
      if (this.session?.status === 'processing' || (this.session?.status === 'complete' && !this.file)) {
        this.accept(await request<UploadInfo>(`/${this.session.id}`));
      } else {
        const file = this.file;
        if (!file) throw new Error('请选择原 ZIP 文件后继续上传。');
        if (!file.size) throw new Error('ZIP 文件不能为空。');
        if (!file.name.toLowerCase().endsWith('.zip')) throw new Error('请选择 ZIP 格式的地图包。');
        if (!this.limits) this.limits = (await request<UploadList>('')).limits;
        if (file.size > this.limits.max_zip_bytes) throw new Error('ZIP 超过服务允许的大小。');
        const resuming = this.session && ['uploading', 'failed'].includes(this.session.status);
        const chunkSize = resuming ? this.session!.chunk_size : this.limits.chunk_size;
        this.phase = 'hashing'; this.hashProgress = 0; this.notify();
        const hashes = await this.deps.hash(file, chunkSize, signal, value => {
          this.hashProgress = value; this.notify();
        });
        signal.throwIfAborted();
        if (resuming) {
          if (this.session!.size !== file.size || this.session!.sha256 !== hashes.sha256) {
            throw new Error('所选文件与上传会话不一致，请重新选择原 ZIP，或新建上传。');
          }
          this.accept(await request<UploadInfo>(`/${this.session!.id}`));
        } else {
          this.accept(await request<UploadInfo>('', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: file.name, size: file.size, sha256: hashes.sha256 }) }));
        }
        signal.throwIfAborted();
        if (!this.processing() && this.session!.status !== 'complete') {
          this.phase = 'uploading'; this.notify();
          const session = this.session!;
          const received = new Set(session.received_chunks);
          const missing = Array.from({ length: session.total_chunks }, (_, index) => index).filter(index => !received.has(index));
          let next = 0, failure: unknown;
          const workers = Array.from({ length: Math.min(3, missing.length) }, async () => {
            while (next < missing.length && !failure) {
              signal.throwIfAborted();
              const index = missing[next++];
              try {
                const value = await request<UploadInfo>(`/${session.id}/chunks/${index}`, { method: 'PUT',
                  headers: { 'Content-Type': 'application/octet-stream', 'X-Chunk-SHA256': hashes.chunks[index] },
                  body: file.slice(index * chunkSize, Math.min(file.size, (index + 1) * chunkSize)) });
                value.received_chunks.forEach(item => received.add(item));
                value.received_chunks = [...received].sort((a, b) => a - b);
                value.received_bytes = [...received].reduce((sum, item) => sum + Math.min(chunkSize, file.size - item * chunkSize), 0);
                value.progress = value.received_bytes / file.size;
                this.accept(value);
              } catch (error) { failure = error; throw error; }
            }
          });
          const results = await Promise.allSettled(workers);
          const failed = results.find(result => result.status === 'rejected');
          if (failed?.status === 'rejected') throw failed.reason;
          signal.throwIfAborted();
          this.accept(await request<UploadInfo>(`/${session.id}/complete`, { method: 'POST' }));
        }
      }
      while (this.processing()) {
        this.phase = 'processing'; this.notify();
        await (this.deps.wait || sleep)(1000, signal);
        this.accept(await request<UploadInfo>(`/${this.session!.id}`));
      }
      signal.throwIfAborted();
      if (this.session?.status === 'complete' && this.session.map) {
        this.phase = 'complete'; this.file = null; this.notify();
        if (this.activated !== this.session.id) {
          await this.deps.onComplete(this.session.map);
          this.activated = this.session.id; this.accept(this.session);
        }
      } else {
        throw new Error(this.session?.error || '上传处理已中断，请继续上传或重试。');
      }
    } catch (error) {
      if (this.phase !== 'cancelled') this.phase = 'paused';
      if (!signal.aborted) this.error = error instanceof Error ? error.message : String(error);
    } finally {
      this.running = false; this.controller = null; this.notify();
    }
  }
}
