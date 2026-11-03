export interface InferenceState {
  status: 'loading' | 'ready' | 'failed';
  options: Record<string, unknown>;
  device: string;
  error: string | null;
}

export function sameOptions(a: Record<string, unknown>, b: Record<string, unknown>) {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => Object.is(a[key], b[key]));
}

export function inferenceControls(state: InferenceState | null, draft: Record<string, unknown>, connected: boolean, activeJobId: string | null) {
  const dirty = !!state && !sameOptions(state.options, draft);
  return {
    dirty,
    canApply: connected && !!state && state.status !== 'loading' && !activeJobId,
    canStart: connected && state?.status === 'ready' && !dirty && !activeJobId,
    message: !connected ? '正在连接模型服务…'
      : state?.status === 'loading' ? '模型加载中，请等待…'
      : state?.status === 'failed' ? `模型加载失败：${state.error || '请重新应用参数重试'}`
      : dirty ? '参数有修改，请先点击“应用参数”。'
      : `模型就绪 · ${state?.device || ''}`,
  };
}

export function shouldSyncOptions(previous: InferenceState | null, next: InferenceState, draft: Record<string, unknown>) {
  return previous !== null && sameOptions(previous.options, draft) && !sameOptions(previous.options, next.options);
}
