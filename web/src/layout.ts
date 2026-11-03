/** Page-local window layout. It deliberately has no map/job or storage state. */
export interface Rect { x: number; y: number; width: number; height: number }
export interface Bounds { width: number; height: number }
export type ResizeEdge = 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'nw';

const MIN_WIDTH = 220;
const MIN_HEIGHT = 180;
const TITLE_HEIGHT = 36;
const SIDEBAR_MIN = 280;
const EDGES: ResizeEdge[] = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'];
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max));

export function clampRect(rect: Rect, bounds: Bounds, minHeight = MIN_HEIGHT): Rect {
  const width = clamp(rect.width, Math.min(MIN_WIDTH, bounds.width), bounds.width);
  const height = clamp(rect.height, Math.min(minHeight, bounds.height), bounds.height);
  return { width, height, x: clamp(rect.x, 0, bounds.width - width), y: clamp(rect.y, 0, bounds.height - height) };
}

/** Preserve the opposite edge while a resize hits the minimum size or boundary. */
export function resizeRect(rect: Rect, edge: ResizeEdge, dx: number, dy: number, bounds: Bounds): Rect {
  const start = clampRect(rect, bounds);
  let left = start.x, right = start.x + start.width, top = start.y, bottom = start.y + start.height;
  const minWidth = Math.min(MIN_WIDTH, bounds.width), minHeight = Math.min(MIN_HEIGHT, bounds.height);
  if (edge.includes('w')) left = clamp(left + dx, 0, right - minWidth);
  if (edge.includes('e')) right = clamp(right + dx, left + minWidth, bounds.width);
  if (edge.includes('n')) top = clamp(top + dy, 0, bottom - minHeight);
  if (edge.includes('s')) bottom = clamp(bottom + dy, top + minHeight, bounds.height);
  return { x: left, y: top, width: right - left, height: bottom - top };
}

interface FloatingWindow {
  panel: HTMLElement;
  title: HTMLElement;
  button: HTMLButtonElement;
  name: string;
  rect: Rect;
  expandedHeight: number;
  collapsed: boolean;
  z: number;
}

interface Gesture {
  pointerId: number;
  handle: HTMLElement;
  x: number;
  y: number;
  rect: Rect;
  sidebarWidth: number;
  kind: 'move' | 'resize' | 'sidebar';
  window?: FloatingWindow;
  edge?: ResizeEdge;
}

export function initializeWorkspaceLayout(root: HTMLElement, resetButton: HTMLElement) {
  const layer = root.querySelector<HTMLElement>('.image-stack')!;
  const divider = root.querySelector<HTMLElement>('.sidebar-separator')!;
  const toolbar = root.querySelector<HTMLElement>('.viewport-toolbar')!;
  const desktop = window.matchMedia('(min-width: 1024px)');
  const events = new AbortController();
  const listener = { signal: events.signal };
  const windows: FloatingWindow[] = Array.from(layer.querySelectorAll<HTMLElement>('[data-floating-window]')).map((panel, index) => ({
    panel, title: panel.querySelector<HTMLElement>('[data-window-drag]')!,
    button: panel.querySelector<HTMLButtonElement>('[data-window-collapse]')!,
    name: panel.dataset.windowName!, rect: { x: 0, y: 0, width: 300, height: 260 },
    expandedHeight: 260, collapsed: false, z: index + 1,
  }));
  let placed = false, sidebarCustom = false, sidebarWidth = 320, topZ = windows.length;
  let gesture: Gesture | undefined;

  function bounds(): Bounds { return { width: layer.clientWidth, height: layer.clientHeight }; }
  function sidebarMax() {
    const gap = parseFloat(getComputedStyle(root).columnGap) || 0;
    return Math.max(SIDEBAR_MIN, bounds().width - divider.offsetWidth - gap * 2 - 320);
  }
  function defaultSidebar() { return clamp(window.innerWidth * 0.29, 320, 440); }

  function paintSidebar() {
    sidebarWidth = clamp(sidebarWidth, SIDEBAR_MIN, sidebarMax());
    root.style.setProperty('--sidebar-width', `${sidebarWidth}px`);
    divider.setAttribute('aria-valuenow', String(Math.round(sidebarWidth)));
    divider.setAttribute('aria-valuemin', String(SIDEBAR_MIN));
    divider.setAttribute('aria-valuemax', String(Math.floor(sidebarMax())));
    divider.setAttribute('aria-valuetext', `${Math.round(sidebarWidth)} 像素`);
  }

  function paint(item: FloatingWindow) {
    const { panel, button, rect } = item;
    panel.style.setProperty('--window-x', `${rect.x}px`);
    panel.style.setProperty('--window-y', `${rect.y}px`);
    panel.style.setProperty('--window-width', `${rect.width}px`);
    panel.style.setProperty('--window-height', `${rect.height}px`);
    panel.style.zIndex = String(item.z);
    panel.classList.toggle('is-collapsed', item.collapsed);
    button.textContent = item.collapsed ? '▾' : '▴';
    button.setAttribute('aria-expanded', String(!item.collapsed));
    button.setAttribute('aria-label', `${item.collapsed ? '展开' : '收起'} ${item.name}`);
    button.title = item.collapsed ? '展开图像' : '收起为标题栏';
    item.title.tabIndex = desktop.matches ? 0 : -1;
  }

  function front(item: FloatingWindow) {
    if (item.z !== topZ) item.z = ++topZ;
    item.panel.style.zIndex = String(item.z);
  }

  function finish(event?: PointerEvent) {
    if (!gesture || (event && event.pointerId !== gesture.pointerId)) return;
    const previous = gesture;
    gesture = undefined;
    delete root.dataset.layoutInteraction;
    document.documentElement.classList.remove('layout-interacting');
    document.documentElement.style.removeProperty('--layout-cursor');
    if (previous.handle.hasPointerCapture(previous.pointerId)) previous.handle.releasePointerCapture(previous.pointerId);
  }

  function measure() {
    if (!desktop.matches) {
      finish();
      windows.forEach(paint);
      return;
    }
    const area = bounds();
    if (area.width < 1 || area.height < 1) return;
    if (!sidebarCustom) sidebarWidth = defaultSidebar();
    paintSidebar();
    if (!placed) {
      const top = Math.min(toolbar.offsetHeight + 12, Math.max(0, area.height - MIN_HEIGHT));
      const height = clamp(Math.floor((area.height - top - 12) / 2), Math.min(MIN_HEIGHT, area.height), Math.min(260, area.height));
      windows.forEach((item, index) => {
        item.rect = clampRect({ x: area.width - 312, y: top + index * (height + 12), width: 300, height }, area);
        item.expandedHeight = item.rect.height;
      });
      placed = true;
    }
    windows.forEach(item => {
      item.rect = clampRect(item.rect, area, item.collapsed ? TITLE_HEIGHT : MIN_HEIGHT);
      paint(item);
    });
  }

  function toggle(item: FloatingWindow) {
    finish();
    front(item);
    if (item.collapsed) {
      item.collapsed = false;
      item.rect.height = item.expandedHeight;
    } else {
      item.expandedHeight = item.rect.height;
      item.collapsed = true;
      item.rect.height = TITLE_HEIGHT;
    }
    if (desktop.matches) item.rect = clampRect(item.rect, bounds(), item.collapsed ? TITLE_HEIGHT : MIN_HEIGHT);
    paint(item);
  }

  function begin(event: PointerEvent, kind: Gesture['kind'], item?: FloatingWindow, edge?: ResizeEdge) {
    if (!desktop.matches || event.button !== 0 || !event.isPrimary || (kind === 'resize' && item?.collapsed)) return;
    finish();
    if (item) front(item);
    event.preventDefault();
    event.stopPropagation();
    const handle = event.currentTarget as HTMLElement;
    gesture = {
      pointerId: event.pointerId, handle, x: event.clientX, y: event.clientY,
      rect: { ...(item?.rect ?? { x: 0, y: 0, width: 0, height: 0 }) },
      sidebarWidth, kind, window: item, edge,
    };
    if (handle.tabIndex >= 0) handle.focus({ preventScroll: true });
    handle.setPointerCapture(event.pointerId);
    root.dataset.layoutInteraction = kind;
    document.documentElement.style.setProperty('--layout-cursor', getComputedStyle(handle).cursor);
    document.documentElement.classList.add('layout-interacting');
  }

  function move(event: PointerEvent) {
    if (!gesture || event.pointerId !== gesture.pointerId) return;
    event.preventDefault();
    const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
    if (gesture.kind === 'sidebar') {
      sidebarCustom = true;
      sidebarWidth = gesture.sidebarWidth + dx;
      paintSidebar();
    } else if (gesture.window) {
      const item = gesture.window;
      item.rect = gesture.kind === 'resize'
        ? resizeRect(gesture.rect, gesture.edge!, dx, dy, bounds())
        : clampRect({ ...gesture.rect, x: gesture.rect.x + dx, y: gesture.rect.y + dy }, bounds(), item.collapsed ? TITLE_HEIGHT : MIN_HEIGHT);
      if (gesture.kind === 'resize') item.expandedHeight = item.rect.height;
      paint(item);
    }
  }

  function keyboardDelta(event: KeyboardEvent) {
    const step = event.shiftKey ? 50 : 10;
    switch (event.key) {
      case 'ArrowLeft': return { x: -step, y: 0 };
      case 'ArrowRight': return { x: step, y: 0 };
      case 'ArrowUp': return { x: 0, y: -step };
      case 'ArrowDown': return { x: 0, y: step };
      default: return undefined;
    }
  }

  windows.forEach(item => {
    item.panel.addEventListener('pointerdown', event => { front(item); event.stopPropagation(); }, listener);
    item.title.addEventListener('pointerdown', event => {
      if (!(event.target as HTMLElement).closest('button')) begin(event, 'move', item);
    }, listener);
    item.title.addEventListener('keydown', event => {
      if (!desktop.matches || event.target !== item.title) return;
      const delta = keyboardDelta(event);
      if (!delta) return;
      event.preventDefault(); event.stopPropagation(); front(item);
      item.rect = clampRect({ ...item.rect, x: item.rect.x + delta.x, y: item.rect.y + delta.y }, bounds(), item.collapsed ? TITLE_HEIGHT : MIN_HEIGHT);
      paint(item);
    }, listener);
    item.title.addEventListener('dblclick', event => {
      if (!(event.target as HTMLElement).closest('button')) toggle(item);
    }, listener);
    item.button.addEventListener('click', () => toggle(item), listener);
    EDGES.forEach(edge => {
      const handle = document.createElement('span');
      handle.className = `window-resize-handle resize-${edge}`;
      handle.dataset.resize = edge;
      if (edge === 'se') {
        handle.tabIndex = 0;
        handle.setAttribute('role', 'button');
        handle.setAttribute('aria-label', `调整 ${item.name} 窗口大小`);
        handle.setAttribute('aria-describedby', 'layout-help');
        handle.addEventListener('keydown', event => {
          if (!desktop.matches || item.collapsed) return;
          const delta = keyboardDelta(event);
          if (!delta) return;
          event.preventDefault(); event.stopPropagation(); front(item);
          item.rect = resizeRect(item.rect, 'se', delta.x, delta.y, bounds());
          item.expandedHeight = item.rect.height;
          paint(item);
        }, listener);
      } else handle.setAttribute('aria-hidden', 'true');
      handle.addEventListener('pointerdown', event => begin(event, 'resize', item, edge), listener);
      item.panel.append(handle);
    });
  });

  divider.addEventListener('pointerdown', event => begin(event, 'sidebar'), listener);
  divider.addEventListener('keydown', event => {
    if (!desktop.matches) return;
    const delta = keyboardDelta(event);
    if (!delta?.x && event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault(); event.stopPropagation(); sidebarCustom = true;
    sidebarWidth = event.key === 'Home' ? SIDEBAR_MIN : event.key === 'End' ? sidebarMax() : sidebarWidth + delta!.x;
    paintSidebar();
  }, listener);
  function reset() {
    finish(); sidebarCustom = false; placed = false; topZ = windows.length;
    windows.forEach((item, index) => {
      item.collapsed = false; item.z = index + 1; item.expandedHeight = 260;
      item.rect = { x: 0, y: 0, width: 300, height: 260 };
      paint(item);
    });
    measure();
  }
  resetButton.addEventListener('click', reset, listener);
  window.addEventListener('pointermove', move, listener);
  window.addEventListener('pointerup', finish, listener);
  window.addEventListener('pointercancel', finish, listener);
  window.addEventListener('lostpointercapture', finish, listener);
  window.addEventListener('blur', () => finish(), listener);
  window.addEventListener('resize', () => { finish(); measure(); }, listener);
  desktop.addEventListener('change', measure, listener);
  const observer = new ResizeObserver(measure);
  observer.observe(layer);
  measure();
  return { reset, dispose: () => { finish(); observer.disconnect(); events.abort(); } };
}
