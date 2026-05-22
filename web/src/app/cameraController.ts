/**
 * Camera controller.
 *
 * Thin wrapper around getUserMedia + track cleanup so views stay free of
 * media plumbing. Exposes the constraints builder as a pure function for unit
 * testing; getUserMedia itself is injected as a dependency so tests can run in
 * a plain jsdom environment without a real camera.
 */

export interface FrontCameraConstraints {
  facingMode: { ideal: 'user' };
  width: { ideal: number };
  height: { ideal: number };
}

export function getPreferredFrontCameraConstraints(): FrontCameraConstraints {
  return {
    facingMode: { ideal: 'user' },
    width: { ideal: 1280 },
    height: { ideal: 720 },
  };
}

export interface CameraControllerOptions {
  /** Override for tests; defaults to navigator.mediaDevices.getUserMedia. */
  requestUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
}

export interface CameraController {
  start(): Promise<MediaStream>;
  stop(): void;
  readonly stream: MediaStream | null;
  readonly running: boolean;
}

function defaultRequestUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices) {
    return Promise.reject(new Error('mediaDevices unavailable in this environment'));
  }
  return navigator.mediaDevices.getUserMedia(constraints);
}

export function describeCameraError(error: unknown): string {
  const name = error instanceof DOMException || error instanceof Error ? error.name : '';
  const message = error instanceof Error ? error.message : String(error || '');
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return '摄像头权限被拒绝，请在浏览器地址栏或系统隐私设置中允许摄像头访问';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return '没有检测到摄像头，请确认设备已连接';
    case 'NotReadableError':
    case 'TrackStartError':
      return '摄像头正在被其他应用占用，请关闭占用摄像头的软件后重试';
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return '当前摄像头不支持请求的分辨率或前置摄像头约束，请切换摄像头或降低分辨率';
    case 'SecurityError':
      return '当前页面环境不允许访问摄像头，请使用 localhost 或 127.0.0.1 打开页面';
    default:
      return message || '摄像头授权失败';
  }
}

function stopStreamTracks(target: MediaStream): void {
  target.getTracks().forEach((t) => t.stop());
}

export function createCameraController(options: CameraControllerOptions = {}): CameraController {
  const request = options.requestUserMedia ?? defaultRequestUserMedia;
  let stream: MediaStream | null = null;
  let startSequence = 0;

  return {
    async start() {
      const token = ++startSequence;
      const nextStream = await request({ audio: false, video: getPreferredFrontCameraConstraints() });
      if (token !== startSequence) {
        stopStreamTracks(nextStream);
        if (stream) return stream;
        throw new Error('stale camera start ignored');
      }
      if (stream && stream !== nextStream) {
        stopStreamTracks(stream);
      }
      stream = nextStream;
      return stream;
    },
    stop() {
      startSequence += 1;
      if (!stream) return;
      stopStreamTracks(stream);
      stream = null;
    },
    get stream() {
      return stream;
    },
    get running() {
      return stream !== null;
    },
  };
}
