import React, { useState, useEffect, useRef, useCallback } from 'react';

// === 空间比对核心算法：计算余弦相似度 ===
// 提取关键骨骼向量（例如：左肩到左肘）
const getVector = (p1, p2) => {
  if (!p1 || !p2) return null;
  return { x: p2.x - p1.x, y: p2.y - p1.y };
};

// 计算两个向量的余弦相似度 (-1 到 1，1表示完全重合)
const cosineSimilarity = (v1, v2) => {
  if (!v1 || !v2) return 0;
  const dotProduct = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  if (mag1 === 0 || mag2 === 0) return 0;
  return dotProduct / (mag1 * mag2);
};

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // 核心数据引用
  const appState = useRef({
    recordedData: [], // 存储录制的"标准"动作 JSON 数据
    recordingStartTime: 0,
    playingStartTime: 0,
  });

  const [aiStatus, setAiStatus] = useState('初始化中...');
  const [isReady, setIsReady] = useState(false);
  const [mode, setMode] = useState('idle'); // 'idle', 'recording', 'playing'
  const [similarityScore, setSimilarityScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(0);

  // 1. 初始化 MediaPipe Pose (保持不变，加载 AI 模型)
  useEffect(() => {
    const loadMediaPipe = async () => {
      const scripts = [
        'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js',
      ];
      try {
        for (const src of scripts) {
          await new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) return resolve();
            const s = document.createElement('script');
            s.src = src;
            s.crossOrigin = 'anonymous';
            s.onload = resolve;
            s.onerror = reject;
            document.head.appendChild(s);
          });
        }
        setAiStatus('AI模型加载完成');
        setIsReady(true);
      } catch (err) {
        setAiStatus('AI加载失败');
      }
    };
    loadMediaPipe();
  }, []);

  // 2. 核心实时循环处理：分录制和跟练两种逻辑
  const onResults = useCallback((results) => {
    const canvasCtx = canvasRef.current?.getContext('2d');
    const canvas = canvasRef.current;
    if (!canvasCtx || !canvas) return;

    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    const state = appState.current;

    // 绘制用户的实时骨架
    if (results.poseLandmarks) {
      window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {
        color: 'rgba(0, 255, 0, 0.5)',
        lineWidth: 2,
      });
      window.drawLandmarks(canvasCtx, results.poseLandmarks, {
        color: '#FF0000',
        lineWidth: 1,
        radius: 2,
      });
    }

    setMode((currentMode) => {
      const now = Date.now();

      // === 模式A：录制标准动作 (相当于离线 Python 生成 JSON 的过程) ===
      if (currentMode === 'recording') {
        const timeElapsed = now - state.recordingStartTime;
        if (results.poseLandmarks) {
          state.recordedData.push({
            time: timeElapsed,
            landmarks: results.poseLandmarks,
          });
        }
        return currentMode;
      }

      // === 模式B：跟练与比对 (核心算法) ===
      if (currentMode === 'playing' && results.poseLandmarks && state.recordedData.length > 0) {
        const timeElapsed = now - state.playingStartTime;

        // 寻找时间戳最接近的标准动作帧
        const targetFrame = state.recordedData.reduce((prev, curr) =>
          Math.abs(curr.time - timeElapsed) < Math.abs(prev.time - timeElapsed) ? curr : prev
        );

        // 如果找到了目标帧，计算动作相似度
        if (targetFrame && targetFrame.landmarks) {
          const user = results.poseLandmarks;
          const target = targetFrame.landmarks;

          // 提取双臂的关键骨骼向量
          // 11: 左肩, 13: 左肘, 15: 左腕 | 12: 右肩, 14: 右肘, 16: 右腕
          const vectors = [
            { u: getVector(user[11], user[13]), t: getVector(target[11], target[13]) },
            { u: getVector(user[13], user[15]), t: getVector(target[13], target[15]) },
            { u: getVector(user[12], user[14]), t: getVector(target[12], target[14]) },
            { u: getVector(user[14], user[16]), t: getVector(target[14], target[16]) },
          ];

          // 计算所有向量的平均相似度
          let totalSim = 0;
          vectors.forEach((v) => {
            const sim = cosineSimilarity(v.u, v.t);
            // 归一化：将 -1~1 映射到 0~100 分
            totalSim += (sim + 1) * 50;
          });
          const avgScore = Math.max(0, Math.round(totalSim / vectors.length));

          setSimilarityScore(avgScore);

          // 在画面上绘制一个幽灵般的“标准动作导师”骨架
          window.drawConnectors(canvasCtx, target, window.POSE_CONNECTIONS, {
            color: 'rgba(255, 255, 255, 0.4)',
            lineWidth: 4,
          });
        }
      }
      return currentMode;
    });
  }, []);

  // 3. 启动引擎基础方法
  const initEngine = async () => {
    if (!videoRef.current.srcObject) {
      const pose = new window.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });
      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      pose.onResults(onResults);
      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) await pose.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      await camera.start();
    }
  };

  // 触发录制模式
  const startRecording = async () => {
    if (!isReady) return;
    await initEngine();

    appState.current.recordedData = [];
    appState.current.recordingStartTime = Date.now();
    setMode('recording');
    setAiStatus('🔴 录制中...');

    // 录制 5 秒钟
    let t = 5;
    setTimeLeft(t);
    const timer = setInterval(() => {
      t -= 1;
      setTimeLeft(t);
      if (t <= 0) {
        clearInterval(timer);
        setMode('idle');
        setAiStatus('✅ 录制完成，已保存为标准动作库');
      }
    }, 1000);
  };

  // 触发跟练模式
  const startPlaying = async () => {
    if (!isReady || appState.current.recordedData.length === 0) return;
    await initEngine();

    appState.current.playingStartTime = Date.now();
    setMode('playing');
    setAiStatus('🟢 跟练中...');

    // 回放 5 秒钟
    let t = 5;
    setTimeLeft(t);
    const timer = setInterval(() => {
      t -= 1;
      setTimeLeft(t);
      if (t <= 0) {
        clearInterval(timer);
        setMode('idle');
        setAiStatus('🏁 练习结束');
        setSimilarityScore(0);
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans selection:bg-pink-500">
      <header className="flex flex-col items-center justify-between border-b border-gray-800 bg-gray-950 p-4 sm:flex-row">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded bg-gradient-to-tr from-pink-500 to-purple-500 text-xl font-bold">
            抖
          </div>
          <h1 className="text-xl font-bold">
            动作捕捉与相似度比对系统{' '}
            <span className="rounded border border-pink-400 px-1 text-sm font-normal text-pink-400">
              Pro
            </span>
          </h1>
        </div>
        <div className="mt-2 text-sm text-gray-400 sm:mt-0">
          状态: <span className="text-yellow-400">{aiStatus}</span>
        </div>
      </header>

      <main className="mx-auto mt-4 max-w-4xl p-4">
        <div className="mb-6 rounded-2xl bg-gray-800 p-6 shadow-2xl">
          <div className="flex flex-col items-center justify-between gap-6 sm:flex-row">
            <div className="flex-1">
              <h2 className="mb-2 text-xl font-bold">1. 结构化相似度比对方案</h2>
              <p className="text-sm text-gray-400">
                第一步：点击录制，随意做几个手臂动作（5秒）；第二步：点击跟练，屏幕中白色的骨架就是刚才的你，努力让自己的绿色骨架与它重合！
              </p>
            </div>

            <div className="flex gap-4">
              <button
                onClick={startRecording}
                disabled={!isReady || mode !== 'idle'}
                className={`rounded-lg px-4 py-2 font-bold shadow-lg transition-all ${
                  !isReady || mode !== 'idle'
                    ? 'cursor-not-allowed bg-gray-700 text-gray-500'
                    : 'bg-red-500 text-white hover:bg-red-600'
                }`}
              >
                🔴 录制标准动作
              </button>

              <button
                onClick={startPlaying}
                disabled={!isReady || mode !== 'idle' || appState.current.recordedData.length === 0}
                className={`rounded-lg px-4 py-2 font-bold shadow-lg transition-all ${
                  !isReady || mode !== 'idle' || appState.current.recordedData.length === 0
                    ? 'cursor-not-allowed bg-gray-700 text-gray-500'
                    : 'bg-green-500 text-white hover:bg-green-600'
                }`}
              >
                🟢 开始跟练
              </button>
            </div>
          </div>
        </div>

        <div className="relative mx-auto aspect-[4/3] w-full max-w-3xl overflow-hidden rounded-2xl border-2 border-gray-700 bg-black">
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full scale-x-[-1] object-cover"
            playsInline
            autoPlay
            muted
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 z-10 h-full w-full scale-x-[-1] object-cover"
          />

          {(mode === 'recording' || mode === 'playing') && (
            <div className="absolute right-4 top-4 z-20 flex h-12 w-12 items-center justify-center rounded-full border border-gray-600 bg-black/50 text-xl font-bold text-white backdrop-blur">
              {timeLeft}s
            </div>
          )}

          {mode === 'playing' && (
            <div className="absolute bottom-6 left-1/2 z-20 flex -translate-x-1/2 items-center gap-4 rounded-full border-2 border-pink-500 bg-black/80 px-8 py-3 backdrop-blur">
              <span className="font-bold text-gray-300">动作相似度</span>
              <span
                className={`text-4xl font-black ${
                  similarityScore > 85
                    ? 'text-green-400'
                    : similarityScore > 60
                      ? 'text-yellow-400'
                      : 'text-red-400'
                }`}
              >
                {similarityScore}%
              </span>
              <span className="text-xl">
                {similarityScore > 85 ? '🔥 Perfect!' : similarityScore > 60 ? '👍 Good' : '⚠️ Miss'}
              </span>
            </div>
          )}

          {mode === 'idle' && !appState.current.recordedData.length && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/60">
              <div className="mb-4 text-4xl text-gray-400">请先点击上方录制标准动作</div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
