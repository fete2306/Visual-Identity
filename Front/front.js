/* Front/front.js — JS 版前端,调用后端 app.py /infer/batch
 * 双用:
 *  1) 浏览器: <script src="./front.js"></script> 后调用 FrontJS.startCamera(videoEl, opts)
 *     或直接配合 Front/front.html 使用 (开箱即用)。
 *     摄像头: 下拉 <select id="camera"> 经 enumerateDevices 枚举, 切换自动重开;
 *     帧率: #fps 显示 渲染fps(overlay rAF 实测) + 推理fps(后端往返节拍), 对标 py 版 --show-fps.
 *  2) Node: node Front/front.js <image.jpg> [--server https://127.0.0.1:8060] [--min-sim 0.4]
 *     Node>=18 自带 fetch/FormData/Blob,无需额外依赖。
 *     自签 https 下 Node CLI 需加环境变量 NODE_TLS_REJECT_UNAUTHORIZED=0 (仅开发机, 浏览器走点击信任)。
 *
 * 后端接口 (app.py /infer/batch, 原样引用): POST {server}/infer/batch?min_sim=0.4
 *   multipart, 字段 {index: img}: 第 i 张人脸以字段名 String(i) 发送, 顺序即索引.
 *   返回: {"time": float, "results": {"0": [name, sim], "1": [...]}} (k 由发送方已知, 无 count)
 *   解析另兼容旧版单体 results=[name, sim].
 *
 * 可视化(浏览器): overlay 画布叠加在 video 上 —
 *   绿色框=MediaPipe BlazeFace 本地人脸框(./vendor/, 全浏览器通用;
 *     Windows 版 Edge/Chrome 原生 FaceDetector 不可用, 仅作备用),
 *     最大框带 name:sim 标签(身份来自最近一次/infer/batch),
 *   蓝色点线=MediaPipe HandLandmarker 手部21点骨架(./vendor/ 本地, 失败不影响识别)。
 * 离线可用: vendor 目录已包含 wasm 运行时与两个模型, 只需用 http 打开页面
 *   (D:\conda\envs\cuda128\python.exe -m http.server 8080, 再访问 http://127.0.0.1:8080/Front/front.html)。
 */
(function (global) {
  // Node 下用 Object.values(os.networkInterfaces()) 取首个非内网IPv4做默认后端;
  // 浏览器下没有 require/os, 返回 null, 由 location.origin(同源) 或 127.0.0.1 兜底.
  function getLocalIp() {
    try {
      if (typeof require === "undefined") return null;
      const os = require("os");
      for (const nets of Object.values(os.networkInterfaces())) {
        for (const net of nets || []) {
          if (net.family === "IPv4" && !net.internal) return net.address;
        }
      }
    } catch (_) {}
    return null;
  }
  const _isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  const _lanIp = _isNode ? (getLocalIp() || "127.0.0.1") : "127.0.0.1";
  const DEFAULTS = {
    server: "https://" + _lanIp + ":8060",
    minSim: 0.4,
    intervalMs: 800,   // 浏览器连续推理节流
    jpegQuality: 0.9,
    captureWidth: 640,
  };

  // 后端输入框只填裸 host; 若粘了 ".../infer/batch" 等完整地址甚至带 ?min_sim, 统一归一化.
  // 去后缀必须长串优先 (先 batch/single 再 infer), 否则 ".../infer/batch" 会剩 "/batch".
  const INFER_PATH = "/infer/batch";
  function normalizeServer(server) {
    server = String(server || "").trim().split("?")[0].replace(/\/+$/, "");
    for (const suffix of ["/infer/batch", "/infer/single", "/infer"]) {
      if (server.toLowerCase().endsWith(suffix)) {
        server = server.slice(0, -suffix.length).replace(/\/+$/, "");
        break;
      }
    }
    return server || DEFAULTS.server;
  }

  async function testConnection(server = DEFAULTS.server) {
    server = normalizeServer(server);
    // openapi.json 是 FastAPI 默认自带的,GET 即可,不依赖模型,适合做连通性探测
    const url = server + "/openapi.json";
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5000);
    try {
      const resp = await fetch(url, { signal: ctrl.signal });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      await resp.json();
      return { ok: true, server };
    } finally {
      clearTimeout(timer);
    }
  }

  // 批量: 一次 POST k 张, 字段 {index: img} (字段名即索引 "0".."k-1", 顺序即索引;
  // 理由: 与后端批量接口对齐, k 张脸 1 次往返; k 发送方已知, 不需要 count).
  // 返回 {list: [{name, sim}...] (按索引对齐, 缺失补 unknow), serverTime, rttMs}.
  // 解析兼容新版 results-dict 与旧版单体 results (理由: 过渡期不断连).
  async function inferBatch(blobs, server = DEFAULTS.server, minSim = DEFAULTS.minSim) {
    server = normalizeServer(server);
    const url = server + INFER_PATH + "?min_sim=" + encodeURIComponent(minSim);
    const fd = new FormData();
    blobs.forEach((b, i) => fd.append(String(i), b, `face${i}.jpg`));
    const t0 = performance.now();
    try {
      const resp = await fetch(url, { method: "POST", body: fd });
    const rtt = performance.now() - t0;
    if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + (await resp.text()).slice(0, 200));
    const data = await resp.json();
    const res = (data.results !== undefined) ? data.results : ["unknow", 0];
    let pairs;
    if (res !== null && typeof res === "object" && !Array.isArray(res)) {
      pairs = Object.keys(res).sort((a, b) => Number(a) - Number(b)).map((k) => res[k]);
    } else if (Array.isArray(res) && res.length === 2 && typeof res[0] === "string") {
      pairs = [res]; // 旧版单体
    } else if (Array.isArray(res)) {
      pairs = res;
    } else {
      pairs = [];
    }
    const list = pairs.map((p) => ({ name: String(p[0]), sim: Number(p[1]) }));
    while (list.length < blobs.length) list.push({ name: "unknow", sim: 0 }); // 索引缺失补齐
    // 成功一次就清零连续失败计数
    if (typeof document !== "undefined" && state) state.failCount = 0;
    return { list, serverTime: Number(data.time || 0), rttMs: rtt };
  } catch (e) {
    // fetch 网络层失败时只有 "Failed to fetch", 在这里补上目标 URL 和排查提示
    if (e instanceof TypeError || /fetch|network|empty|aborted/i.test(e.message || "")) {
      throw new Error(
        `连不上后端 ${url} (${e.message})。请检查: 1)后端是否已启动 ` +
        `uvicorn app:app --host 0.0.0.0 --port 8060; 2)浏览器能否打开 ` +
        `${server}/docs; 3)端口是否被其它程序占用(被占用就换端口,前后端一起改); ` +
        `4)后端输入框只填 http://127.0.0.1:8060,不要带/infer/batch`
      );
    }
    throw e;
  }
  }

  // 单图兼容: 老调用点 (传图/Node CLI) 走批量通道发 1 张, 取索引 0
  // (理由: 单图路径复用同一份批量逻辑, 不写两套).
  async function inferBlob(blob, server = DEFAULTS.server, minSim = DEFAULTS.minSim) {
    const r = await inferBatch([blob], server, minSim);
    const one = r.list[0] || { name: "unknow", sim: 0 };
    return { name: one.name, sim: one.sim, serverTime: r.serverTime, rttMs: r.rttMs };
  }

  // ---- 浏览器: 摄像头循环推理 (对标 base1.solve 的跳帧+节流思想) ----
  const state = { stream: null, timer: null, inflight: false, faceDetector: null, failCount: 0,
                  identity: { name: "?", sim: 0 }, deviceId: "" }; // 最近一次 /infer/batch 结果,画在最大人脸框上
  // 帧率表: renderFps=overlay rAF 实测(对标 py 版 show_fps), inferFps=captureAndInfer 成功节拍
  const fpsMeter = { frames: 0, lastT: 0, renderFps: 0,
                     infers: 0, inferLastT: 0, inferFps: 0 };

  // ---- Overlay: 人脸框 + 手部点,纯本地绘制,不走后端 ----
  // 人脸框: MediaPipe BlazeFace (./vendor/ 本地, 全浏览器通用; Windows 下 Chromium
  //   原生 FaceDetector 不可用, 仅作备用降级)
  // 手部点: MediaPipe HandLandmarker (./vendor/ 本地; 加载失败则只画框)
  const overlay = { faces: [], facesTs: 0, hands: [], handsTs: 0,
                    faceBusy: false, handBusy: false, raf: 0,
                    vision: null, visionFailed: false,
                    faceDetectorMp: null, faceInitDone: false,
                    handLandmarker: null, handInitDone: false, _ts: 0,
                    lastBatch: { boxes: [], ids: [], ts: 0 } };
  // 21点手部骨架 (与 mediapipe HAND_CONNECTIONS 一致, py版同款)
  const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

  // 身份状态预留字段字典 (与后端保留标签对齐; 后端新增保留字段只需在这里加一行映射)
  // INFO绿=已识别 / WARNING黄=未知(unknow) / ERROR红=跳过失败(outOfrecognitionCount) / IDLE灰=无结果
  const STATUS = {
    INFO: { color: "#16a34a", box: "#22c55e", label: "已识别" },
    WARNING: { color: "#ca8a04", box: "#eab308", label: "未知" },
    ERROR: { color: "#dc2626", box: "#ef4444", label: "跳过/失败" },
    IDLE: { color: "#9ca3af", box: "#9ca3af", label: "无结果" },
  };
  function statusOf(name) {
    if (name === "outOfrecognitionCount") return "ERROR";
    if (name === "unknow") return "WARNING";
    if (!name || name === "?") return "IDLE";
    return "INFO";
  }

  function setHandStatus(msg) {
    if (typeof document === "undefined") return;
    const el = document.getElementById("handStatus");
    if (el) el.textContent = msg;
  }

  function setFaceStatus(msg) {
    if (typeof document === "undefined") return;
    const el = document.getElementById("faceStatus");
    if (el) el.textContent = msg;
  }

  // ---- 帧率显示 (对标 front_py --show-fps): renderFps 每 500ms 刷一次 #fps ----
  function setFpsText() {
    if (typeof document === "undefined") return;
    const el = document.getElementById("fps");
    if (!el) return;
    const r = fpsMeter.renderFps ? fpsMeter.renderFps.toFixed(1) : "--";
    const inf = fpsMeter.inferFps ? fpsMeter.inferFps.toFixed(1) : "--";
    el.textContent = `FPS: 渲染 ${r} | 推理 ${inf}`;
  }

  function tickRenderFps() {
    const now = performance.now();
    if (!fpsMeter.lastT) fpsMeter.lastT = now;
    fpsMeter.frames++;
    if (now - fpsMeter.lastT >= 500) { // 半秒一算, 避免每帧写 DOM
      fpsMeter.renderFps = (fpsMeter.frames * 1000) / (now - fpsMeter.lastT);
      fpsMeter.frames = 0;
      fpsMeter.lastT = now;
      setFpsText();
    }
  }

  function tickInferFps() {
    const now = performance.now();
    if (fpsMeter.inferLastT) { // 两次成功推理的间隔 -> 即时 fps, 再 EMA 平滑 (理由: setInterval 节流+跳帧, 瞬时值抖)
      const inst = 1000 / Math.max(1, now - fpsMeter.inferLastT);
      fpsMeter.inferFps = fpsMeter.inferFps ? fpsMeter.inferFps * 0.7 + inst * 0.3 : inst;
      setFpsText();
    }
    fpsMeter.inferLastT = now;
  }

  // ---- 摄像头列表: enumerateDevices 枚举, 下拉选择 deviceId (理由: 笔记本+USB 多路时 facingMode 不够用) ----
  async function listCameras() {
    if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return [];
    }
    const devs = await navigator.mediaDevices.enumerateDevices();
    return devs.filter((d) => d.kind === "videoinput").map((d, i) => ({
      deviceId: d.deviceId || "",
      label: d.label || `摄像头 ${i + 1}`, // 未授权时 label 为空, 先编号占位, 授权后刷新即有真名
    }));
  }

  // deviceId 很长(几十位 hash), 日志只打头尾定位, 不刷屏
  function shortId(id) {
    id = String(id || "");
    if (id.length <= 16) return id || "(空)";
    return id.slice(0, 8) + "…" + id.slice(-4);
  }

  async function refreshCameraList() {
    if (typeof document === "undefined") return [];
    const sel = document.getElementById("camera");
    if (!sel) return [];
    const prev = sel.value;
    // 非安全上下文直接给结论, 不残留“加载中” (理由: file:// 或 http+局域网IP 下 Chrome 会把
    // navigator.mediaDevices 置空; https 同源版正常, 异常多半是自签未信任或权限被拒)
    if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      sel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "当前页面被禁摄像头枚举(确认用 https 打开且已过 /docs 自签信任)";
      sel.appendChild(opt);
      log("摄像头枚举不可用: 非安全上下文或权限策略拦截。请用 https 同源地址打开 (先过 /docs 信任), 或点“刷新摄像头”重试");
      return [];
    }
    // 枚举明细进日志 (理由: https 下 label 为空/数量不对时, 靠明细判断是未授权还是真没设备)
    try { log(`[cam] 开始枚举 secure=${window.isSecureContext}`); } catch (_) {}
    let cams = [];
    let rawKinds = "";
    try {
      cams = await listCameras();
      try { // 顺手统计原始种类, 空列表时能区分“一个设备都没有”还是“有音频无视频”
        const raw = await navigator.mediaDevices.enumerateDevices();
        const kinds = {};
        for (const d of raw) kinds[d.kind || "?"] = (kinds[d.kind || "?"] || 0) + 1;
        rawKinds = Object.keys(kinds).map((k) => `${k}x${kinds[k]}`).join(",");
      } catch (_) {}
    } catch (e) {
      // 之前 bug: catch 里直接 return, 下拉永远残留“加载中” —— 这里必须先把占位换掉
      sel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = prev || "";
      opt.textContent = "枚举失败(点刷新重试): " + (e && e.message ? e.message : e);
      sel.appendChild(opt);
      try { log("枚举摄像头失败: " + (e && e.message ? e.message : e)); } catch (_) {}
      return [];
    }
    sel.innerHTML = "";
    if (!cams.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "未找到摄像头(点刷新重试)";
      sel.appendChild(opt);
      log(`未找到摄像头: 原始设备[${rawKinds || "?"}]。请确认已授权 + 用 https 打开 (自签先过 /docs 信任, 见页首说明)`);
      return cams;
    }
    for (const c of cams) {
      const opt = document.createElement("option");
      opt.value = c.deviceId;
      opt.textContent = c.label;
      sel.appendChild(opt);
    }
    // 尽量保持上次选择 (理由: 刷新列表不应把用户已选的路踢掉)
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
    state.deviceId = sel.value || "";
    log(`摄像头列表: ${cams.length} 路 (原始设备[${rawKinds || "?"}])`);
    cams.forEach((c, i) => log(`[cam #${i}] "${c.label}" id=${shortId(c.deviceId)}`)); // 逐路明细, label 为空=未授权, 开流后刷新即有真名
    return cams;
  }

  function getSelectedDeviceId() {
    if (typeof document === "undefined") return state.deviceId || "";
    return document.getElementById("camera")?.value || "";
  }

  function largestFace() {
    if (!overlay.faces.length) return null;
    return overlay.faces.slice().sort((a, b) => b.w * b.h - a.w * a.h)[0];
  }

  // MediaPipe VIDEO 模式要求时间戳整数且单调递增, 人脸/手部两个检测器共用一个计数器
  function nextTs() {
    overlay._ts = Math.max(Math.floor(performance.now()), overlay._ts + 1);
    return overlay._ts;
  }

  async function ensureVision() {
    if (overlay.vision || overlay.visionFailed) return overlay.vision;
    if (!global.__mpVision) return null; // module 脚本还没加载好, 等下一帧
    try {
      overlay.vision = await global.__mpVision.FilesetResolver.forVisionTasks(global.__mpVision.wasmBase);
    } catch (e) {
      overlay.visionFailed = true;
      setFaceStatus("人脸检测: 本地运行时加载失败: " + e.message);
      setHandStatus("手部模型: 本地运行时加载失败: " + e.message);
      log("本地检测运行时加载失败(确认用 http 方式打开页面): " + e.message);
    }
    return overlay.vision;
  }

  async function ensureFaceDetectorMp() {
    if (overlay.faceDetectorMp || overlay.faceInitDone) return overlay.faceDetectorMp;
    const vision = await ensureVision();
    if (!vision) return null;
    overlay.faceInitDone = true;
    try {
      const { FaceDetector } = global.__mpVision;
      const mk = (delegate) => FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: global.__mpVision.faceModel, delegate },
        runningMode: "VIDEO", minDetectionConfidence: 0.5, minSuppressionThreshold: 0.3,
      });
      try { overlay.faceDetectorMp = await mk("GPU"); }
      catch (e) { overlay.faceDetectorMp = await mk("CPU"); } // GPU delegate 不可用时回退
      setFaceStatus("人脸检测: MediaPipe BlazeFace 已就绪(本地)");
      log("人脸检测 BlazeFace 已就绪");
    } catch (e) {
      overlay.faceDetectorMp = null;
      setFaceStatus("人脸检测: BlazeFace 加载失败, 尝试原生 FaceDetector 降级: " + e.message);
    }
    return overlay.faceDetectorMp;
  }

  // 跳帧间隔读页面输入 (理由: 机器性能差很多, 硬编码 120/100 不通用; 读 DOM 即时生效, 不用重开流)
  function getDetectThrottleMs() {
    if (typeof document === "undefined") return { face: 120, hand: 100 };
    const f = parseInt(document.getElementById("faceInterval")?.value ?? "120", 10);
    const h = parseInt(document.getElementById("handInterval")?.value ?? "100", 10);
    return {
      face: Number.isFinite(f) ? Math.max(16, Math.min(2000, f)) : 120, // 下限一帧, 上限 2s, 非法回退默认
      hand: Number.isFinite(h) ? Math.max(16, Math.min(2000, h)) : 100,
    };
  }

  async function pumpFaces(video) {
    // 节流约 8fps, 避免阻塞渲染
    if (overlay.faceBusy) return;
    const now = performance.now();
    if (overlay.facesTs !== 0 && now - overlay.facesTs < getDetectThrottleMs().face) return;
    overlay.faceBusy = true;
    try {
      const mp = await ensureFaceDetectorMp();
      if (mp) {
        const res = mp.detectForVideo(video, nextTs());
        overlay.faces = (res.detections || []).map((d) => {
          const b = d.boundingBox; // 像素坐标 {originX, originY, width, height}
          return { x: b.originX, y: b.originY, w: b.width, h: b.height,
                   score: (d.categories && d.categories[0]) ? d.categories[0].score : 0 };
        });
        overlay.facesTs = performance.now();
      } else if ("FaceDetector" in window) {
        // 备用: 原生 FaceDetector (macOS/ChromeOS/Android 的 Chromium 才有, Windows 没有)
        if (!state.faceDetector) state.faceDetector = new FaceDetector({ fastMode: true });
        const dets = await state.faceDetector.detect(video);
        overlay.faces = dets.map((d) => {
          const b = d.boundingBox;
          return { x: b.x, y: b.y, w: b.width, h: b.height, score: 0 };
        });
        overlay.facesTs = performance.now();
        setFaceStatus("人脸检测: 原生 FaceDetector 降级模式");
      } else if (!overlay.vision && !overlay.visionFailed) {
        setFaceStatus("人脸检测: 本地模型加载中(须用 http 打开页面, 见页首说明)...");
      }
    } catch (e) { /* 忽略单帧失败 */ }
    if (overlay.faces.length !== overlay._lastFaceN) { // 只在数量变化时记, 不刷屏
      overlay._lastFaceN = overlay.faces.length;
      log(`人脸检测: 当前 ${overlay.faces.length} 张脸`);
    }
    overlay.faceBusy = false;
  }

  async function ensureHandLandmarker() {
    if (overlay.handLandmarker || overlay.handInitDone) return overlay.handLandmarker;
    const vision = await ensureVision();
    if (!vision) return null;
    overlay.handInitDone = true;
    try {
      const { HandLandmarker } = global.__mpVision;
      const mk = (delegate) => HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: global.__mpVision.handModel, delegate },
        runningMode: "VIDEO", numHands: 2,
      });
      try { overlay.handLandmarker = await mk("GPU"); }
      catch (e) { overlay.handLandmarker = await mk("CPU"); } // GPU delegate 不可用时回退
      setHandStatus("手部模型: 已就绪(本地)");
      log("手部模型已就绪");
    } catch (e) {
      setHandStatus("手部模型加载失败(人脸框不受影响): " + e.message);
      overlay.handLandmarker = null;
    }
    return overlay.handLandmarker;
  }

  async function pumpHands(video) {
    if (overlay.handBusy) return;
    const now = performance.now();
    if (overlay.handsTs !== 0 && now - overlay.handsTs < getDetectThrottleMs().hand) return;
    const lm = await ensureHandLandmarker();
    if (!lm) return;
    overlay.handBusy = true;
    try {
      const res = lm.detectForVideo(video, nextTs());
      const h = res && res.landmarks ? res.landmarks : (res && res.hand_landmarks ? res.hand_landmarks : []);
      overlay.hands = (h || []).map((hand) => hand.map((p) => ({ x: p.x, y: p.y })));
      overlay.handsTs = performance.now();
    } catch (e) { /* 时间戳/帧率问题时忽略单帧 */ }
    if (overlay.hands.length !== overlay._lastHandN) { // 只在数量变化时记, 不刷屏
      overlay._lastHandN = overlay.hands.length;
      log(`手部检测: 当前 ${overlay.hands.length} 只手`);
    }
    overlay.handBusy = false;
  }

  // 批量身份回填: 最近一次 inferBatch 的第 i 个结果对应发送时的第 i 张裁图.
  // 框在 120ms 级刷新而请求往返要几百 ms, 按中心点最近邻匹配 (漂移超一框则不认,
  // 理由: 顺序下标只在发送瞬间有效, 绘制时必须用几何匹配, 否则张冠李戴).
  // batch 参数化后 video 循环与静态图片共用 (理由: 两套匹配逻辑必分叉).
  function matchBatchIdentity(f, batch) {
    const lb = batch || overlay.lastBatch;
    if (!lb || !lb.boxes.length || performance.now() - lb.ts > 2000) return null;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < lb.boxes.length; i++) {
      const b = lb.boxes[i];
      const dx = (f.x + f.w / 2) - (b.x + b.w / 2), dy = (f.y + f.h / 2) - (b.y + b.h / 2);
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    const tol = Math.max(f.w, f.h);
    if (best < 0 || bestD > tol * tol) return null;
    return lb.ids[best] || null;
  }
  function identityFor(f) { return matchBatchIdentity(f, overlay.lastBatch); }

  function drawOverlay(video, canvas) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, vw, vh);
    // 过期自动清空 (人离开/模型停更时不留残影)
    if (overlay.facesTs && performance.now() - overlay.facesTs > 1500) overlay.faces = [];
    if (overlay.handsTs && performance.now() - overlay.handsTs > 1500) overlay.hands = [];
    paintFrame(ctx, vw, vh, overlay.faces, overlay.hands, overlay.lastBatch);
  }

  // 纯绘制核心: video 循环与静态图片共用 (理由: 两套画法必分叉).
  // faces: [{x,y,w,h,score}] 像素坐标; hands: [[{x,y}归一化]]; batch: {boxes, ids, ts}
  function paintFrame(ctx, vw, vh, faces, hands, batch) {
    const showBox = document.getElementById("showBox")?.checked !== false;
    const showHands = document.getElementById("showHands")?.checked !== false;
    const big = faces.slice().sort((a, b) => b.w * b.h - a.w * a.h)[0] || null;
    if (showBox) {
      for (const f of faces) {
        const isBig = big && f === big;
        // 每框独立身份 + 按 STATUS 着色: 批量结果按中心最近邻回填,
        // 还没有结果标 "?" (IDLE灰); 不再用整帧级身份只标最大框
        const id = matchBatchIdentity(f, batch) || { name: "?", sim: 0 };
        const st = STATUS[statusOf(id.name)];
        ctx.strokeStyle = st.box;
        ctx.lineWidth = isBig ? 3 : 2;
        ctx.strokeRect(f.x, f.y, f.w, f.h);
        const detTxt = (f.score ? ` det:${Number(f.score).toFixed(2)}` : "");
        const label = `${id.name}:${Number(id.sim).toFixed(2)}${detTxt}`;
        ctx.font = "16px system-ui";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = st.box;
        ctx.fillRect(f.x, Math.max(0, f.y - 20), tw + 8, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, f.x + 4, Math.max(14, f.y - 5));
      }
    }
    if (showHands) {
      for (const hand of hands) {
        const pts = hand.map((p) => ({ x: p.x * vw, y: p.y * vh }));
        ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2;
        ctx.beginPath();
        for (const [a, b] of HAND_EDGES) {
          if (pts[a] && pts[b]) { ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); }
        }
        ctx.stroke();
        ctx.fillStyle = "#3b82f6";
        for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill(); }
      }
    }
  }

  function overlayLoop(video, canvas) {
    pumpFaces(video);   // 异步,不阻塞
    pumpHands(video);   // 异步,不阻塞
    drawOverlay(video, canvas);
    tickRenderFps();    // 每帧计数, 半秒刷一次 #fps (理由: 写 DOM 比推理便宜但也别每帧写)
    overlay.raf = requestAnimationFrame(() => overlayLoop(video, canvas));
  }

  function startOverlay(video) {
    const canvas = document.getElementById("overlay");
    if (!video || !canvas) return;
    setFaceStatus("人脸检测: 本地模型加载中...");
    setHandStatus("手部模型: 本地模型加载中...");
    if (overlay.raf) cancelAnimationFrame(overlay.raf);
    overlay.raf = requestAnimationFrame(() => overlayLoop(video, canvas));
  }

  function stopOverlay() {
    if (overlay.raf) cancelAnimationFrame(overlay.raf);
    overlay.raf = 0;
    const canvas = typeof document !== "undefined" && document.getElementById("overlay");
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }

  function log(msg) {
    if (typeof document !== "undefined") {
      const el = document.getElementById("log");
      if (el) {
        const line = document.createElement("div");
        line.textContent = new Date().toLocaleTimeString() + " " + msg;
        el.prepend(line);
        while (el.children.length > 50) el.removeChild(el.lastChild);
      }
    }
    console.log("[FrontJS]", msg);
  }

  // 结果面板: 批量列表逐条展示 (k 由发送方已知, 不需要 count)
  function showResult(list, serverTime, rttMs) {
    state.identity = list[0] || { name: "?", sim: 0 };
    if (typeof document === "undefined") return;
    const el = document.getElementById("result");
    if (!el) return;
    el.innerHTML = list.map((r, i) => {
      const st = STATUS[statusOf(r.name)];
      return `<div><b>#${i}</b>` +
        ` <span style="display:inline-block;background:${st.color};color:#fff;font-size:11px;border-radius:3px;padding:1px 5px">${statusOf(r.name)}</span>` +
        ` <span style="color:${st.color};font-weight:bold">${r.name}</span>` +
        ` <span>sim=${Number(r.sim).toFixed(4)}</span></div>`;
    }).join("") +
      `<div style="color:#666">server=${(Number(serverTime) * 1000).toFixed(0)}ms` +
      ` rtt=${Number(rttMs).toFixed(0)}ms faces=${list.length}</div>`;
  }

  async function captureAndInfer(videoEl, canvasEl, opts) {
    if (state.inflight) return; // 上一帧还没回来就跳过 (跳帧节流, 同 solve)
    const video = videoEl || document.getElementById("video");
    const canvas = canvasEl || document.getElementById("canvas");
    if (!video || !canvas || video.readyState < 2) return;
    state.inflight = true;
    try {
      const w = video.videoWidth || opts.captureWidth;
      const h = video.videoHeight || Math.round(w * 3 / 4);
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, w, h);

      // 批量: 本帧全部新鲜人脸各裁一张, 一次 POST {index: img}
      // (理由: 与批量接口对齐, k 脸 1 次往返; 批量下标 = 本批发送顺序)
      const useCrop = document.getElementById("useCrop")?.checked;
      const freshFaces = (useCrop && overlay.facesTs && performance.now() - overlay.facesTs < 1500)
        ? overlay.faces.slice() : [];
      const pad = 20;
      let crops = []; // [{box:{x,y,w,h}, blob}]
      if (freshFaces.length) {
        for (const f of freshFaces) {
          const x = Math.max(0, f.x - pad), y = Math.max(0, f.y - pad);
          const cw = Math.min(w - x, f.w + pad * 2), ch = Math.min(h - y, f.h + pad * 2);
          if (cw <= 0 || ch <= 0) continue;
          const crop = document.createElement("canvas");
          crop.width = cw; crop.height = ch;
          crop.getContext("2d").drawImage(canvas, x, y, cw, ch, 0, 0, cw, ch);
          const blob = await new Promise((res) => crop.toBlob(res, "image/jpeg", opts.jpegQuality));
          if (blob) crops.push({ box: { x, y, w: cw, h: ch }, blob });
        }
      } else if (useCrop && "FaceDetector" in window) {
        try {
          if (!state.faceDetector) state.faceDetector = new FaceDetector({ fastMode: true });
          const faces = await state.faceDetector.detect(canvas);
          if (faces.length > 0) {
            const b = faces[0].boundingBox;
            const crop = document.createElement("canvas");
            const x = Math.max(0, b.x - pad), y = Math.max(0, b.y - pad);
            const cw = Math.min(w - x, b.width + pad * 2), ch = Math.min(h - y, b.height + pad * 2);
            crop.width = cw; crop.height = ch;
            crop.getContext("2d").drawImage(canvas, x, y, cw, ch, 0, 0, cw, ch);
            const blob = await new Promise((res) => crop.toBlob(res, "image/jpeg", opts.jpegQuality));
            if (blob) crops.push({ box: { x, y, w: cw, h: ch }, blob });
          }
        } catch (e) { /* 降级送整帧 */ }
      }
      let blobs, boxes;
      if (crops.length) {
        blobs = crops.map((c) => c.blob);
        boxes = crops.map((c) => c.box);
      } else {
        const whole = await new Promise((res) => canvas.toBlob(res, "image/jpeg", opts.jpegQuality));
        if (!whole) return;
        blobs = [whole]; boxes = []; // 无框时整帧送检, 索引 0
      }
      const r = await inferBatch(blobs, opts.server, opts.minSim);
      state.failCount = 0;
      tickInferFps(); // 成功一次记一次节拍 (理由: 推理 fps 看后端往返, 与渲染 fps 分开)
      // 发送时的框快照 + 按序结果存档, 绘制时按中心最近邻回填 (见 identityFor)
      overlay.lastBatch = { boxes, ids: r.list, ts: performance.now() };
      showResult(r.list, r.serverTime, r.rttMs);
      log(`infer k=${blobs.length} -> ` + r.list.map((o, i) => `#${i} ${o.name}:${Number(o.sim).toFixed(2)}`).join(", "));
    } catch (e) {
      state.failCount = (state.failCount || 0) + 1;
      // 连续失败(比如后端没启动/端口被占)时自动停掉定时器,避免每800ms刷一条Failed to fetch
      if (state.failCount >= 5) {
        if (state.timer) clearInterval(state.timer);
        state.timer = null;
        log(`已连续失败${state.failCount}次,自动暂停连续推理。请点"测试后端连接"排查,修好后再点"打开摄像头并连续推理"。最后错误: ${e.message}`);
      } else {
        log(`推理失败(${state.failCount}/5): ${e.message}`);
      }
    } finally {
      state.inflight = false;
    }
  }

  async function startCamera(videoEl, opts = {}) {
    opts = { ...DEFAULTS, ...opts };
    // 下拉没传 deviceId 时回读 <select>, 保持“选完直接点开始”可用
    if (!opts.deviceId) opts.deviceId = getSelectedDeviceId();
    const video = videoEl || document.getElementById("video");
    // 先停旧流再开新路 (理由: 切换摄像头不关旧 track 会占着设备切不过去)
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    let stream;
    const baseVideo = { width: opts.captureWidth, audio: false };
    if (opts.deviceId) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { ...baseVideo, deviceId: { exact: opts.deviceId } }, audio: false,
        });
      } catch (e) { // exact 失败(如设备被拔)则退回自动, 不直接炸 (理由: 下拉残留 id 比开不了强)
        log(`指定摄像头打开失败, 回退自动选择: ${e.message}`);
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: opts.captureWidth, facingMode: "user" }, audio: false,
        });
      }
    } else {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: opts.captureWidth, facingMode: "user" }, audio: false,
      });
    }
    state.stream = stream;
    state.deviceId = opts.deviceId || "";
    const cv = document.getElementById("canvas");
    if (cv) cv.style.display = "none"; // 切回摄像头时收起静态图
    video.srcObject = stream;
    await video.play();
    const trackLabel = stream.getVideoTracks()[0]?.label || "摄像头";
    // fps 从零起算, 否则切路后沿用旧值误导 (理由: render/infer 都是按流统计)
    fpsMeter.frames = 0; fpsMeter.lastT = performance.now(); fpsMeter.renderFps = 0;
    fpsMeter.inferLastT = 0; fpsMeter.inferFps = 0; setFpsText();
    log(`摄像头已打开 [${trackLabel}], 每 ` + opts.intervalMs + "ms 推理一次 -> " + normalizeServer(opts.server) + INFER_PATH);
    refreshCameraList(); // 授权后 label 才有真名, 开流后刷一次把“摄像头 1”换成真名
    startOverlay(video); // 本地画框: 人脸框(FaceDetector) + 手部点(MediaPipe)
    state.timer = setInterval(() => captureAndInfer(video, null, opts), opts.intervalMs);
    return stream;
  }

  function stopCamera() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    stopOverlay();
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    fpsMeter.renderFps = 0; fpsMeter.inferFps = 0; setFpsText(); // 停流即清零, 不留旧 fps 误导
    log("摄像头已关闭");
  }

  function currentOpts() {
    if (typeof document === "undefined") return { ...DEFAULTS };
    return {
      server: document.getElementById("server")?.value.trim() || DEFAULTS.server,
      minSim: parseFloat(document.getElementById("minSim")?.value ?? DEFAULTS.minSim),
      intervalMs: parseInt(document.getElementById("interval")?.value ?? DEFAULTS.intervalMs, 10),
      jpegQuality: DEFAULTS.jpegQuality,
      captureWidth: DEFAULTS.captureWidth,
      deviceId: getSelectedDeviceId(),
    };
  }

  // 供 front.html 按钮调用
  async function onStart() { try { await startCamera(null, currentOpts()); } catch (e) { log("打开摄像头失败: " + e.message + " (若下拉为空请先点“刷新摄像头”; 自签页先过 /docs 信任)"); } }
  async function onRefreshCameras() { await refreshCameraList(); }
  // 切换下拉后若已在推流则直接换路重开 (理由: 选完再点开始多一步, 直接换最顺手; 未开流则只记住选择)
  async function onCameraChange() {
    state.deviceId = getSelectedDeviceId();
    if (!state.stream) return;
    try { await startCamera(null, currentOpts()); }
    catch (e) { log("切换摄像头失败: " + e.message); }
  }
  async function onTest() {
    const o = currentOpts();
    log("正在测试 " + normalizeServer(o.server) + "/openapi.json ...");
    try {
      await testConnection(o.server);
      log("后端连接正常: " + normalizeServer(o.server));
    } catch (e) {
      log("后端连接失败: " + e.message + " —— 请确认后端已启动(uvicorn app:app --host 0.0.0.0 --port 8060)且端口未被占用");
    }
  }
  // 静态图片管线: 显示原图 -> 本地检测(框+手) -> 绘制 -> k 脸一次 inferBatch -> 名牌回填.
  // 检测/绘制/STATUS/匹配全部复用 video 那套, 数据源换成 image canvas (理由: 两套必分叉).
  // 检测跑在离屏 base 上而非可见 canvas (理由: 可见画布一旦画上框, 下一轮检测会把框当人脸).
  // 路由保持摄像头优先 (不动): 本函数只在没开摄像头时被 onSingle 调用.
  async function processImageFile(f, opts) {
    const canvas = document.getElementById("canvas");
    let bmp;
    try {
      bmp = await createImageBitmap(f);
    } catch (e) { log("图片解码失败: " + e.message); return; }
    const maxSide = 1280; // 超长边等比缩 (理由: 大图对 BlazeFace 与传输无收益, 只烧时间)
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const base = document.createElement("canvas");
    base.width = w; base.height = h;
    base.getContext("2d").drawImage(bmp, 0, 0, w, h);
    if (bmp.close) bmp.close();
    canvas.width = w; canvas.height = h;
    canvas.style.display = "block";
    const ctx = canvas.getContext("2d");
    const blit = (faces, hands, batch) => {
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(base, 0, 0);
      paintFrame(ctx, w, h, faces, hands, batch);
    };
    let faces = [], hands = [];
    try {
      const mp = await ensureFaceDetectorMp();
      if (mp) {
        const res = mp.detectForVideo(base, nextTs()); // VIDEO 模式照收 canvas 帧
        faces = (res.detections || []).map((d) => {
          const b = d.boundingBox;
          return { x: b.originX, y: b.originY, w: b.width, h: b.height,
                   score: (d.categories && d.categories[0]) ? d.categories[0].score : 0 };
        });
      } else if ("FaceDetector" in window) {
        if (!state.faceDetector) state.faceDetector = new FaceDetector({ fastMode: true });
        const dets = await state.faceDetector.detect(base);
        faces = dets.map((d) => { const b = d.boundingBox; return { x: b.x, y: b.y, w: b.width, h: b.height, score: 0 }; });
      } else {
        log("本地人脸模型未就绪, 本次只做整图识别");
      }
    } catch (e) { log("图片人脸检测失败(继续整图识别): " + e.message); }
    if (faces.length) {
      const areas = faces.map((x) => Math.round(x.w * x.h));
      log(`图片人脸检测: ${faces.length} 张, 框面积px²=[${areas.join(",")}] (图 ${w}x${h})`);
    } else {
      log(`图片人脸检测: 0 张, 降级整图送检 (图 ${w}x${h}; 合照人脸小可能是短距模型+0.5阈值滤掉了)`);
    }
    try {
      const lm = await ensureHandLandmarker();
      if (lm) {
        const res = lm.detectForVideo(base, nextTs());
        const hh = res && res.landmarks ? res.landmarks : [];
        hands = (hh || []).map((hand) => hand.map((p) => ({ x: p.x, y: p.y })));
      }
    } catch (e) { log("图片手部检测失败(继续识别): " + e.message); }
    log(`图片手部检测: ${hands.length} 只手`);
    const pad = 20, crops = [];
    for (const fc of faces) {
      const x = Math.max(0, fc.x - pad), y = Math.max(0, fc.y - pad);
      const cw = Math.min(w - x, fc.w + pad * 2), ch = Math.min(h - y, fc.h + pad * 2);
      if (cw <= 0 || ch <= 0) continue;
      const crop = document.createElement("canvas");
      crop.width = cw; crop.height = ch;
      crop.getContext("2d").drawImage(base, x, y, cw, ch, 0, 0, cw, ch);
      const blob = await new Promise((res) => crop.toBlob(res, "image/jpeg", opts.jpegQuality));
      if (blob) crops.push({ box: { x, y, w: cw, h: ch }, blob });
    }
    let blobs, boxes;
    if (crops.length) {
      blobs = crops.map((c) => c.blob);
      boxes = crops.map((c) => c.box);
    } else {
      const whole = await new Promise((res) => base.toBlob(res, "image/jpeg", opts.jpegQuality));
      if (!whole) { blit(faces, hands, { boxes: [], ids: [], ts: 0 }); return; }
      blobs = [whole]; boxes = []; // 无框时整图 1 张, 索引 0
    }
    blit(faces, hands, { boxes: [], ids: [], ts: performance.now() }); // 先画框手+?名牌, 不等后端
    try {
      const r = await inferBatch(blobs, opts.server, opts.minSim);
      showResult(r.list, r.serverTime, r.rttMs);
      blit(faces, hands, { boxes, ids: r.list, ts: performance.now() }); // 回填名牌重画
      log(`图片推理 k=${blobs.length} -> ` + r.list.map((o, i) => `#${i} ${o.name}:${Number(o.sim).toFixed(2)}`).join(", "));
    } catch (e) {
      log("推理失败: " + e.message);
    }
  }

  async function onSingle() {
    const o = currentOpts();
    const video = document.getElementById("video");
    if (!video || !state.stream) { // 没开摄像头则走静态图片管线 (路由保持摄像头优先, 不动)
      const f = document.getElementById("file")?.files?.[0];
      if (!f) return log("请先打开摄像头或选择图片");
      await processImageFile(f, o);
      return;
    }
    await captureAndInfer(video, null, o);
  }

  const FrontJS = { inferBlob, inferBatch, testConnection, normalizeServer, getLocalIp, startCamera, stopCamera, captureAndInfer, startOverlay, stopOverlay, processImageFile, onStart, onTest, onSingle, listCameras, refreshCameraList, onRefreshCameras, onCameraChange, DEFAULTS };
  global.FrontJS = FrontJS;

  // 页面加载即枚举一次 (无权限时 label 为空先显示编号, 授权/开流后会再刷真名)
  // readyState 兜底: 若脚本执行时 DOM 已解析完 (缓存/回退场景), DOMContentLoaded 不会再触发,
  // 之前只监听事件就会导致下拉永远“加载中” —— 这里双保险
  function initCameraUI() {
    try { setFpsText(); } catch (_) {}
    refreshCameraList().catch((e) => {
      try { log("初始化摄像头列表失败: " + (e && e.message ? e.message : e)); } catch (_) {}
    });
  }
  if (typeof document !== "undefined" && typeof window !== "undefined") {
    if (document.readyState === "loading") {
      window.addEventListener("DOMContentLoaded", initCameraUI);
    } else {
      initCameraUI();
    }
  }

  // ---- Node CLI: node Front/front.js <image.jpg> ----
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  if (isNode && typeof require !== "undefined" && require.main === module) {
    (async () => {
      const fs = require("fs");
      const path = require("path");
      const argv = process.argv.slice(2);
      const getArg = (k, d) => {
        const i = argv.indexOf(k);
        return i >= 0 && argv[i + 1] ? argv[i + 1] : d;
      };
      const imgPath = argv.find((a) => !a.startsWith("--"));
      const server = getArg("--server", "https://" + (getLocalIp() || "127.0.0.1") + ":8060");
      const minSim = parseFloat(getArg("--min-sim", "0.4"));
      if (!imgPath) {
        console.log("用法: node Front/front.js <image.jpg> [--server URL] [--min-sim 0.4]");
        process.exit(1);
      }
      const buf = fs.readFileSync(imgPath);
      const blob = new Blob([buf], { type: "image/jpeg" });
      blob.name = path.basename(imgPath);
      try {
        const r = await inferBatch([blob], server, minSim);
        r.list.forEach((one, i) => console.log(`${imgPath} #${i} -> ${one.name}:${Number(one.sim).toFixed(4)}`));
        console.log(`server=${(r.serverTime * 1000).toFixed(0)}ms rtt=${r.rttMs.toFixed(0)}ms faces=${r.list.length}`);
      } catch (e) {
        console.error("推理失败:", e.message);
        process.exit(1);
      }
    })();
  }
})(typeof window !== "undefined" ? window : globalThis);
