import sys
import os

import cv2
import numpy as np
import onnxruntime as ot
import mediapipe as mp
from numpy.typing import NDArray

from typing import Any, List, Union, Dict, Tuple, Optional

FaceBlob = NDArray[np.float32]       # (N,3,112,112) 输入
FaceEmbedding = NDArray[np.float32]  # (N,512) 已L2归一输出
FaceImage = NDArray[np.uint8]        # (H,W,3) BGR原图


def _make_ort_session(model_path: str, enable_mem_pattern: bool = False) -> ot.InferenceSession:
    """CPU优先、避免CUDA缺失刷屏；开满线程。

    enable_mem_pattern=False 给 SCRFD 这类动态 shape 模型用，避免
    VerifyOutputSizes 刷屏 + 每次重分配；112x112 固定输入的识别模型可传 True。
    """
    # 只显示 Error，屏蔽 VerifyOutputSizes 的 Expected shape ... does not match 警告
    try:
        ot.set_default_logger_severity(3)
    except Exception:
        pass
    opts = ot.SessionOptions()
    try:
        opts.log_severity_level = 3
    except Exception:
        pass
    try:
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    except Exception:
        pass
    opts.enable_mem_pattern = enable_mem_pattern
    # 先看机器真正可用的provider，缺cudnn时直接只用CPU，避免每次run都报CUDA错
    try:
        avail = ot.get_available_providers()
    except Exception:
        avail = ["CPUExecutionProvider"]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    # 即使avail里有CUDA，缺cudnn也会在创建时失败，这里再兜底一次
    try:
        return ot.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception:
        return ot.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


class FaceRecognition:
    def __init__(self, faceRecognitionModelPath: str = r"Model\buffalo_l\w600k_r50.onnx", faceEmbeddingsPath: str = r"Data\FaceEmbedding\test0926.npz"):
        # 112x112 固定输入，开 mem_pattern 没问题
        self.faceRecognitionModel = _make_ort_session(faceRecognitionModelPath, enable_mem_pattern=True)

        self.faceEmbeddingsPath=faceEmbeddingsPath

        self.faceModelInputName = self.faceRecognitionModel.get_inputs()[0].name  # 获取模型输入张量名，用于传入ort
        # faceEmbeddingsPath 可能是 .npy 文件 / 目录 / 缺失, 做容错: 空库时 match 直接返回 unknow
        self.G = np.zeros((0, 512), dtype=np.float32)
        self.ids: List[str] = []
        try:
            if os.path.isdir(self.faceEmbeddingsPath):
                # 目录: 尝试找里面的 .npy/.npz, 没有则保持空库
                cands = [os.path.join(self.faceEmbeddingsPath, f) for f in os.listdir(self.faceEmbeddingsPath)
                         if f.endswith(".npy") or f.endswith(".npz")]
                if len(cands) == 0:
                    print(f"[FaceRecognition] {self.faceEmbeddingsPath} 为空目录, 使用空人脸库")
                    self.faceEmbeddings = {}
                else:
                    self.faceEmbeddings = np.load(cands[0], allow_pickle=True)
            else:
                self.faceEmbeddings = np.load(self.faceEmbeddingsPath, allow_pickle=True)
            # dict: {id: (K,512)} 或单数组
            if hasattr(self.faceEmbeddings, "files"): # .npz
                items = [(k, self.faceEmbeddings[k]) for k in self.faceEmbeddings.files]
            elif isinstance(self.faceEmbeddings, dict):
                items = self.faceEmbeddings.items()
            else: # .npy 存 dict
                items = self.faceEmbeddings.item().items()
            mats, ids = [], []
            for k, arr in items:
                a = np.asarray(arr, dtype=np.float32).reshape(-1, 512)
                a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
                mats.append(a)
                ids += [k] * len(a)
            if mats:
                self.G = np.vstack(mats).astype(np.float32)  # Mx512，已归一
                self.ids = ids
        except FileNotFoundError:
            print(f"[FaceRecognition] 未找到 {self.faceEmbeddingsPath}, 使用空人脸库")
        except PermissionError:
            print(f"[FaceRecognition] {self.faceEmbeddingsPath} 不可读, 使用空人脸库")
        except Exception as e:
            print(f"[FaceRecognition] 加载人脸库失败 ({e}), 使用空人脸库")

    def getFaceTensor(self, faceImg: FaceImage) -> FaceEmbedding:
        # 兼容任意裁剪尺寸，统一 resize 到 112x112 (ArcFace/R50 输入)
        if faceImg.shape[0] != 112 or faceImg.shape[1] != 112:
            faceImg = np.asarray(cv2.resize(faceImg, (112, 112)), dtype=np.uint8)
        blob = ((faceImg.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        v = np.array(self.faceRecognitionModel.run(None, {self.faceModelInputName: blob})[0])[0]

        return (v / np.linalg.norm(v)).astype(np.float32)

    def match(self, embedding: FaceEmbedding, min_sim=0.4) -> tuple[str, float]:
        if self.G.shape[0] == 0:
            return ("stranger", 0.0)
        q = embedding / (np.linalg.norm(embedding) + 1e-9)
        sims = self.G @ q.astype(np.float32)  # M,
        i = int(sims.argmax())
        return (self.ids[i], float(sims[i])) if sims[i] > min_sim else ("stranger", float(sims[i]))

    def add(self, person_id: str, emb: FaceEmbedding):
        q = np.asarray(emb, dtype=np.float32).reshape(1, 512)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        self.G = np.vstack([self.G, q]).astype(np.float32)
        self.ids.append(person_id)

    def save(self, path: Optional[str] = None):
        if path is None:
            path=self.faceEmbeddingsPath
        db: dict[str, list] = {}
        for pid, vec in zip(self.ids, self.G):
            db.setdefault(pid, []).append(vec)
        db_arr = {k: np.stack(v).astype(np.float32) for k, v in db.items()}
        np.savez_compressed(path, **{str(k): np.asanyarray(v) for k, v in db_arr.items()})  # {id: (K,512)}



class _TasksHandsCompat:
    """mediapipe>=1.0 兼容层: 对外仍提供 .process(img_rgb) 接口.

    内部使用 tasks.vision.HandLandmarker (IMAGE 模式, 逐帧 detect).
    返回 SimpleNamespace(multi_hand_landmarks, handedness, raw),
    其中 multi_hand_landmarks 兼容旧版 solutions 结果的命名.
    """
    def __init__(self, model_path: str = r"Model\hand_landmarker.task",
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 running_mode: str = "VIDEO"):
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        self.model_path = model_path
        self._mp_vision = mp_vision
        self._BaseOptions = BaseOptions
        self._VisionTaskRunningMode = VisionTaskRunningMode
        self._params = dict(max_num_hands=max_num_hands,
                            min_detection_confidence=min_detection_confidence,
                            min_tracking_confidence=min_tracking_confidence)
        self.running_mode_str = running_mode.upper()
        self.landmarker = None
        self._init_error: Optional[Exception] = None
        self._video_ts_ms = 0

        if not os.path.exists(model_path):
            # 尝试自动下载官方模型, 失败则降级为占位模式 (不阻塞人脸流程)
            try:
                import urllib.request
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                print(f"[_TasksHandsCompat] 未找到 {model_path}, 正在下载...")
                urllib.request.urlretrieve(url, model_path)
                print(f"[_TasksHandsCompat] 下载完成: {model_path}")
            except Exception as e:
                self._init_error = e
                print(f"[_TasksHandsCompat] 手部模型缺失且下载失败 ({e}), 将以空结果占位运行.")
                print("手动下载: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
                return
        try:
            mode = VisionTaskRunningMode.VIDEO if self.running_mode_str == "VIDEO" else VisionTaskRunningMode.IMAGE
            base = BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        except Exception as e:
            self._init_error = e
            print(f"[_TasksHandsCompat] HandLandmarker 初始化失败: {e}")

    def process(self, img_rgb: np.ndarray, timestamp_ms: Optional[int] = None):
        import types
        import time
        if self.landmarker is None:
            # 占位空结果
            return types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None, handedness=None, raw=None)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
        # VIDEO模式用tracking，比IMAGE每帧重检快很多；时间戳必须单调递增
        if self.running_mode_str == "VIDEO":
            if timestamp_ms is None:
                self._video_ts_ms += 33#模拟30fps
                timestamp_ms = self._video_ts_ms
            else:
                self._video_ts_ms = max(self._video_ts_ms + 1, timestamp_ms)
                timestamp_ms = self._video_ts_ms
            res = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            res = self.landmarker.detect(mp_image)
        # res.hand_landmarks: List[List[NormalizedLandmark]]; res.handedness: List[List[Category]]
        out = types.SimpleNamespace(
            multi_hand_landmarks=getattr(res, "hand_landmarks", None),
            multi_handedness=getattr(res, "handedness", None),
            handedness=getattr(res, "handedness", None),
            raw=res,
        )
        return out

    def close(self):
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass


class FaceCheckWithMediapipe:
    def __init__(self, faceCheckModelPath: str = r"Model\SCRFD\det_500m.onnx",

                 static_image_mode=False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,

                 input_size: Tuple[int, int] = (320, 320),  # (h, w)，320比640快约4倍
                 det_thresh: float = 0.5,
                 nms_thresh: float = 0.4,
                 pre_nms_topk: int = 1000,
                 ):
        # det_2.5g(2.5GFlops)换det_500m(0.5GFlops)约快3-5倍；要精度再换回2.5g
        # NOTE: det_*.onnx 按 640x640 导出(shape推理里是12800/3200/800)，
        # 这里默认用 320x320 提速(实际是3200/800/200)，功能正常但 ORT 会报
        # VerifyOutputSizes 警告，已在 _make_ort_session 里屏蔽；想彻底消除就传 input_size=(640,640)
        # 动态 shape 必须关 mem_pattern，否则每次重分配更慢
        self.faceCheckModel = _make_ort_session(faceCheckModelPath, enable_mem_pattern=False)
        self.faceModelInputName = self.faceCheckModel.get_inputs()[0].name  # 获取模型输入张量名，用于传入ort

        # ---- Mediapipe Hands: 兼容旧版 solutions 和新版 tasks ----
        # 要求: 对外统一为 self.hands.process(img_rgb)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):  # mediapipe==0.10.x
            self.mp_hands = mp.solutions.hands  # type: ignore
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self._hands_backend = "solutions"
        else:  # mediapipe>=1.0, 如 cuda128 环境的 1.0.1
            self.mp_hands = None
            self.hands = _TasksHandsCompat(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode="VIDEO",  # VIDEO跟踪比IMAGE每帧重检快
            )
            self._hands_backend = "tasks"

        self.FaceRecognition = FaceRecognition()

        # ---- SCRFD 解码参数 ----
        self.input_size = input_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2
        self._anchor_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _get_anchors(self, Ht: int, Wt: int, stride: int) -> np.ndarray:
        key = (Ht, Wt, stride)
        a = self._anchor_cache.get(key)
        if a is None:
            anchor_centers = np.stack(np.mgrid[:Ht, :Wt][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            a = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
            self._anchor_cache[key] = a
        return a

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
        inter = w * h
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / (ua + 1e-9)

    # ---------- 工具: NMS ----------
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    # ---------- 用 self.faceCheckModel 获取人脸框 ----------
    def detect_face(self, img_bgr: FaceImage) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """SCRFD 前向 + 解码.

        Returns:
            bboxes: (N,5) [x1,y1,x2,y2,score] (原图坐标)
            kpss: (N,5,2) 5关键点 (原图坐标) or None
        """
        h, w = img_bgr.shape[:2]
        model_h, model_w = self.input_size

        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))

        det_img = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = ((det_img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        # outs: [s8,s16,s32, b8,b16,b32, k8,k16,k32]
        outs = self.faceCheckModel.run(None, {self.faceModelInputName: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self.feat_strides):
            scores = outs[idx].ravel()
            # 注意: det_*.onnx 直接输出 0~1 概率 (空白图实测 0.001~0.035),
            # 若再做 sigmoid 会全变成 ~0.5 导致误检, 所以这里不做 sigmoid
            bbox_pred = outs[idx + 3] * stride  # (N,4) l,t,r,b 距离
            kps_pred = outs[idx + 6] * stride   # (N,10)

            Ht, Wt = model_h // stride, model_w // stride
            anchor_centers = self._get_anchors(Ht, Wt, stride)

            mask = scores > self.det_thresh
            if not np.any(mask):
                continue
            scores_f = scores[mask]
            bbox_f = bbox_pred[mask]
            kps_f = kps_pred[mask]
            anchors_f = anchor_centers[mask]
            # 预截断：避免NMS的O(N^2)爆炸
            if scores_f.shape[0] > self.pre_nms_topk:
                topk = np.argpartition(scores_f, -self.pre_nms_topk)[-self.pre_nms_topk:]
                scores_f = scores_f[topk]
                bbox_f = bbox_f[topk]
                kps_f = kps_f[topk]
                anchors_f = anchors_f[topk]

            x1 = anchors_f[:, 0] - bbox_f[:, 0]
            y1 = anchors_f[:, 1] - bbox_f[:, 1]
            x2 = anchors_f[:, 0] + bbox_f[:, 2]
            y2 = anchors_f[:, 1] + bbox_f[:, 3]

            kps = np.zeros((kps_f.shape[0], 5, 2), dtype=np.float32)
            for j in range(5):
                kps[:, j, 0] = anchors_f[:, 0] + kps_f[:, j * 2]
                kps[:, j, 1] = anchors_f[:, 1] + kps_f[:, j * 2 + 1]

            boxes = np.stack([x1, y1, x2, y2, scores_f], axis=1)
            scores_list.append(scores_f)
            bboxes_list.append(boxes)
            kpss_list.append(kps)

        if len(bboxes_list) == 0:
            return np.zeros((0, 5), dtype=np.float32), None

        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        keep = self._nms(bboxes[:, :4], bboxes[:, 4], self.nms_thresh)
        bboxes = bboxes[keep]
        kpss = kpss[keep]

        # 映射回原图
        bboxes[:, :4] /= scale
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w - 1)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h - 1)
        kpss /= scale
        kpss[:, :, 0] = kpss[:, :, 0].clip(0, w - 1)
        kpss[:, :, 1] = kpss[:, :, 1].clip(0, h - 1)

        return bboxes.astype(np.float32), kpss.astype(np.float32)


    def detect_hands(self, img_bgr: FaceImage) -> List[Dict[str, Any]]:
            """独立手势识别方法 (供外部demo调用, 不依赖solve循环).
    
            Args:
                img_bgr: (H,W,3) BGR原图.
    
            Returns:
                List[Dict], 每个元素:
                  {
                    "hand_index": int,
                    "handedness": "Right"/"Left"/"Unknown",
                    "handedness_score": float,
                    "landmarks_norm": [{"x","y","z"} x21] (归一化0~1),
                    "index_tip_norm": {"x","y"} (landmark[8] 快捷, 供鼠标demo用),
                  }
                无手时返回 [].
            """
            import time
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            try:
                try:
                    hands_results = self.hands.process(img_rgb, timestamp_ms=int(time.time() * 1000))
                except TypeError:
                    hands_results = self.hands.process(img_rgb)
            except Exception:
                return []
            lms_list = getattr(hands_results, "multi_hand_landmarks", None)
            if not lms_list:
                return []
            handed_list = getattr(hands_results, "multi_handedness", None)
            if handed_list is None:
                handed_list = getattr(hands_results, "handedness", None)
    
            out: List[Dict[str, Any]] = []
            for i, hand_lms in enumerate(lms_list):
                # ---- handedness 解析 (兼容 solutions/tasks) ----
                label, score = "Unknown", 0.0
                try:
                    if handed_list is not None and i < len(handed_list):
                        h = handed_list[i]
                        if hasattr(h, "classification"):  # solutions
                            c = h.classification[0]
                            label = str(getattr(c, "label", "Unknown"))
                            score = float(getattr(c, "score", 0.0))
                        elif isinstance(h, (list, tuple)) and len(h) > 0:  # tasks
                            c = h[0]
                            label = str(getattr(c, "category_name", None)
                                        or getattr(c, "label", None)
                                        or getattr(c, "display_name", "Unknown"))
                            score = float(getattr(c, "score", 0.0))
                        else:  # 单个 Category
                            label = str(getattr(h, "category_name", "Unknown"))
                            score = float(getattr(h, "score", 0.0))
                except Exception:
                    pass
                # ---- landmarks 解析 (兼容 solutions/tasks) ----
                try:
                    pts = hand_lms.landmark if hasattr(hand_lms, "landmark") else hand_lms
                    lms = [{"x": float(p.x), "y": float(p.y), "z": float(getattr(p, "z", 0.0))} for p in pts]
                    if len(lms) != 21:
                        continue
                except Exception:
                    continue
                out.append({
                    "hand_index": i,
                    "handedness": label,
                    "handedness_score": score,
                    "landmarks_norm": lms,
                    "index_tip_norm": {"x": lms[8]["x"], "y": lms[8]["y"]},
                })
            return out

    

    def solve(self, camera_id: int = 0,
                detect_interval: int = 3, hands_interval: int = 4,recognition_interval:int=3,
                cam_width: int = 640, cam_height: int = 480,
                show_fps: bool = True, iou_reuse_thresh: float = 0.6):
        """获取摄像头，每帧:
        1. 用 self.faceCheckModel 获取人脸框 -> 裁剪 -> self.FaceRecognition 获取人脸结果
        2. 用 Mediapipe 获取 self.hands.process(img_rgb) 的结果
        (对这些结果的处理暂时不写, 这里只获取 + 最简可视化)

        速度优化: 检测/手部都跳帧复用 + 人脸IoU复用 + 320输入 + 500m模型 + VIDEO跟踪。
        之前卡的原因见控制台打印的耗时分解。
        """
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[solve] 无法打开摄像头 {camera_id}")
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 丢旧帧，降延迟
        except Exception:
            pass

        print(f"[solve] 按 q 退出 | detect每{detect_interval}帧 hands每{hands_interval}帧 input={self.input_size}")
        import time
        frame_idx = 0
        bboxes = np.zeros((0, 5), dtype=np.float32)
        kpss = None
        face_results: List[Dict[str, Any]] = []
        hands_results = None
        t_prev = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[solve] 读取帧失败, 退出")
                break
            frame_idx += 1
            t0 = time.time()

            # ---- 1. 人脸检测 (跳帧) ----
            t_det = 0.0
            if frame_idx == 1 or ((frame_idx - 1) % detect_interval == 0):
                ta = time.time()
                bboxes, kpss = self.detect_face(frame)  # 内部调用 self.faceCheckModel
                t_det = (time.time() - ta) * 1000

            # ---- 2. 裁剪对应图像 + 人脸识别 (IoU复用: 框没大动就不重算embedding) ----
            t_rec = 0.0
            new_face_results: List[Dict[str, Any]] = []
            ta = time.time()
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[i]
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                if x2i <= x1i or y2i <= y1i:
                    continue
                # 找上一轮最匹配的框，IoU够大直接复用身份
                reused = None
                for fr in face_results:
                    ox1, oy1, ox2, oy2 = fr["bbox"]
                    if self._bbox_iou(np.array([x1i, y1i, x2i, y2i]), np.array([ox1, oy1, ox2, oy2])) > iou_reuse_thresh:
                        reused = fr
                        break
                if reused is not None and ((frame_idx - 1) % detect_interval != 0):
                    new_face_results.append({
                        "bbox": (x1i, y1i, x2i, y2i),
                        "det_score": float(score),
                        "name": reused["name"], "sim": reused["sim"],
                        "embedding": reused["embedding"],
                    })
                    continue
                
                face_crop = frame[y1i:y2i, x1i:x2i]
                if face_crop.size == 0:
                    continue
                face_crop_112 = cv2.resize(face_crop, (112, 112))
                emb = self.FaceRecognition.getFaceTensor(face_crop_112)
                name, sim = self.FaceRecognition.match(emb)
                new_face_results.append({
                    "bbox": (x1i, y1i, x2i, y2i),
                    "det_score": float(score),
                    "name": name,
                    "sim": float(sim),
                    "embedding": emb,
                })
            face_results = new_face_results
            t_rec = (time.time() - ta) * 1000

            # ---- 3. Mediapipe 手部 (跳帧) ----
            t_hand = 0.0
            if frame_idx == 1 or ((frame_idx - 1) % hands_interval == 0):
                ta = time.time()

                hands_result=self.detect_hands(frame)

            # NOTE: 对 face_results / hands_results / kpss 的后续处理暂时不写
            # TODO: 在这里处理人脸结果和手部结果

            # ---- 最简可视化 ----
            for fr in face_results:
                x1i, y1i, x2i, y2i = fr["bbox"]
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(frame, f"{fr['name']}:{fr['sim']:.2f}",
                            (x1i, max(0, y1i - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 手部点(新版tasks结果在multi_hand_landmarks里)
            try:
                lms_list = getattr(hands_results, "multi_hand_landmarks", None) if hands_results is not None else None
                if lms_list:
                    hh, ww = frame.shape[:2]
                    for hand_lms in lms_list:
                        pts = [(int(p.x * ww), int(p.y * hh)) for p in hand_lms]
                        for (x, y) in pts:
                            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            except Exception:
                pass

            now = time.time()
            inst = 1.0 / max(1e-6, now - t_prev)
            fps = fps * 0.9 + inst * 0.1
            t_prev = now
            if show_fps:
                cv2.putText(frame, f"FPS:{fps:.1f} det:{t_det:.0f}ms rec:{t_rec:.0f}ms hand:{t_hand:.0f}ms",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if frame_idx % 30 == 1:
                print(f"[t] fps={fps:.1f} det={t_det:.0f}ms rec={t_rec:.0f}ms hand={t_hand:.0f}ms faces={len(face_results)}")

            cv2.imshow("FaceCheckWithMediapipe - q to quit", frame)

            key=cv2.waitKey(1) & 0xFF

            if  key == ord('q'):
                break
            elif key == ord('s') and face_results:
                print("name : ",end=" ")
                name=str(input().split()[0])
                faceList=[]
                print("set n to quit FaceSave and set y to save")
                for _idx in range(len(face_results)):
                    fr=face_results[_idx]
                    x1i, y1i, x2i, y2i = fr["bbox"]
                    faceImg = frame[y1i:y2i, x1i:x2i]
                    tempFaceImg=np.zeros((640, 640, 3), dtype=np.uint8)
                    h, w = faceImg.shape[:2]
                    y0, x0 = 320 - h//2, 320 - w//2
                    tempFaceImg[y0:y0+h, x0:x0+w] = faceImg
                    cv2.imshow(f"FaceImg{_idx}", tempFaceImg)
                    faceList.append(faceImg)
                ikey=cv2.waitKey(0) & 0xFF
                if  ikey == ord('n'):
                    for fIdx in range(len(faceList)):
                        cv2.destroyWindow(f"FaceImg{fIdx}")
                    continue
                elif ikey == ord('y'):
                    faceListIdx=int(input().split()[0])
                    emb=self.FaceRecognition.getFaceTensor(faceList[faceListIdx])
                    try:
                        self.FaceRecognition.add(name,emb=emb)
                    except Exception:
                        print("failed to save faceTensor")
                    print(f"succes to save FaceImg{faceListIdx} as {name}")

                    for fIdx in range(len(faceList)):
                        cv2.destroyWindow(f"FaceImg{fIdx}")
                    

        cap.release()
        cv2.destroyAllWindows()
        self.FaceRecognition.save()


if __name__ == "__main__":
    checker = FaceCheckWithMediapipe()
    checker.solve()
