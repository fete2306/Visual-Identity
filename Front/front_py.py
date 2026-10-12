"""
Front/front_py.py — Python 版前端 (参考 base1.FaceCheckWithMediapipe, 识别走后端 app.py /infer/batch)

逻辑对照 base1.FaceCheckWithMediapipe.solve():
  本地做: 摄像头采集 / SCRFD detect() / Mediapipe hands.process() / IoU复用 / 跳帧 / 可视化
  远端做: 原来 solve() 里 self.FaceRecognition.getFaceTensor+match 的部分,
          改为把人脸 crop JPEG 编码后 POST 到后端 /infer/batch, 用返回的 (name, sim) 显示.

后端接口 (app.py /infer/batch, 按 app.py 原样引用, 不改后端):
  POST {server}/infer/batch?min_sim=0.4  multipart, 字段 {index: img}:
    第 i 张人脸 crop 以字段名 str(i) 发送 ("0".."k-1"), 顺序即索引.
  (/infer/single 为单图端, 前端统一走批量, k=1 也是合法批量.)
  返回: {"time": float, "results": {"0": [name, sim], "1": [name, sim], ...}}
  (解析另兼容旧版单体 results=[name, sim]; k 由发送方已知, 不需要 count.)

 用法 (统一用 cuda128 环境, cwd=项目根 Https 版):
   D:\\conda\\envs\\cuda128\\python.exe Https\Front\front_py.py --help
   启动后端:  D:\\conda\\envs\\cuda128\\python.exe -m uvicorn Https.app:app --host 0.0.0.0 --port 8060 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
   列出摄像头: D:\\conda\\envs\\cuda128\\python.exe Https\Front\front_py.py --list-cameras
   单图测试:  D:\\conda\\envs\\cuda128\\python.exe Https\Front\front_py.py --image Data/test.jpg --server https://127.0.0.1:8060
   摄像头:    D:\\conda\\envs\\cuda128\\python.exe Https\Front\front_py.py --server https://127.0.0.1:8060 --camera 0
   纯远端模式(本机无模型/只送整帧): --remote-only
   自签验签默认用 ../certs/cert.pem, 对不上加 --no-verify 跳过.

 可视化: 绿色框=人脸检测框(标签 name:sim), 蓝色点线=手部21关键点骨架.
 帧率: 默认叠加 FPS/det/infer/hand (对标 base1.solve show_fps, --no-show-fps 可关).
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

try:
    import requests
except ImportError:
    print("需要 requests: pip install requests")
    raise

from Back import FaceCheckWithMediapipe

# 让 Front/ 下的脚本能 import 到项目根目录的 base1 / app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 21点手部骨架连线 (与 mediapipe solutions HAND_CONNECTIONS 一致, tasks后端同样适用)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


# 身份状态预留字段字典 (与后端保留标签对齐, BGR; 后端新增保留字段只需加映射)
# INFO绿=已识别 / WARNING黄=未知(unknow) / ERROR红=跳过失败(outOfrecognitionCount) / IDLE灰=无结果
STATUS = {
    "INFO": {"color": (0, 200, 0), "label": "已识别"},
    "WARNING": {"color": (0, 200, 200), "label": "未知"},
    "ERROR": {"color": (0, 0, 220), "label": "跳过/失败"},
    "IDLE": {"color": (160, 160, 160), "label": "无结果"},
}


def status_of(name: str) -> str:
    if name == "outOfrecognitionCount":
        return "ERROR"
    if name == "unknow":
        return "WARNING"
    if not name or name == "?":
        return "IDLE"
    return "INFO"


def normalize_server(server: str) -> str:
    """输入框可能粘了 .../infer/batch 等完整地址甚至带 ?min_sim, 统一归一化成裸 host."""
    # 去后缀必须长串优先 (先 batch/single 再 infer), 否则 ".../infer/batch" 会剩 "/batch".
    s = (server or "").strip().split("?")[0].rstrip("/")
    for suffix in ("/infer/batch", "/infer/single", "/infer"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break
    return s or "https://127.0.0.1:8060"


def get_lan_ip() -> str:
    """首选 psutil.net_if_addrs().values() 取首个非回环IPv4, 失败回退 socket."""
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs or []:
                fam = getattr(a.family, "value", a.family)
                try:
                    import socket as _so
                    is_v4 = (fam == _so.AF_INET) or (fam == 2)
                except Exception:
                    is_v4 = (fam == 2)
                if not is_v4:
                    continue
                ip = a.address or ""
                if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                    return ip
    except Exception:
        pass
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def list_cameras(max_index: int = 8):
    """枚举 0..max_index-1, 返回 [(index, 宽, 高, 可读帧)] (理由: OpenCV 无直接枚举 API, 逐个试开最可靠).

    先 CAP_DSHOW (Windows 原生, 快), 失败回退默认后端. 每个 index 试读一帧,
    能打开但读不到帧的仍保留 (可能被占用), 由调用方提示.
    """
    found = []
    for i in range(max(1, max_index)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        readable = bool(ok and frame is not None)
        cap.release()
        found.append((i, w, h, readable))
    return found


def print_cameras(max_index: int = 8):
    found = list_cameras(max_index)
    if not found:
        print(f"未找到摄像头 (0..{max_index - 1} 均打不开)")
    else:
        print(f"找到 {len(found)} 路摄像头:")
        for i, w, h, readable in found:
            flag = "可读" if readable else "能打开但读不到帧(可能被占用)"
            print(f"  [{i}] {w}x{h} {flag}")
    return found


def draw_hands(frame, hands_results):
    """在 frame 上画手部关键点 + 骨架连线 (蓝色). 兼容 solutions/tasks 两种后端."""
    try:
        lms_list = getattr(hands_results, "multi_hand_landmarks", None)
        if not lms_list:
            return
        hh, ww = frame.shape[:2]
        for hand_lms in lms_list:
            pts = [(int(p.x * ww), int(p.y * hh)) for p in hand_lms]
            for a, b in HAND_CONNECTIONS:  # 先画线
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)
            for (x, y) in pts:  # 再画点
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    except Exception:
        pass


def _cert_pem():
    # Https/Front/front_py.py -> 项目根/certs/cert.pem (自签 https 验签用)
    p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "certs", "cert.pem")
    return p if os.path.exists(p) else None


def _verify_arg(no_verify: bool):
    if no_verify:
        return False
    p = _cert_pem()
    if p is not None:
        return p
    try:  # 找不到证书则关警告放行 (理由: 开发机回环, 加密有、无验证次之, 总比明文强)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    return False


def _encode_jpg(img_bgr: np.ndarray, quality: int = 90):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def infer_remote_batch(crops, server: str, min_sim: float = 0.4,
                       timeout: float = 10.0, no_verify: bool = False):
    """批量 POST 到 /infer/batch, 字段 {index: img} (理由: 与后端批量接口对齐,
    k 张脸 1 次往返; 顺序即索引).

    crops: List[BGR ndarray], 顺序即索引 0..k-1.
    返回: ({idx: (name, sim)}, server_time, rtt_ms).
    索引缺失/整批失败时对应项为 ("unknow", 0.0).
    解析兼容新版 results-dict 与旧版单体 results (理由: 过渡期不断连).
    """
    url = normalize_server(server) + "/infer/batch"
    out: dict = {}
    if not crops:
        return out, 0.0, 0.0
    files = {}
    for i, c in enumerate(crops):
        b = _encode_jpg(c)
        if b is not None:
            files[str(i)] = (f"face{i}.jpg", b, "image/jpeg")
    if not files:
        return out, 0.0, 0.0
    t0 = time.time()
    try:
        r = requests.post(
            url,
            files=files,
            params={"min_sim": min_sim},
            timeout=timeout,
            verify=_verify_arg(no_verify),
        )
        rtt = (time.time() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        res = data.get("results", {})
        if isinstance(res, dict):
            items = list(res.items())
        elif isinstance(res, (list, tuple)) and len(res) == 2 and isinstance(res[0], str):
            items = [("0", res)]  # 旧版单体 [name, sim]
        elif isinstance(res, (list, tuple)):
            items = list(enumerate(res))  # 旧版列表
        else:
            items = []
        for k, v in items:
            try:
                out[int(k)] = (str(v[0]), float(v[1]))
            except Exception:
                continue
        return out, float(data.get("time", 0.0)), rtt
    except Exception as e:
        rtt = (time.time() - t0) * 1000
        print(f"[infer_remote_batch] 请求失败(k={len(files)}): {e}")
        return out, 0.0, rtt


def infer_remote(img_bgr: np.ndarray, server: str, min_sim: float = 0.4,
                 timeout: float = 10.0, no_verify: bool = False):
    """单图兼容: 走批量通道发 1 张, 取索引 0 (理由: 单图/整帧路径复用同一份批量逻辑)."""
    out, srv_t, rtt = infer_remote_batch([img_bgr], server, min_sim, timeout, no_verify)
    name, sim = out.get(0, ("unknow", 0.0))
    return name, sim, srv_t, rtt


def infer_image_file(image_path: str, server: str, min_sim: float = 0.4,
                     no_verify: bool = False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"读图失败: {image_path}")
        return
    name, sim, srv_t, rtt = infer_remote(img, server, min_sim, 10.0, no_verify)
    print(f"{image_path} -> {name}:{sim:.4f} server_time={srv_t:.3f}s rtt={rtt:.1f}ms")


def run_camera(args):
    server = args.server
    # ---- 尝试加载本地检测器 (对标 solve 的本地部分) ----
    checker = None
    if not args.remote_only:
        try:
            
            checker = FaceCheckWithMediapipe()
            print("[front_py] 本地 SCRFD + Mediapipe 加载成功, 检测在本地, 识别走 /infer/batch")
        except Exception as e:
            print(f"[front_py] 本地模型加载失败 ({e}), 自动降级为纯远端整帧模式")
            checker = None
    else:
        print("[front_py] remote-only 模式: 整帧直接 POST /infer/batch, 不做本地检测")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        print_cameras(args.max_cameras)  # 打不开时直接给出列表, 不让用户猜 index
        print("用法: 先 --list-cameras 看列表, 再 --camera <index> 指定 (如 --camera 1)")
        return
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("按 q 退出 | bboxes 本地 detect, 身份远端 /infer/batch")
    frame_idx = 0
    bboxes = np.zeros((0, 5), dtype=np.float32)
    face_results = []  # [{bbox, det_score, name, sim}]
    hands_results = None
    t_prev = time.time()
    fps = 0.0

    # IoU复用阈值 (照抄 solve 的思路)
    iou_thresh = 0.6

    def bbox_iou(a, b):
        xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
        xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
        w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
        inter = w * h
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + \
            max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / (ua + 1e-9)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取帧失败, 退出")
            break
        frame_idx += 1
        t_det = t_rec = t_hand = 0.0

        if checker is not None:
            # ---- 1. 本地人脸检测 (跳帧, 同 solve) ----
            if frame_idx == 1 or ((frame_idx - 1) % args.detect_interval == 0):
                ta = time.time()
                bboxes, _kpss = checker.detect(frame)
                t_det = (time.time() - ta) * 1000

            # ---- 2. 远端批量识别: IoU复用 + recognition_interval 节流,
            #      本轮未命中的人脸收集成一批, 一次 POST {index: crop}
            #      (理由: 批量接口即为此设计; 批量下标 = 本批发送顺序,
            #      与 new_results 的对应靠 pending 位置映射, 不是人脸序号)
            ta = time.time()
            new_results = []
            pending = []  # [(new_results下标, crop)]
            need_remote = (frame_idx == 1 or
                           ((frame_idx - 1) % args.recognition_interval == 0))
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[i]
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                if x2i <= x1i or y2i <= y1i:
                    continue
                reused = None
                for fr in face_results:
                    if bbox_iou((x1i, y1i, x2i, y2i), fr["bbox"]) > iou_thresh:
                        reused = fr
                        break
                if reused is not None and not need_remote:
                    new_results.append({**reused, "bbox": (x1i, y1i, x2i, y2i),
                                        "det_score": float(score)})
                    continue
                if reused is not None and not args.refresh_identity:
                    # recognition_interval 未到且允许复用身份: 直接复用, 不请求后端
                    new_results.append({**reused, "bbox": (x1i, y1i, x2i, y2i),
                                        "det_score": float(score)})
                    continue
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    continue
                new_results.append({"bbox": (x1i, y1i, x2i, y2i),
                                    "det_score": float(score),
                                    "name": "?", "sim": 0.0})
                pending.append((len(new_results) - 1, crop))
            if pending:
                batch_res, _srv_t, _rtt = infer_remote_batch(
                    [c for _, c in pending], server, args.min_sim, args.timeout, args.no_verify)
                for pos, (ri, _c) in enumerate(pending):
                    name, sim = batch_res.get(pos, ("unknow", 0.0))
                    new_results[ri]["name"] = name
                    new_results[ri]["sim"] = sim
            face_results = new_results
            t_rec = (time.time() - ta) * 1000

            # ---- 3. 本地手部 (跳帧, 同 solve) ----
            if frame_idx == 1 or ((frame_idx - 1) % args.hands_interval == 0):
                ta = time.time()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                try:
                    hands_results = checker.hands.process(
                        img_rgb, timestamp_ms=int(time.time() * 1000))
                except TypeError:
                    hands_results = checker.hands.process(img_rgb)
                t_hand = (time.time() - ta) * 1000

            # ---- 可视化: 人脸检测框 + 身份标签(按 STATUS 着色), 手部点线 ----
            for fr in face_results:
                x1i, y1i, x2i, y2i = fr["bbox"]
                c = STATUS[status_of(fr.get("name", "?"))]["color"]
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), c, 2)
                det_s = fr.get("det_score", 0.0)
                cv2.putText(frame, f"{fr['name']}:{fr['sim']:.2f} det:{det_s:.2f}",
                            (x1i, max(0, y1i - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            draw_hands(frame, hands_results)
        else:
            # ---- 纯远端模式: 整帧按间隔 POST ----
            if frame_idx == 1 or ((frame_idx - 1) % args.detect_interval == 0):
                ta = time.time()
                name, sim, _srv_t, _rtt = infer_remote(
                    frame, server, args.min_sim, args.timeout, args.no_verify)
                t_rec = (time.time() - ta) * 1000
                face_results = [{"bbox": (10, 10, 200, 200), "name": name, "sim": sim}]
            for fr in face_results:
                c = STATUS[status_of(fr.get("name", "?"))]["color"]
                cv2.putText(frame, f"{fr['name']}:{fr['sim']:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

        now = time.time()
        fps = fps * 0.9 + (1.0 / max(1e-6, now - t_prev)) * 0.1
        t_prev = now
        if args.show_fps:
            cv2.putText(frame, f"FPS:{fps:.1f} det:{t_det:.0f}ms infer:{t_rec:.0f}ms hand:{t_hand:.0f}ms",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if frame_idx % 30 == 1:
            print(f"[t] fps={fps:.1f} det={t_det:.0f}ms infer={t_rec:.0f}ms "
                  f"hand={t_hand:.0f}ms faces={len(face_results)}")

        cv2.imshow("Front/py - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        if checker is not None and hasattr(checker.hands, "close"):
            checker.hands.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Python前端: 本地检测+手部, 识别调用后端/infer/batch")
    ap.add_argument("--server", default="https://127.0.0.1:8060", help="后端地址(只填host,如https://127.0.0.1:8060)")
    ap.add_argument("--camera", type=int, default=0, help="摄像头 index, 先 --list-cameras 查看可用列表")
    ap.add_argument("--list-cameras", action="store_true", help="只枚举摄像头列表并退出, 不开流")
    ap.add_argument("--max-cameras", type=int, default=8, help="枚举上限: 探测 0..MAX-1 (默认8)")
    ap.add_argument("--image", default=None, help="单图模式路径")
    ap.add_argument("--min-sim", type=float, default=0.4)
    ap.add_argument("--detect-interval", type=int, default=3)
    ap.add_argument("--hands-interval", type=int, default=4)
    ap.add_argument("--recognition-interval", type=int, default=3,
                    help="每N个detect周期才重新请求一次/infer/batch, 其余帧IoU复用身份")
    ap.add_argument("--refresh-identity", action="store_true",
                    help="每轮都重新请求/infer/batch(默认按recognition-interval复用)")
    ap.add_argument("--remote-only", action="store_true", help="不加载本地模型, 整帧送后端")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--no-verify", action="store_true", help="跳过自签证书验证(证书对不上/过期应急用)")
    ap.add_argument("--show-fps", action="store_true", default=True)
    ap.add_argument("--no-show-fps", dest="show_fps", action="store_false")
    args = ap.parse_args()

    try:
        lan = get_lan_ip()
        print(f"[net] 本机LAN IP: {lan} (局域网用 https://{lan}:8060, 本机用 https://127.0.0.1:8060)")
    except Exception:
        pass
    if args.list_cameras:
        print_cameras(args.max_cameras)
        return
    if args.image:
        infer_image_file(args.image, args.server, args.min_sim, args.no_verify)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
