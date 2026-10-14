"""手势鼠标 demo (独立, 调用 Back.py 解耦后的手势方法).

逻辑: 后端.detect_hands(BGR) -> 手势字典 -> 预留判断 -> 右手食指相对位移 -> 移动系统鼠标.
鼠标移动使用 Windows ctypes, 无需安装 pyautogui.

运行:
    python dev/demo_hand_mouse.py
按键: q 退出 | m 开关鼠标控制 | f 镜像左右手标签(自拍镜像时用)
"""
import os
import sys
import time
import math
import ctypes

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根: Back.py 所在
from Back import FaceCheckWithMediapipe  # noqa: E402  调用解耦后端, 不改原逻辑

# ---------------- 配置 ----------------
CAMERA_ID = 0
GAIN = 2.0            # 位移增益: 屏幕像素 = 归一化位移 * 屏幕尺寸 * GAIN
DEAD_ZONE = 0.002     # 归一化位移死区, 小于此值视为抖动丢弃
SMOOTH_ALPHA = 0.4    # EMA 平滑系数 0~1, 越大越跟手、越小越稳
LOST_RESET_FRAMES = 5  # 右手连续丢失 N 帧后重置 prev, 防止跳变
MIRROR_PREVIEW = True  # 仅预览镜像, 不影响归一化坐标计算
RIGHT_LABEL = "Right"  # 目标手标签; 若摄像头镜像导致左右反了, 按 f 切换

# ---------------- 鼠标 (ctypes, Windows) ----------------
_user32 = ctypes.windll.user32


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def get_cursor():
    p = _POINT()
    _user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def set_cursor(x, y):
    _user32.SetCursorPos(int(x), int(y))


def get_screen_size():
    return _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)

def setLeftDown():
#   mouse_event(
#     MOUSEEVENTF_LEFTDOWN, // 第1个参数：dwFlags，事件类型
#     0,                    // 第2个参数：dx，鼠标水平移动量
#     0,                    // 第3个参数：dy，鼠标垂直移动量
#     0,                    // 第4个参数：dwData，滚轮数据
#     0                     // 第5个参数：dwExtraInfo，附加信息
# )
    _user32.mouse_event(_user32.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def setLeftUp():
    _user32.mouse_event(_user32.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def setRightDown():
    _user32.mouse_event(_user32.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def setRightUp():
    _user32.mouse_event(_user32.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


# ---------------- 手部骨骼 ----------------
# MediaPipe 21关键点拓扑: 拇指0-4, 食指5-8, 中指9-12, 环指13-16, 小指17-20
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_skeleton(img, hand: dict, mirror: bool = False, is_target: bool = False):
    """在预览图上画手部节点+骨骼连线."""
    lms = hand.get("landmarks_norm", [])
    if len(lms) != 21:
        return
    h, w = img.shape[:2]
    pts = []
    for lm in lms:
        px = int(float(lm["x"]) * w)
        py = int(float(lm["y"]) * h)
        if mirror:  # 预览做了 flip, x 需同步镜像
            px = w - 1 - px
        pts.append((px, py))
    bone = (0, 255, 0) if is_target else (255, 200, 0)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], bone, 2, cv2.LINE_AA)
    for x, y in pts:  # 节点
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pts[8], 6, (0, 255, 255), 2, cv2.LINE_AA)  # 食指指尖高亮
    cv2.putText(img, f"{hand.get('handedness', '?')}",
                (pts[0][0] - 10, max(0, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bone, 2)


# ---------------- 预留留空判断 ----------------
def gesture_gate(hand: dict) -> bool:
    """预留的手势门控判断, 后续在此填类别/投票/冷却逻辑.

    Args:
        hand: 后端 detect_hands 返回的单个手势字典.

    Returns:
        True 允许移鼠标, False 丢弃本帧.
    """
    # TODO: 例1: 连续5帧投票判定类别后再放行
    # TODO: 例2: 只有 OK/张开手势才允许移鼠标: if hand.get("gesture") != "OPEN": return False
    # TODO: 例3: 冷却1s防误触
    return True  # 当前直通, 仅做演示


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)  # 让 Model\... 相对路径可用

    backend = FaceCheckWithMediapipe()
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[demo_hand_mouse] 无法打开摄像头 {CAMERA_ID}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    scr_w, scr_h = get_screen_size()
    print(f"[demo_hand_mouse] screen={scr_w}x{scr_h} gain={GAIN} | q退出 m开关 f切换左右手")
    mouse_on = True
    right_label = RIGHT_LABEL
    prev = None  # (x, y) 上一帧右手食指归一化坐标
    prev_sdx, prev_sdy = 0.0, 0.0
    lost = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[demo_hand_mouse] 读帧失败")
            break

        hands = backend.detect_hands(frame)  # 解耦调用: 输入BGR -> 手势字典列表
        rhand = next((h for h in hands if h.get("handedness") == right_label), None)

        status = "no-hand"
        if rhand is not None and gesture_gate(rhand):  # 预留判断通过后才动鼠标
            tip = rhand["index_tip_norm"]
            x, y = float(tip["x"]), float(tip["y"])
            if prev is not None:
                dx, dy = x - prev[0], y - prev[1]
                if math.hypot(dx, dy) >= DEAD_ZONE and mouse_on:
                    # EMA 平滑 + 相对位移映射到屏幕
                    sdx = SMOOTH_ALPHA * dx + (1 - SMOOTH_ALPHA) * prev_sdx
                    sdy = SMOOTH_ALPHA * dy + (1 - SMOOTH_ALPHA) * prev_sdy
                    prev_sdx, prev_sdy = sdx, sdy
                    cx, cy = get_cursor()
                    nx = min(max(cx + sdx * scr_w * GAIN, 0), scr_w - 1)
                    ny = min(max(cy + sdy * scr_h * GAIN, 0), scr_h - 1)
                    set_cursor(nx, ny)
                    status = f"move dx={dx:+.4f} dy={dy:+.4f}"
                else:
                    status = "hold(dead-zone)" if mouse_on else "mouse-off"
            else:
                status = "lock"
            prev = (x, y)
            lost = 0
        else:
            lost += 1
            if lost >= LOST_RESET_FRAMES:
                prev = None
                prev_sdx, prev_sdy = 0.0, 0.0
            if rhand is None:
                status = "no-right-hand"
            else:
                status = "gated"

        # 可视化: 骨骼连线 + 状态
        show = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
        for h in hands:
            draw_hand_skeleton(show, h, mirror=MIRROR_PREVIEW,
                               is_target=(h.get("handedness") == right_label))
        cv2.putText(show, f"{status} mouse={'ON' if mouse_on else 'OFF'} [{right_label}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("demo_hand_mouse - q quit, m toggle, f flip hand", show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("m"):
            mouse_on = not mouse_on
            print(f"[demo_hand_mouse] mouse_on={mouse_on}")
        elif k == ord("f"):
            right_label = "Left" if right_label == "Right" else "Right"
            prev = None
            print(f"[demo_hand_mouse] right_label -> {right_label}")

    cap.release()
    cv2.destroyAllWindows()
    try:
        backend.hands.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
