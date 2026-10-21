"""手势鼠标 demo (独立, 调用 Back.py 解耦后的手势方法).

逻辑: 后端.detect_hands(BGR) -> 手势字典 -> 手身份跟踪 -> 相对位移 -> 移动系统鼠标.
鼠标移动使用 Windows ctypes, 无需安装 pyautogui.
位移使用等比例映射(不随屏幕宽高比拉伸), 并跟随镜像预览方向.

手势:
    光标      : 光标手食指指尖
    左键      : 同手拇指-中指捏合, 按住=拖拽, 松开=抬起
    右键      : 同手拇指-无名指捏合, 按住=按下, 松开=抬起
    倍率调节  : 另一只手食指贴住光标手食指(折线 5-6-7-8)上下滑动,
                滑向指尖增大/滑向指根减小; 离开退出, 调节时冻结光标.

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

# ---- 光标 ----
GAIN = 2.0             # 初始倍率: 1 摄像头像素 -> GAIN * min(屏宽/帧宽, 屏高/帧高) 屏幕像素
GAIN_MIN, GAIN_MAX = 0.5, 4.0
DEAD_ZONE_PX = 4.0     # 屏幕像素死区; 小于此值的位移会累积, 不会丢失
SMOOTH_ALPHA = 0.4     # EMA 平滑系数 0~1, 越大越跟手、越小越稳
LOST_RESET_FRAMES = 5  # 光标手连续丢失 N 帧后重置平滑状态
REACQUIRE_FRAMES = 15  # 光标手丢失后保持身份的最长帧数
TRACK_MAX_DIST = 0.12  # 连续帧手腕跟踪点最近邻阈值(归一化画面坐标)
REACQUIRE_MAX_DIST = 0.22  # 断线重连阈值
MIRROR_PREVIEW = True  # 仅预览镜像
MIRROR_CONTROL_X = True  # 控制方向跟随摄像头(手右移 -> 光标右移); 关预览镜像时可改 False
RIGHT_LABEL = "Right"  # 目标手标签; 若摄像头镜像导致左右反了, 按 f 切换
PREVIEW_W = 1280  # 预览窗口初始宽度(可拖动缩放), 过高时按屏幕自动缩小
WINDOW_NAME = "demo_hand_mouse - q quit, m toggle, f flip hand"

# ---- 点击 (距离均按掌长 dist(0,9) 归一化, 需按手掌大小实测微调) ----
PINCH_ENTER = 0.30          # 拇指-中指: 进入捏合
PINCH_EXIT = 0.45           # 拇指-中指: 退出捏合
RIGHT_PINCH_ENTER = 0.32    # 拇指-无名指: 进入捏合
RIGHT_PINCH_EXIT = 0.48     # 拇指-无名指: 退出捏合
CLICK_CONFIRM_FRAMES = 3    # 捏合持续 N 帧才触发按下
CLICK_RELEASE_FRAMES = 2    # 张开持续 N 帧才触发抬起
CLICK_COOLDOWN_S = 0.20     # 左键抬起后的冷却

# ---- 倍率调节 (另一只手食指沿光标手食指滑动) ----
GAIN_CONTACT_ENTER = 0.18   # 第二只手食指到光标手食指折线的距离阈值(掌长归一化)
GAIN_CONTACT_EXIT = 0.30
GAIN_CONTACT_FRAMES = 3     # 接触持续 N 帧才进入调节
GAIN_RELEASE_FRAMES = 2     # 离开持续 N 帧才退出调节
GAIN_VALUE_SMOOTH = 0.35      # 倍率向目标值收敛的 EMA, 越大越跟手
                              # 目标值按接触点在食指折线上的位置线性映射: 指根=MIN, 指尖=MAX
GAIN_OVERLAY_SCALE = 0.7      # 右下角倍率面板相对原始尺寸的缩放
BUTTON_RELEASE_FRAMES = 5   # 光标手丢失 N 帧后强制释放按键

# ---------------- 鼠标 (ctypes, Windows) ----------------
_user32 = ctypes.windll.user32

# mouse_event 的 dwFlags 是常量, 不是 user32 导出的函数, 不能写 _user32.MOUSEEVENTF_*
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


_user32.GetCursorPos.argtypes = [ctypes.POINTER(_POINT)]
_user32.GetCursorPos.restype = ctypes.c_bool
_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = ctypes.c_bool
_user32.mouse_event.argtypes = [
    ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_size_t
]
_user32.mouse_event.restype = None


def enable_dpi_awareness():
    """让 GetSystemMetrics / GetCursorPos / SetCursorPos 使用同一套物理像素坐标."""
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:  # PER_MONITOR_DPI_AWARE
            return
    except Exception:
        pass
    try:
        _user32.SetProcessDPIAware()
    except Exception:
        pass


def get_cursor():
    p = _POINT()
    _user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def set_cursor(x, y):
    _user32.SetCursorPos(int(round(x)), int(round(y)))


def get_screen_size():
    return _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)


def setLeftDown():
    _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def setLeftUp():
    _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def setRightDown():
    _user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)


def setRightUp():
    _user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)


# ---------------- 手部数学 ----------------
def _dist(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _dist_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _palm_len(lms):
    return max(_dist(lms[0], lms[9]), 1e-6)


def _point_segment_distance(p, a, b):
    """点到线段距离, 返回 (距离, 最近点, 段内归一化位置 t)."""
    ax, ay = float(a["x"]), float(a["y"])
    bx, by = float(b["x"]), float(b["y"])
    px, py = float(p["x"]), float(p["y"])
    vx, vy = bx - ax, by - ay
    seg2 = vx * vx + vy * vy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay), (ax, ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / seg2))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy), (qx, qy), t


def _point_to_index_finger(p, lms):
    """点到食指折线 (5-6-7-8) 的距离, 返回 (距离, 最近点, 折线归一化位置 t)."""
    pts = [lms[i] for i in (5, 6, 7, 8)]
    lengths = [_dist(pts[i], pts[i + 1]) for i in range(3)]
    total = max(sum(lengths), 1e-6)
    best = None
    acc = 0.0
    for i in range(3):
        d, q, seg_t = _point_segment_distance(p, pts[i], pts[i + 1])
        t = (acc + seg_t * lengths[i]) / total
        if best is None or d < best[0]:
            best = (d, q, t)
        acc += lengths[i]
    return best


def _hand_track_point(hand):
    """手腕偏掌心方向的位置, 用于身份跟踪; 比手中心更远离食指指尖, 两手触碰时不易混淆."""
    lms = hand.get("landmarks_norm", [])
    if len(lms) != 21:
        return None
    return (0.7 * float(lms[0]["x"]) + 0.3 * float(lms[9]["x"]),
            0.7 * float(lms[0]["y"]) + 0.3 * float(lms[9]["y"]))


class CursorHandTracker:
    """用上一帧手腕位置最近邻跟踪光标手, 避免两手靠近/交叉时 handedness 翻转导致换手."""

    def __init__(self, target_label):
        self.target_label = target_label
        self.center = None
        self.label = None
        self.missing = 0

    def reset(self, target_label=None):
        if target_label is not None:
            self.target_label = target_label
        self.center = None
        self.label = None
        self.missing = 0

    def match(self, hands):
        """返回 (光标手, 调节手); 无光标手时返回 (None, None)."""
        cands = []
        for h in hands:
            c = _hand_track_point(h)
            if c is not None:
                cands.append((h, c))

        cursor = None
        if self.center is not None and self.missing <= REACQUIRE_FRAMES:
            best_h, best_d = None, None
            for h, c in cands:
                d = _dist_xy(c, self.center)
                if best_d is None or d < best_d:
                    best_h, best_d = h, d
            max_dist = TRACK_MAX_DIST if self.missing == 0 else REACQUIRE_MAX_DIST
            # 断线重连时额外要求标签一致, 防止把另一只手误认成光标手
            if best_h is not None and best_d <= max_dist:
                if self.missing == 0 or best_h.get("handedness") == self.label:
                    cursor = best_h

        if cursor is None and (self.center is None or self.missing > REACQUIRE_FRAMES):
            labeled = [(h, c) for h, c in cands
                       if h.get("handedness") == self.target_label]
            if labeled:
                if self.center is not None and len(labeled) > 1:
                    cursor = min(labeled, key=lambda hc: _dist_xy(hc[1], self.center))[0]
                else:
                    cursor = labeled[0][0]

        if cursor is None:
            self.missing += 1
            if self.missing > REACQUIRE_FRAMES:
                self.center = None
                self.label = None
            return None, None

        self.center = _hand_track_point(cursor)
        self.label = cursor.get("handedness")
        self.missing = 0

        adjuster = None
        best_d = None
        for h, c in cands:
            if h is cursor:
                continue
            d = _dist_xy(c, self.center)
            if best_d is None or d < best_d:
                adjuster, best_d = h, d
        return cursor, adjuster


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


def draw_hand_skeleton(img, hand: dict, mirror: bool = False, role: str = "other"):
    """在预览图上画手部节点+骨骼连线. role: cursor/adjuster/other."""
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
    if role == "cursor":
        bone = (0, 255, 0)
    elif role == "adjuster":
        bone = (255, 200, 0)
    else:
        bone = (150, 150, 150)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], bone, 2, cv2.LINE_AA)
    for x, y in pts:  # 节点
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pts[8], 6, (0, 255, 255), 2, cv2.LINE_AA)  # 食指指尖高亮
    cv2.putText(img, f"{hand.get('handedness', '?')}",
                (pts[0][0] - 10, max(0, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bone, 2)


def draw_gain_indicator(img, gain: float, active: bool = False):
    """右下角显示当前缩放倍率、进度条和调节状态."""
    h, w = img.shape[:2]
    s = GAIN_OVERLAY_SCALE
    x1, y1 = w - 10, h - 32
    x0 = max(6, x1 - int(220 * s))
    y0 = max(6, y1 - int(72 * s))
    cv2.rectangle(img, (x0, y0), (x1, y1), (25, 25, 25), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    color = (0, 165, 255) if active else (0, 255, 255)
    cv2.putText(img, f"GAIN x{gain:.2f}", (x0 + int(10 * s), y0 + int(25 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72 * s, color, 1, cv2.LINE_AA)
    if active:
        cv2.putText(img, "ADJUST", (x1 - int(66 * s), y0 + int(17 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, color, 1, cv2.LINE_AA)
    bx0, bx1 = x0 + int(10 * s), x1 - int(10 * s)
    by0, by1 = y0 + int(34 * s), y0 + int(48 * s)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (110, 110, 110), 1)
    ratio = (gain - GAIN_MIN) / (GAIN_MAX - GAIN_MIN)
    fill = int((bx1 - bx0 - 2) * min(max(ratio, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(img, (bx0 + 1, by0 + 1), (bx0 + 1 + fill, by1 - 1), color, -1)
    cv2.putText(img, f"{GAIN_MIN:.1f}", (bx0, y0 + int(66 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"{GAIN_MAX:.1f}", (bx1 - int(22 * s), y0 + int(66 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (180, 180, 180), 1, cv2.LINE_AA)


# ---------------- 预留留空判断 ----------------
def gesture_gate(hand: dict) -> bool:
    """预留的手势门控判断, 后续在此填类别/投票/冷却逻辑.

    Args:
        hand: 后端 detect_hands 返回的单个手势字典.

    Returns:
        True 允许控制, False 丢弃本帧.
    """
    # TODO: 例1: 连续5帧投票判定类别后再放行
    # TODO: 例2: 只有 OK/张开手势才允许移鼠标: if hand.get("gesture") != "OPEN": return False
    # TODO: 例3: 冷却1s防误触
    return True  # 当前直通, 仅做演示


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)  # 让 Model\... 相对路径可用
    enable_dpi_awareness()

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
    print("[demo_hand_mouse] 左键=拇指中指捏合, 右键=拇指无名指, "
          "倍率=另一只手食指沿操控手食指滑动(指尖增/指根减)")

    mouse_on = True
    right_label = RIGHT_LABEL
    tracker = CursorHandTracker(right_label)
    gain = GAIN

    smooth = None  # 平滑后的食指位置(控制坐标系)
    anchor = None  # 上次真正移动光标时的平滑位置, 死区内的位移靠它累积
    lost = 0
    mode = "cursor"  # cursor / armed / drag / gain / lost

    left_down = False
    right_down = False
    left_latched = False   # 抬起后必须先张开, 才能再次触发
    right_latched = False
    arm_frames = 0
    release_frames = 0
    right_arm_frames = 0
    right_release_frames = 0
    contact_frames = 0
    gain_release_frames = 0
    cooldown_until = 0.0
    window_created = False

    def release_buttons():
        nonlocal left_down, right_down
        if left_down:
            setLeftUp()
            left_down = False
        if right_down:
            setRightUp()
            right_down = False

    def reset_gesture_state():
        nonlocal arm_frames, release_frames, right_arm_frames, right_release_frames
        nonlocal contact_frames, gain_release_frames
        arm_frames = 0
        release_frames = 0
        right_arm_frames = 0
        right_release_frames = 0
        contact_frames = 0
        gain_release_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[demo_hand_mouse] 读帧失败")
                break

            hands = backend.detect_hands(frame)  # 解耦调用: 输入BGR -> 手势字典列表
            cursor, adjuster = tracker.match(hands)
            frame_h, frame_w = frame.shape[:2]
            now = time.monotonic()

            left_ratio = None
            right_ratio = None
            contact_ratio = None
            contact_pt = None
            contact_t = None
            can_control = False
            cam_to_screen = gain * min(scr_w / frame_w, scr_h / frame_h)

            if cursor is None:
                lost += 1
                mode = "lost"
                left_latched = True
                right_latched = True
                if lost >= BUTTON_RELEASE_FRAMES and (left_down or right_down):
                    release_buttons()
                if lost >= LOST_RESET_FRAMES:
                    smooth = None
                    anchor = None
                reset_gesture_state()
                status = "lost" if lost < LOST_RESET_FRAMES else "no-hand"
            else:
                lms = cursor["landmarks_norm"]
                palm = _palm_len(lms)
                tip = lms[8]
                ctrl_x = 1.0 - float(tip["x"]) if MIRROR_CONTROL_X else float(tip["x"])
                ctrl_y = float(tip["y"])

                if smooth is None or lost > 0:
                    # 首次/丢手/门控恢复后重新锁定, 不补跳丢失期间的位移
                    smooth = (ctrl_x, ctrl_y)
                    anchor = smooth
                    reset_gesture_state()
                    if lost > 0:
                        mode = "drag" if left_down else "cursor"
                else:
                    # 先平滑位置, 再对 anchor 求相对位移; 慢速小位移会累积
                    smooth = (SMOOTH_ALPHA * ctrl_x + (1.0 - SMOOTH_ALPHA) * smooth[0],
                              SMOOTH_ALPHA * ctrl_y + (1.0 - SMOOTH_ALPHA) * smooth[1])
                lost = 0

                left_ratio = _dist(lms[4], lms[12]) / palm
                right_ratio = _dist(lms[4], lms[16]) / palm
                if adjuster is not None:
                    dist, nearest, contact_t = _point_to_index_finger(
                        adjuster["landmarks_norm"][8], lms)
                    contact_ratio = dist / palm
                    contact_pt = nearest

                left_inside = left_ratio < PINCH_ENTER
                left_outside = left_ratio > PINCH_EXIT
                right_outside = right_ratio > RIGHT_PINCH_EXIT
                if left_outside:
                    left_latched = False
                if right_outside:
                    right_latched = False

                can_control = mouse_on and gesture_gate(cursor)
                if not can_control:
                    if left_down or right_down:
                        release_buttons()
                    left_latched = True
                    right_latched = True
                    mode = "cursor"
                    anchor = smooth
                    reset_gesture_state()
                    status = "mouse-off" if not mouse_on else "gated"
                elif left_down:
                    mode = "drag"
                    if left_outside:
                        release_frames += 1
                    else:
                        release_frames = 0
                    if release_frames >= CLICK_RELEASE_FRAMES:
                        setLeftUp()
                        left_down = False
                        left_latched = True
                        cooldown_until = now + CLICK_COOLDOWN_S
                        release_frames = 0
                        mode = "cursor"
                        anchor = smooth
                elif mode == "gain":
                    if (adjuster is None or contact_ratio is None or contact_t is None
                            or contact_ratio > GAIN_CONTACT_EXIT):
                        gain_release_frames += 1
                    else:
                        gain_release_frames = 0
                        t = min(max(contact_t, 0.0), 1.0)
                        target = GAIN_MIN + t * (GAIN_MAX - GAIN_MIN)
                        gain += (target - gain) * GAIN_VALUE_SMOOTH
                        gain = min(max(gain, GAIN_MIN), GAIN_MAX)
                    if gain_release_frames >= GAIN_RELEASE_FRAMES or adjuster is None:
                        mode = "cursor"
                        anchor = smooth
                        gain_release_frames = 0
                        contact_frames = 0
                else:
                    # 左键: 捏合确认期间冻结光标, 丢弃这段位移
                    if (not left_latched and not right_down
                            and now >= cooldown_until and left_inside):
                        arm_frames += 1
                        mode = "armed"
                        anchor = smooth
                        if arm_frames >= CLICK_CONFIRM_FRAMES:
                            setLeftDown()
                            left_down = True
                            mode = "drag"
                            arm_frames = 0
                            anchor = smooth
                    else:
                        if arm_frames > 0:
                            anchor = smooth
                        arm_frames = 0
                        if mode == "armed":
                            mode = "cursor"

                    # 右键: 拇指-无名指捏合
                    if mode == "cursor" and not left_down:
                        if right_down:
                            if right_outside:
                                right_release_frames += 1
                            else:
                                right_release_frames = 0
                            if right_release_frames >= CLICK_RELEASE_FRAMES:
                                setRightUp()
                                right_down = False
                                right_latched = True
                                right_release_frames = 0
                        elif not right_latched and right_ratio < RIGHT_PINCH_ENTER:
                            right_arm_frames += 1
                            if right_arm_frames >= CLICK_CONFIRM_FRAMES:
                                setRightDown()
                                right_down = True
                                right_arm_frames = 0
                        else:
                            right_arm_frames = 0

                    # 倍率调节: 另一只手食指沿光标手食指滑动, 无按键按下时进入
                    if (mode == "cursor" and not left_down and not right_down
                            and left_ratio >= PINCH_ENTER and adjuster is not None
                            and contact_ratio is not None and contact_t is not None
                            and contact_ratio < GAIN_CONTACT_ENTER):
                        contact_frames += 1
                        if contact_frames >= GAIN_CONTACT_FRAMES:
                            mode = "gain"
                            anchor = smooth
                            contact_frames = 0
                    else:
                        contact_frames = 0

                    if mode != "cursor":
                        right_arm_frames = 0

            # 光标移动与状态显示
            if not mouse_on:
                status = "mouse-off"
            elif mode == "armed":
                status = "click-armed"
            elif mode == "gain":
                status = "gain-adjust"
            elif can_control and mode in ("cursor", "drag"):
                if smooth is not None and anchor is not None:
                    # smooth/anchor 是归一化坐标, 先换成摄像头像素, 再换成等比例屏幕像素
                    dx = (smooth[0] - anchor[0]) * frame_w * cam_to_screen
                    dy = (smooth[1] - anchor[1]) * frame_h * cam_to_screen
                    if math.hypot(dx, dy) >= DEAD_ZONE_PX:
                        cx, cy = get_cursor()
                        nx = min(max(cx + dx, 0), scr_w - 1)
                        ny = min(max(cy + dy, 0), scr_h - 1)
                        set_cursor(nx, ny)
                        anchor = smooth
                        status = f"move dx={dx:+.1f}px dy={dy:+.1f}px"
                    else:
                        status = "drag" if mode == "drag" else "hold(dead-zone)"
                else:
                    status = "lost"

            # 可视化: 骨骼连线 + 状态
            show = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
            for h in hands:
                if h is cursor:
                    role = "cursor"
                elif h is adjuster:
                    role = "adjuster"
                else:
                    role = "other"
                draw_hand_skeleton(show, h, mirror=MIRROR_PREVIEW, role=role)
            if contact_pt is not None and contact_ratio is not None \
                    and contact_ratio < GAIN_CONTACT_EXIT:
                px = int(contact_pt[0] * frame_w)
                py = int(contact_pt[1] * frame_h)
                if MIRROR_PREVIEW:
                    px = frame_w - 1 - px
                cv2.circle(show, (px, py), 7, (0, 255, 255), 2, cv2.LINE_AA)

            if cursor is not None:
                contact_txt = "--" if contact_ratio is None else f"{contact_ratio:.2f}"
                status_line = (f"{status} "
                               f"L={left_ratio:.2f} R={right_ratio:.2f} C={contact_txt}")
            else:
                status_line = status
            button_line = (f"L-btn={'DOWN' if left_down else 'UP'} "
                           f"R-btn={'DOWN' if right_down else 'UP'} "
                           f"mouse={'ON' if mouse_on else 'OFF'} [{right_label}]")
            hud_color = (0, 255, 255) if mouse_on else (0, 0, 255)
            cv2.putText(show, status_line, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
            cv2.putText(show, button_line, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
            cv2.putText(show, "L: thumb+middle | R: thumb+ring | GAIN: slide 2nd index on index (tip=+)",
                        (10, show.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            draw_gain_indicator(show, gain, active=(mode == "gain"))
            if not window_created:  # 可缩放窗口, 初始尺寸按帧比例和屏幕大小确定
                win_w = min(PREVIEW_W, int(scr_w * 0.9))
                win_h = int(win_w * frame_h / frame_w)
                max_h = int(scr_h * 0.9)
                if win_h > max_h:
                    win_h = max_h
                    win_w = int(win_h * frame_w / frame_h)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, max(320, win_w), max(240, win_h))
                window_created = True
            cv2.imshow(WINDOW_NAME, show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("m"):
                mouse_on = not mouse_on
                if not mouse_on:
                    release_buttons()
                    left_latched = True
                    right_latched = True
                    mode = "cursor"
                    if smooth is not None:
                        anchor = smooth
                    reset_gesture_state()
                print(f"[demo_hand_mouse] mouse_on={mouse_on}")
            elif k == ord("f"):
                right_label = "Left" if right_label == "Right" else "Right"
                release_buttons()
                tracker.reset(right_label)
                left_latched = True
                right_latched = True
                smooth = None
                anchor = None
                mode = "cursor"
                reset_gesture_state()
                print(f"[demo_hand_mouse] right_label -> {right_label}")
    finally:
        release_buttons()
        cap.release()
        cv2.destroyAllWindows()
        try:
            backend.hands.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
