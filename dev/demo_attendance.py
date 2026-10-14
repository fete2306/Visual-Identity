"""签到 demo (独立, 调用 Back.py 解耦后的人脸方法 + sqlite 写入).

逻辑: 后端.detect_faces(BGR) -> 人脸框 -> 裁图 -> FaceRecognition 识别 -> attendance_db.try_sign_in()
对应 FR-05 实时签到 (达阈值写入、当日去重、stranger 不记).

运行:
    python dev/demo_attendance.py
按键: q 退出
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根: Back.py 所在
from Back import FaceCheckWithMediapipe  # noqa: E402
from attendance_db import init_db, try_sign_in, fetch_today  # noqa: E402

CAMERA_ID = 0
CAM_ID = "cam0"
MIN_SIM = 0.4
DETECT_INTERVAL = 3
COOLDOWN_S = 60
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)  # 让 Model\... 相对路径可用
    init_db(DB_PATH)

    backend = FaceCheckWithMediapipe()
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[demo_attendance] 无法打开摄像头 {CAMERA_ID}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print(f"[demo_attendance] db={DB_PATH} min_sim={MIN_SIM} | q退出")
    frame_idx = 0
    bboxes = None
    face_results = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[demo_attendance] 读帧失败")
            break
        frame_idx += 1

        # 解耦调用1: 输入BGR -> 人脸框坐标
        if frame_idx == 1 or ((frame_idx - 1) % DETECT_INTERVAL == 0):
            bboxes, _kpss = backend.detect_face(frame)  # Back.py 接口名: detect_face
            new_results = []
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[i]
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                if x2i <= x1i or y2i <= y1i:
                    continue
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    continue
                crop112 = cv2.resize(crop, (112, 112))
                emb = backend.FaceRecognition.getFaceTensor(crop112)
                name, sim = backend.FaceRecognition.match(emb, min_sim=MIN_SIM)
                # 解耦调用2(识别结果 -> sqlite): stranger/低分不写, 当日去重+冷却
                ok, reason = try_sign_in(DB_PATH, name, name, float(sim),
                                         cam_id=CAM_ID, min_sim=MIN_SIM,
                                         cooldown_s=COOLDOWN_S)
                new_results.append({"bbox": (x1i, y1i, x2i, y2i),
                                    "name": name, "sim": float(sim),
                                    "signed": ok, "reason": reason})
                if ok:
                    print(f"[sign] {name} sim={sim:.2f} 已签到")
            face_results = new_results

        for fr in face_results:
            x1i, y1i, x2i, y2i = fr["bbox"]
            color = (0, 255, 0) if fr["signed"] or fr["reason"] in ("duplicate-today", "cooldown") else (0, 0, 255)
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(frame, f"{fr['name']}:{fr['sim']:.2f}:{fr['reason']}",
                        (x1i, max(0, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"today={len(fetch_today(DB_PATH))} q to quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("demo_attendance - q to quit", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[demo_attendance] 今日记录:", fetch_today(DB_PATH))


if __name__ == "__main__":
    main()
