"""缺失模型并行补全脚本 (仅标准库, 无第三方依赖).

用法 (start.bat 会自动调用, 也可手动跑):
    python tools/download_models.py [--root 项目根目录] [--workers 4] [--check-only]

逻辑:
    检查 MODEL_MANIFEST 中的每个文件, 缺失或为空才下载;
    多个缺失文件用 ThreadPoolExecutor 并行下载 (文件间无依赖, 可并行).
"""
import argparse
import concurrent.futures
import os
import sys
import urllib.request

# 相对路径 -> 下载地址 (文件名与仓库内现有文件一一对应)
MODEL_MANIFEST = {
    # SCRFD 人脸检测 (base1.py 默认 det_500m, 2.5g 为高精度备选)
    os.path.join("Model", "SCRFD", "det_500m.onnx"):
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx",
    os.path.join("Model", "SCRFD", "det_2.5g.onnx"):
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
    # ArcFace R50 人脸识别 (base1.py FaceRecognition 默认)
    os.path.join("Model", "buffalo_l", "w600k_r50.onnx"):
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx",
    # MediaPipe 手势 (与 base1.py 内置自动下载地址一致)
    os.path.join("Model", "hand_landmarker.task"):
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

MIN_VALID_SIZE = 1024  # 小于此字节视为损坏, 触发重下
RETRIES = 3
TIMEOUT = 120
CHUNK = 1024 * 256


def is_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) >= MIN_VALID_SIZE


def download_one(url: str, dest: str) -> str:
    """下载单个文件, 返回 dest. 失败抛异常 (由调用方统计)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".downloading"
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
            if not is_ok(tmp):
                raise IOError("下载后文件过小, 疑似损坏")
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001 需重试所有网络/IO异常
            last_err = e
            print(f"  [retry {attempt}/{RETRIES}] {os.path.basename(dest)}: {e}",
                  flush=True)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass
    raise RuntimeError(f"{dest}: {last_err}")


def find_missing(root: str, manifest=None):
    manifest = manifest or MODEL_MANIFEST
    return [(rel, url) for rel, url in manifest.items()
            if not is_ok(os.path.join(root, rel))]


def download_all(root: str, manifest=None, workers: int = 4):
    """并行下载缺失模型. 返回 (ok_list, fail_list)."""
    manifest = manifest or MODEL_MANIFEST
    missing = find_missing(root, manifest)
    if not missing:
        print("[models] 全部模型已存在, 无需下载.")
        return [], []
    print(f"[models] 缺失 {len(missing)} 个, {workers} 线程并行下载 ...")
    ok, fail = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut2rel = {ex.submit(download_one, url, os.path.join(root, rel)): rel
                   for rel, url in missing}
        for fut in concurrent.futures.as_completed(fut2rel):
            rel = fut2rel[fut]
            try:
                fut.result()
                print(f"  [ok] {rel}")
                ok.append(rel)
            except Exception as e:  # noqa: BLE001 汇总失败项, 最后统一返回
                print(f"  [fail] {rel}: {e}")
                fail.append(rel)
    return ok, fail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="缺失模型并行补全")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-only", action="store_true", help="只检查, 不下载")
    ap.add_argument("--log", default=None, help="同时把输出写到该日志文件")
    ap.add_argument("--rc-file", default=None, help="结束时把退出码写到该文件(供start.bat后台等待)")
    args = ap.parse_args(argv)

    import builtins
    log_f = None
    real_print = builtins.print
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
        log_f = open(args.log, "w", encoding="utf-8")

        def tee_print(*a, **k):
            k.setdefault("flush", True)
            real_print(*a, **k)
            try:
                log_f.write(" ".join(str(x) for x in a) + k.get("end", "\n"))
                log_f.flush()
            except OSError:
                pass

        builtins.print = tee_print

    rc = 0
    try:
        missing = find_missing(args.root)
        if args.check_only:
            for rel, _url in missing:
                print(f"[missing] {rel}")
            if not missing:
                print("[models] 全部模型已存在.")
            rc = 1 if missing else 0
        else:
            _ok, fail = download_all(args.root, workers=args.workers)
            if fail:
                print(f"[models] {len(fail)} 个下载失败, 请检查网络后重跑 start.bat.")
                rc = 1
            else:
                print("[models] 模型齐全.")
                rc = 0
    finally:
        builtins.print = real_print
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass
        if args.rc_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.rc_file)) or ".", exist_ok=True)
                with open(args.rc_file, "w", encoding="utf-8") as f:
                    f.write(str(rc))
            except OSError:
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
