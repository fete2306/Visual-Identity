"""缺失模型并行补全脚本 (仅标准库, 无第三方依赖).

用法 (start.bat 会自动调用, 也可手动跑):
    python tools/download_models.py [--root 项目根目录] [--workers 4] [--check-only]

逻辑:
    检查 MODEL_MANIFEST 中的每个文件, 缺失或为空才下载;
    多个缺失文件用 ThreadPoolExecutor 并行下载 (文件间无依赖, 可并行).
"""
import argparse
import concurrent.futures
import hashlib
import os
import re
import sys
import urllib.request

# 相对路径 -> 下载地址 (文件名与仓库内现有文件一一对应)
# 版本钉死: tasks-vision==0.10.14 (wasm 大小指纹比对确认, 与 vendor 内 mjs 同版)
_TASKS_VER = "0.10.14"
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
    # 浏览器前端 vendor 大文件 (git忽略, 与 Model 内同名 task 内容一致)
    os.path.join("Front", "vendor", "hand_landmarker.task"):
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_internal.wasm"):
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{_TASKS_VER}/wasm/vision_wasm_internal.wasm",
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_nosimd_internal.wasm"):
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{_TASKS_VER}/wasm/vision_wasm_nosimd_internal.wasm",
}

# 精确大小 (B, 与现网文件一一对应): 命中时按精确值校验, 防截断残留被当成完成
EXPECTED_SIZE = {
    os.path.join("Model", "SCRFD", "det_500m.onnx"): 2524817,
    os.path.join("Model", "SCRFD", "det_2.5g.onnx"): 3292009,
    os.path.join("Model", "buffalo_l", "w600k_r50.onnx"): 174383860,
    os.path.join("Model", "hand_landmarker.task"): 7819105,
    os.path.join("Front", "vendor", "hand_landmarker.task"): 7819105,
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_internal.wasm"): 9423986,
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_nosimd_internal.wasm"): 9294247,
}


# SHA256 (与现网文件一一对应): 大小命中后再比哈希, 防内容损坏但大小恰好的极端情况
EXPECTED_SHA256 = {
    os.path.join("Model", "SCRFD", "det_500m.onnx"):
        "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    os.path.join("Model", "SCRFD", "det_2.5g.onnx"):
        "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
    os.path.join("Model", "buffalo_l", "w600k_r50.onnx"):
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    os.path.join("Model", "hand_landmarker.task"):
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    os.path.join("Front", "vendor", "hand_landmarker.task"):
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_internal.wasm"):
        "f82a8e6c05e08a44cc9f9e7ec5f845935bcbb1b1500ebe8c2f4812fb4e2917dc",
    os.path.join("Front", "vendor", "tasks-vision", "wasm", "vision_wasm_nosimd_internal.wasm"):
        "38b61feab2fd7934e05cbe9f68baa308978a5e3b7f85c1913bb8ae89b8ef8b97",
}


def _safe(s, limit: int = 160) -> str:
    """清洗外部错误文本: 只留安全字符 (bat 回显 start.bat 等待循环会原样 echo)."""
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,:;\[\]/+='-]", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _sha256(path: str) -> str:
    h = hashlib.new("sha256")
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

MIN_VALID_SIZE = 1024  # 小于此字节视为损坏, 触发重下
RETRIES = 3
TIMEOUT = 120
CHUNK = 1024 * 256


def is_ok(path: str, rel=None) -> bool:
    """rel 已知时按精确大小+SHA256双校验 (防截断/损坏残留), 否则按最小值."""
    if not os.path.isfile(path):
        return False
    if rel is not None and rel in EXPECTED_SIZE:
        if os.path.getsize(path) != EXPECTED_SIZE[rel]:
            return False
        if rel in EXPECTED_SHA256:
            return _sha256(path) == EXPECTED_SHA256[rel]
        return True
    return os.path.getsize(path) >= MIN_VALID_SIZE


def download_one(url: str, dest: str, rel=None) -> str:
    """下载单个文件, 返回 dest. 失败抛异常 (由调用方统计)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".downloading"
    name = os.path.basename(dest)
    exp = EXPECTED_SIZE.get(rel) if rel else None
    print(f"  [..] {name} 开始下载..."
          + (f" (共约{exp // 1048576}MB)" if exp else ""), flush=True)
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            written, mark = 0, 32 * 1048576
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    written += len(buf)
                    if written >= mark:
                        print(f"  [..] {name} {written // 1048576}MB"
                              + (f"/{exp // 1048576}MB" if exp else ""), flush=True)
                        mark += 32 * 1048576
            if not is_ok(tmp, rel):
                try:
                    bad = os.path.getsize(tmp)
                except OSError:
                    bad = -1
                raise IOError(f"文件校验失败(实得{bad}B), 疑似截断损坏")
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001 需重试所有网络/IO异常
            last_err = e
            print(f"  [retry {attempt}/{RETRIES}] {name}: {_safe(e)}",
                  flush=True)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass
    raise RuntimeError(f"{dest}: {_safe(last_err)}")


def find_missing(root: str, manifest=None):
    manifest = manifest or MODEL_MANIFEST
    return [(rel, url) for rel, url in manifest.items()
            if not is_ok(os.path.join(root, rel), rel)]


def download_all(root: str, manifest=None, workers: int = 4):
    """并行下载缺失模型. 返回 (ok_list, fail_list)."""
    manifest = manifest or MODEL_MANIFEST
    print(f"[models] 按大小+SHA256校验 {len(manifest)} 个文件 ...", flush=True)
    missing = find_missing(root, manifest)
    if not missing:
        print("[models] 全部模型已存在, 无需下载.")
        return [], []
    print(f"[models] 缺失 {len(missing)} 个, {workers} 线程并行下载 ...")
    ok, fail = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut2rel = {ex.submit(download_one, url, os.path.join(root, rel), rel): rel
                   for rel, url in missing}
        for fut in concurrent.futures.as_completed(fut2rel):
            rel = fut2rel[fut]
            try:
                fut.result()
                size = os.path.getsize(os.path.join(root, rel))
                print(f"  [ok] {rel} ({size // 1048576}MB)")
                ok.append(rel)
            except Exception as e:  # noqa: BLE001 汇总失败项, 最后统一返回
                print(f"  [fail] {rel}: {_safe(e)}")
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
