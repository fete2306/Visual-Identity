# Https 版后端: 接口与父目录 app.py 完全一致 (/infer/batch + /infer/single),
# 另挂载 ./Front 实现 https 同源 (页面与接口同域: 无 mixed content, 无 CORS 烦恼).
# Back.py/Model/Data/certs 均引用父目录, 不复制. 运行 (cwd 必须为 D:\Visual-Identity_r):
#   D:\conda\envs\cuda128\python.exe -m uvicorn Https.app:app --host 0.0.0.0 --port 8060 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
# 页面: https://192.168.178.142:8060/front/front.html (先开同域 /docs 点“继续前往”完成自签信任)
import sys
import os

import cv2
import numpy as np
import onnxruntime as ot
import mediapipe as mp
from numpy.typing import NDArray

from typing import Any, List, Union, Dict, Tuple, Optional

# ---- Https 变体: 静态目录按本文件定位, back 走父目录 (Model/Data 相对路径依赖 cwd=项目根) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Https
sys.path.insert(0, os.path.dirname(BASE_DIR))

from Back import FaceRecognition

from fastapi import FastAPI,UploadFile,File, Form,Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Body

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

import time
import json
import atexit


FaceBlob = NDArray[np.float32]       # (N,3,112,112) 输入
FaceEmbedding = NDArray[np.float32]  # (N,512) 已L2归一输出
FaceImage = NDArray[np.uint8]        # (H,W,3) BGR原图

checker=FaceRecognition(faceEmbeddingsPath=r"Data\FaceEmbedding\Default.npz")
recognitionCount=20
atexit.register(checker.save)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以改成特定域名列表，生产环境建议这样做
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 OPTIONS、POST、GET等
    allow_headers=["*"],
)

# 同源静态: https://host:8060/front/front.html (API 路由优先匹配, mount 只接剩下的)
app.mount("/front", StaticFiles(directory=os.path.join(BASE_DIR, "Front"), html=True), name="front")

# README 托管: https://host:8060/readme (有 markdown 库则渲染, 否则原样 <pre>)
README_PATH = os.path.join(BASE_DIR, "README.md")
if not os.path.isfile(README_PATH):
    README_PATH = os.path.join(os.path.dirname(BASE_DIR), "README.md")

@app.get("/readme", response_class=HTMLResponse)
async def readme():
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return HTMLResponse("<h1>README.md 缺失</h1>", status_code=404)
    try:
        import markdown as _md
        body = _md.markdown(text, extensions=["fenced_code", "tables"])
    except Exception:
        import html as _html
        body = "<pre>" + _html.escape(text) + "</pre>"
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Visual-Identity README</title></head>"
        "<body style='max-width:900px;margin:2em auto;font-family:sans-serif'>"
        + body + "</body></html>")

@app.post("/infer/batch")
async def infer_batch(request: Request, min_sim: float = 0.4):
    t0 = time.time()
    form = await request.form()
    
    out = {}
    try:
        for key, val in form.multi_items()[:recognitionCount]:
            if not isinstance(val, StarletteUploadFile):
                continue
            try:
                raw = await val.read()
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    out[str(key)] = ["unknow", 0.0]
                    continue
                try:
                    emb = await run_in_threadpool(checker.getFaceTensor, img)  # 贵的放进去
                    name, sim = checker.match(emb, min_sim=min_sim)            # match 只是小矩阵乘, 留同步即可
                    out[str(key)] = [name, float(sim)]
                finally:
                    del img  # 理由见上: 给异常路径松手, 正常路径靠重赋值已释放
            except Exception:
                out[str(key)] = ["unknow", 0.0]  # 单张坏图不炸整批, 理由: 批量接口按索引返回, 缺一张补 unknow 比整体 500 合理
            finally:
                try:
                    await val.close()  # 临死文件逐个关, 不靠 GC
                except Exception:
                    pass
    finally:
        try:
            await form.close()  # 兜底全关: 上面漏掉的、非 UploadFile 混入的, 一次清干净
        except Exception:
            pass
    out_ = {f"{k}": [...] for k, v in form.multi_items()[recognitionCount:] if isinstance(v, StarletteUploadFile)}
    if out_ :
        out= out_ | out
    return {"time": time.time() - t0, "results": out}

@app.post("/infer/single")
async def infer(img: UploadFile = File(...), min_sim=0.4):
    results = {}
    startTime = time.time()
    
    try:
        contents = await img.read()
        img_ = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        try:
            res = checker.match(checker.getFaceTensor(img_), min_sim=min_sim)
            results["results"] = res
        finally:
            del img_  # 显式删除图像数组
    finally:
        # 确保文件对象被关闭
        if not img.file.closed:
            img.file.close()
    
    endTime = time.time()
    results["time"] = (endTime - startTime)
    
    return results

@app.post("/infer/save")
async def infer_save(img: UploadFile = File(...), name: str = ""):
    """载入人脸: 对标 Back.FaceCheckWithMediapipe.solve() 按 's' 存脸流程.

    Back 流程: 输入name -> 选第几张脸 crop -> getFaceTensor -> add(name, emb) -> 退出时 save().
    这里前端已裁好单人脸 crop, 后端只做 解码->getFaceTensor->add, 内存即时生效 (后续 match 可见),
    磁盘经 atexit.save 兜底 + 本次主动 save 一次 (低频, 失败不影响本次返回).

    调用: POST /infer/save?name=xxx  multipart {img: jpeg}
    返回: {"ok": True, "name": xxx, "total": 库总量, "time": 秒}
    """
    t0 = time.time()
    try:
        name = (name or "").strip()
        if not name:
            return JSONResponse({"ok": False, "error": "name 为空"}, status_code=400)
        raw = await img.read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return JSONResponse({"ok": False, "error": "图片解码失败"}, status_code=400)
        try:
            emb = await run_in_threadpool(checker.getFaceTensor, arr)  # ONNX阻塞, 扔线程池 (同 /infer/batch)
            checker.add(name, emb=emb)
        finally:
            try:
                del arr
            except Exception:
                pass
        save_warning = None
        try:
            checker.save()  # 主动落盘一次, 崩溃也不丢本次录入; 正常退出时 atexit 再存
        except Exception as e:
            save_warning = str(e)
        out = {"ok": True, "name": name, "total": len(checker.ids), "time": time.time() - t0}
        if save_warning:
            out["save_warning"] = save_warning
        return out
    finally:
        try:
            await img.close()
        except Exception:
            try:
                if not img.file.closed:
                    img.file.close()
            except Exception:
                pass

@app.post("/infer/del")
async def infer_del(name: str = ""):
    t0 = time.time()
    name = (name or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name 为空"}, status_code=400)
    try:
        checker.delFace(name)
    finally:
        pass
    save_warning = None
    try:
        checker.save()  # 主动落盘一次, 崩溃也不丢本次录入; 正常退出时 atexit 再存
    except Exception as e:
        save_warning = str(e)
    out = {"ok": True, "name": name, "total": len(checker.ids), "time": time.time() - t0}
    if save_warning:
        out["save_warning"] = save_warning
    return out
