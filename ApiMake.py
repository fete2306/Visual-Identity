from fastapi import FastAPI,UploadFile,File, Form,Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from TorchForApi1 import Check_pt
import cv2
import numpy as np
import sys
import traceback
import time
import threading
import queue
import asyncio

frame_queue = queue.Queue(maxsize=1)  # 用于传输图片
hand_result_queue = queue.Queue(maxsize=1)
lock = threading.Lock()


face_path=r""
model=None
model_error=""
try:
    model = Check_pt(face_bank=face_path)
except Exception as e:
    model_error=f"model error is {e}"



def excepthook(type, value, tb):
    print("Uncaught exception:")
    traceback.print_exception(type, value, tb)

sys.excepthook = excepthook



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以改成特定域名列表，生产环境建议这样做
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 OPTIONS、POST、GET等
    allow_headers=["*"],
)

@app.post("/infer/")
async def infer(
    img: UploadFile = File(...),
    use_onnx: bool = Form(True),#人脸检测
    use_hand: bool = Form(False),#手势识别
    use_nms: bool = Form(False),#非极大值抑制
    use_face:bool = Form(False),#人脸识别
    min_sim:float = Form(0.8)
):
    results = {}
    error_list=[]
    
    try:
        face_data = np.load(face_path)
    except Exception as e:
        error_list.append(f"face_data error is {e}")
        use_face=False
    if model_error:
        error_list.append(model_error)
        use_onnx=False
        
    # return {"boxs":[],"hands":[],"faces":[],"time":[]}
    contents = await img.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # if image is None:
        
    #     return {"boxs":[],"hands":[],"faces":[],"time":[],"error":"image error"}
    start_img = time.time()  # 记录当前帧开始时间
    
    

    if use_onnx:
        start_onnx = time.time()  # 记录当前帧开始时间
        try:
            tenser = model.pre(image)
            boxes = model.solve(image, tenser, NMS=use_nms)
            results["boxes"] = boxes
        except Exception as e:
            results["boxes"] = []
            error_list.append(f"onnx error is {e}")
            
        end_onnx = time.time()  # 记录当前帧开始时间
        if use_face:
            start_face=time.time()
            try:
                faces=model.solve_face(image,boxes,min_sim=min_sim)
                results["faces"] = faces
            
            except Exception as e:
                # print("face error ",e)
                results["faces"] = []
                error_list.append(f"face error is {e}")
            end_face=time.time()

        else:
            results["faces"] = []
    else:results["boxes"] = []

    if use_hand:
        start_hand=time.time()
        
        try:
            hands = model.solve_hands(image)
            
            results["hands"]=hands
        except Exception as e:
            results["hands"] = []
            error_list.append(f"hand error is {e}")
        end_hand=time.time()
    else:results["hands"] = []
    
    time_run={}
    if use_onnx:
        time_run["onnx"] = float(end_onnx - start_onnx)
        if use_face:
            time_run["face"] = float(end_face - start_face)
    if use_hand:
        time_run["hand"] = float(end_hand - start_hand)
    end_img=time.time()
    time_run["all"] = float(end_img - start_img)
    
    results["time"] = time_run
    results["error"] = error_list
    return results

@app.get("/img/{name}")
def get_img(name: str):
    img_map = {
        "up": r".\img\1f642.png",
        "down": r".\img\1f621.png",
        "lookdown": r".\img\1f612.png",
        "ok": r".\img\1f60b.png"
    }
    filepath = img_map.get(name)
    if filepath:
        return FileResponse(filepath, media_type="image/png")
    return {"error": "emoji not found"}


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()}
    )

templates = Jinja2Templates(directory="templates")

@app.get("/cam", response_class=HTMLResponse)
async def show_page(request: Request):
    return templates.TemplateResponse("cam1.html", {"request": request})

@app.get("/camHttps", response_class=HTMLResponse)
async def show_page(request: Request):
    return templates.TemplateResponse("camHttos.html", {"request": request})