import cv2
import requests
import numpy as np
import math
import time
import threading
import queue
import os

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

img_bytes_queue = queue.Queue(maxsize=1)  # 用于传输帧
result_dict = {"hands": [], "boxes": [], "faces": []}
lock = threading.Lock()


th=0#测试输出

error_th=0
all_start_time=time.time()

cap = cv2.VideoCapture(0)
ldm_flag=True
local_list=[""]#本地ip列表
#参数设置
data = {
    "use_onnx": "True",
    "use_nms" : "True",
    "use_hand": "True",
    "use_face": "True"
    }

Real_IP=''


name_img=["up","down","lookdown","ok"]


for i in range(len(local_list)):
    try:
        r = requests.get(fr"http://{local_list[i]}:8000/img/up",timeout=2)
        if r.status_code == 200:
            Real_IP=fr"http://{local_list[i]}:8000/"
            break
    except:
        continue
    

#表情

img_dict={}

for i in range(len(name_img)):
    try:
        response = requests.get(fr"{Real_IP}img/{name_img[i]}")
        if response.status_code == 200:#正常读取
            img_bytes = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img_dict[fr"{name_img[i]}"]=img
        else:
            print(fr"Failed to load image: {name_img[i]}")
    except Exception as e:
        print("e")
        

img_fg,img_mask,img_inv={},{},{}#存储处理后的图片信息
lower = np.array([0, 0, 0])
upper = np.array([50, 50, 50])
    
for key,img in img_dict.items():
    mask = cv2.inRange(img, lower, upper)  # 黑色区域掩码
    mask_inv = cv2.bitwise_not(mask)       # 非黑色区域掩码
    fg = cv2.bitwise_and(img, img, mask=mask_inv)
    img_fg[name_img[i]]=fg
    img_mask[name_img[i]]=mask
    img_inv[name_img[i]]=mask_inv


def infer_thread_func():#子进程
    global th
    global error_th
    while True:
        if not img_bytes_queue.empty():
            img_bytes = img_bytes_queue.get()
            start_time = time.time()  # 记录当前帧开始时间
            try:
                response = requests.post(fr"{Real_IP}infer/", files={"img": ("image.jpg", img_bytes, "image/jpeg")}, data=data,timeout=10.0,proxies={"http": None, "https": None})#超时处理
                if response.status_code == 200:
                    result = response.json()
                    
                        
                    with lock:
                        result_dict["hands"] = result.get("hands", [])
                        result_dict["boxes"] = result.get("boxes", [])
                        result_dict["faces"] = result.get("faces", [])
                        result_dict["time"]  = result.get("time",[])
                        result_dict["error"] = result.get("error",[])
                        result_dict["scale"] = result.get("scale",[])
                        
                        end_time = time.time()
                        fps = 1 / (end_time - start_time + 1e-5)  # 防止除0
                        fps_text = f"{fps:.1f}"
                        result_dict["func_fps"]=fps_text
                        
                        if th==1:#调试
                            print(rf"第{error_th}次输出")
                            error_th+=1
                            if result_dict["boxes"]:
                                print("box")
                                for item in result_dict["boxes"]:
                                    print("boxes:", item)
                            if result_dict["faces"]:
                                print("face")
                                for item in result_dict["faces"]:
                                    print("faces:", item)
                            if result_dict["hands"]:
                                print("hand")
                                for item in result_dict["hands"]:
                                    # print("hands:", item)
                                    for i, point in enumerate(landmarks):
                                        print(f"  landmark[{i}]: x={point['x']}, y={point['y']}, z={point['z']:.4f}")
                            if result_dict["time"]:
                                run_time=result_dict["time"]
                                print("time")
                                onnx_time=run_time.get("onnx")
                                hand_time=run_time.get("hand")
                                face_time=run_time.get("face")
                                all_time=run_time.get("all")
                                if onnx_time:
                                    print(f"onnx time = {onnx_time}")
                                if hand_time:
                                    print(f"hand time = {hand_time:}")
                                if face_time:
                                    print(f"face time = {face_time:}")
                                print(f"all time = {all_time:}")
                            if result_dict["error"]:
                                for i in result_dict["error"]:
                                    print(i)
                                
                                
                            th=0
                            
                            
                else:
                    error_end_time=time.time()
                    error_time=error_end_time-all_start_time
                    print(f"errot time = {error_time:.4f} ""网络错误")
                    th=1
            except Exception as e:
                print("网络请求失败:", e)

infer_thread = threading.Thread(target=infer_thread_func, daemon=True)
infer_thread.start()#启动子进程

def put_latest_jpeg(img_bytes):
    while True:
        try:
            img_bytes_queue.get_nowait()  # 丢弃旧帧
        except queue.Empty:
            break
    img_bytes_queue.put_nowait(img_bytes)



while True:
    start_time = time.time()  # 记录当前帧开始时间
    
    
    ret, frame = cap.read()
    if not ret:
        print("无法接收视频帧，退出")
        break
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    
    
    with lock:
        result = result_dict.copy()#读取子进程的信息
    
    flag1_toe=False#检察大拇指
    flag1_up,flag1_down=False,False#大拇指上下
    flag2_lookdown=False#检察中指
    flag_ok=False
    boxes = result.get("boxes",[])#取识别框
    hands = result.get("hands", [])#List[Dict[str, Union[str, List[Dict[str, Union[int, float]]]]]]
    faces=result.get("faces",[])#取人脸识别结果
    run_time=result.get("time")
    error_list=result.get("error")
    
    func_fps=result.get("func_fps",0)#子进程帧率
    
    if hands:
        for hand in hands:
            label = hand.get("label", "Right")  # "Left" or "Right"
            color = (0, 255, 255) if label == "Left" else (255, 0, 0)  # Yellow or Blue
            landmarks = hand.get("landmarks", [])
            for point in landmarks:
                x = point["x"]
                y = point["y"]
                cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)
                #radius=3：点的半径，适中,thickness=-1：填充圆
            if not flag1_toe:
                if landmarks[3]["x"]>0 and landmarks[3]["y"]>0:#检测到拇指指第二关节（比指尖稳定
                    x0,y0=landmarks[0]["x"],landmarks[0]["y"]#手腕的横纵坐标
                    s=math.hypot(landmarks[4]["x"]-x0,landmarks[4]["y"]-y0)
                    if math.hypot(landmarks[12]["x"]-x0,landmarks[12]["y"]-y0) < s and math.hypot(landmarks[8]["x"]-x0,landmarks[8]["y"]-y0) < s and math.hypot(landmarks[16]["x"]-x0,landmarks[16]["y"]-y0) < s and math.hypot(landmarks[20]["x"]-x0,landmarks[20]["y"]-y0) < s:
                    #检测其他四个指尖是否靠近手腕               
                        flag1_toe=True
            if flag1_toe:
                x0,y0=landmarks[0]["x"],landmarks[0]["y"]#手腕的横纵坐标
                x1,y1=landmarks[5]["x"],landmarks[5]["y"]#食指根节点的横纵坐标
                if landmarks[4]["y"]<y0 and landmarks[4]["y"]<y1:
                    flag1_up=True
                else:
                    flag1_down=True
            if not flag2_lookdown:
                if landmarks[12]["x"]>0 and landmarks[12]["y"]>0:
                    x0,y0=landmarks[0]["x"],landmarks[0]["y"]#手腕的横纵坐标
                    s=math.hypot(landmarks[12]["x"]-x0,landmarks[12]["y"]-y0)
                    if math.hypot(landmarks[4]["x"]-x0,landmarks[4]["y"]-y0) < s and math.hypot(landmarks[8]["x"]-x0,landmarks[8]["y"]-y0) < s and math.hypot(landmarks[16]["x"]-x0,landmarks[16]["y"]-y0) < s and math.hypot(landmarks[20]["x"]-x0,landmarks[20]["y"]-y0) < s:
                        #检测中指耸立
                        flag2_lookdown=True
            if not flag_ok:
                if landmarks[4]["x"]>0 and landmarks[4]["y"] >0 and landmarks[8]["x"]>0 and landmarks[8]["y"] >0:#检测食指和拇指指尖
                    s=math.hypot(landmarks[8]["x"]-landmarks[4]["x"],landmarks[8]["y"]-landmarks[4]["y"])#拇指与食指指尖距离
                    
                    s1=math.hypot(landmarks[12]["x"]-landmarks[8]["x"],landmarks[12]["y"]-landmarks[8]["y"])#食指与中指指尖距离
                    s2=math.hypot(landmarks[16]["x"]-landmarks[12]["x"],landmarks[16]["y"]-landmarks[12]["y"])#中指与无名指指尖距离
                    s3=math.hypot(landmarks[20]["x"]-landmarks[16]["x"],landmarks[20]["y"]-landmarks[16]["y"])#无名指与小指指尖距离
                    
                    if s<s1 and s<s2 and s<s3:flag_ok=True
                
            
            
    if  boxes:
        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes[index]["x1"], boxes[index]["y1"], boxes[index]["x2"], boxes[index]["y2"]
            confidence = boxes[index].get("confidence", 0)
            ldm=boxes[index].get("landmarks",[])
            scale=boxes[index].get("scale",1)
            #确认表情
            if  flag1_up or flag1_down or flag2_lookdown or flag_ok:
                img=[]
                Real_Name=""
                if flag1_up:
                    Real_Name="up"
                    if img_dict.get(Real_Name):
                        img=img_dict.get(Real_Name)
                        fg = img_fg[Real_Name]
                        mask = img_mask[Real_Name]
                        mask_inv = img_inv[Real_Name]
                    else:
                        print(fr"{Real_Name} is None")
                elif flag1_down:
                    Real_Name="down"
                    if img_dict.get(Real_Name):
                        img=img_dict.get(Real_Name)
                        fg = img_fg[Real_Name]
                        mask = img_mask[Real_Name]
                        mask_inv = img_inv[Real_Name]
                    else:
                        print(fr"{Real_Name} is None")
                elif flag2_lookdown:
                    Real_Name="lookdown"
                    if img_dict.get(Real_Name):
                        img=img_dict.get(Real_Name)
                        fg = img_fg[Real_Name]
                        mask = img_mask[Real_Name]
                        mask_inv = img_inv[Real_Name]
                    else:
                        print(fr"{Real_Name} is None")
                elif flag_ok:
                    Real_Name="ok"
                    if img_dict.get(Real_Name):
                        img=img_dict.get(Real_Name)
                        fg = img_fg[Real_Name]
                        mask = img_mask[Real_Name]
                        mask_inv = img_inv[Real_Name]
                    else:
                        print(fr"{Real_Name} is None")
                else:
                    fg = mask = mask_inv = None
                
                w, h = abs(x2 - x1), abs(y2 - y1)
                fg_resized = cv2.resize(fg, (w, h))
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_inv_resized = cv2.bitwise_not(mask_resized)

                roi = frame[y1:y2, x1:x2]
                bg = cv2.bitwise_and(roi, roi, mask=mask_resized)#取背景
                combined = cv2.add(bg, fg_resized)

                frame[y1:y2, x1:x2] = combined
            else :
                cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 0, 255 ), 2)
                cv2.putText(frame, f"conf = {confidence:.4f}", (x2+10,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), thickness=1, lineType=None)
                if ldm_flag:
                    if ldm:
                        for point in ldm:
                            x=point[0]
                            y=point[1]
                            cv2.circle(frame, (x, y), radius=2, color=(255, 128, 0), thickness=-1)
                            #radius=3：点的半径，适中,thickness=-1：填充圆
            if faces:
                if index<len(faces):
                    x0,y0=abs(x2+x1)//2,y1-10 if y1-10 >0 else 0
                    cv2.putText(frame, faces[index], (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), thickness=1, lineType=None)
    if run_time:
        # print("run_time:", run_time)
        onnx_time=run_time.get("onnx")
        hand_time=run_time.get("hand")
        face_time=run_time.get("face")
        all_time=run_time.get("all")
        all_fps=1/all_time
        cv2.putText(frame, f"{all_time:.4f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if onnx_time:
            cv2.putText(frame,f"{onnx_time:.4f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if hand_time:
            cv2.putText(frame,f"{hand_time:.4f}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if face_time:
            cv2.putText(frame,f"{face_time:.4f}",(190, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
        # cv2.putText(frame,f"hou duan fps {all_fps:.1f}", (frame.shape[1]-200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,f"{all_fps:.1f}", (frame.shape[1]-70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    if error_list:
        for i in error_list:
            print(i)
    
    if img_bytes_queue.full():
        try:
            img_bytes_queue.get_nowait()
        except:
            pass
    # img_bytes_queue.put_nowait(frame.copy())#读取最新帧
    put_latest_jpeg(img_bytes)
            
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-5)  # 防止除0
    fps_text = f"{fps:.1f}"
    # cv2.putText(frame, f"zhu xian cheng fps {fps_text}", (frame.shape[1]-200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (frame.shape[1]-70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    if func_fps:
        # cv2.putText(frame, f"zi xian cheng fps {func_fps}", (frame.shape[1]-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, func_fps, (frame.shape[1]-70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    
            
    cv2.imshow("Detection Result", frame)
    key=cv2.waitKey(1)& 0xFF
    if key == 27:#ESC
        break
    if key == ord('q'):
        th=1
cv2.destroyAllWindows()