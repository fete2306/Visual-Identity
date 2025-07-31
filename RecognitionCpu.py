import sys
import os
sys.path.append(r"./yolov5-face-master/")

import cv2
import numpy as np
import onnxruntime as ot
import mediapipe as mp
from typing import Any,List,Union,Dict
from numpy.typing import NDArray

class Check_onnx:
    
    def __init__(self,model_path:str=r".\yolov5-face-master\weights\yolov5-blazeface.onnx",
                 input_size:int=640,
                 check_num:int=3,
                 
                 static_image_mode=False,
                 max_num_hands:int=2,
                 min_detection_confidence:float=0.5,
                 min_tracking_confidence:float=0.5,
                 
                 face_model_path:str=r".\Model\glintasia_r50.onnx",
                 face_input:int=112,
                 face_bank:str=r""
                 ):
        
        self.input_size=input_size
        self.model_path=model_path
        self.num=check_num
        self.model = ot.InferenceSession(model_path)
        self.check_hand=False
        self.check_face=False
        
        
        self.mp_hands=None
        self.hands=None
            
        self.static_image_mode=static_image_mode
        self.max_num_hands=max_num_hands
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
            
        #延迟启用（？）
        self.face_model_path=face_model_path
            
        self.face_input=face_input
        self.face_bank=face_bank
            
        self.face_model = None
        self.face_model_name = None
        self.face_dict = None
            
            
    def ensure_hands(self):
        if not self.check_hand:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
                )
            self.check_hand=True
            
    def ensure_face(self):
        if not self.check_face:
            self.face_model = ot.InferenceSession(self.face_model_path)
            self.face_model_name=self.face_model.get_inputs()[0].name#获取模型输入张量名，用于传入ort
            self.face_data = np.load(self.face_bank)
            self.face_dict = {key: arr for key, arr in self.face_data.items()}#启用人脸库
            self.face_num=len(self.face_dict)#脸的数量
            self.check_face=True
    
    def pre(self,img:np.ndarray) -> np.ndarray:
        #接受np数组形式的img，进行初始化
        img_resized = cv2.resize(img, (self.input_size, self.input_size))#调整大小适配模型
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)#将BGR转成用于模型的RGB
        img_normalized = img_rgb.astype(np.float32) / 255.0#归一化，生成浮点数组成的图像
        img_transposed = np.transpose(img_normalized, (2, 0, 1))#改变通道顺序，从HWC转到CHW
        blob = np.expand_dims(img_transposed, axis=0)#增加一个batch维度
        #总之就是处理成onnx接受的标准格式
        return blob
    
    def solve(self,img:np.ndarray,blob:np.ndarray,NMS:bool=False) -> List[Dict[str, Union[int, float]]]:
        #使用onnx模型进行识别
        inputs = {self.model.get_inputs()[0].name: blob}
        outputs = self.model.run(None, inputs)
        output = outputs[0]
        
        if output is None or len(output) == 0:
            return []
        
        output = np.squeeze(output, axis=0)
        
        if output.ndim == 1 and output.shape[0] != 5:
            return []
        
        if output.ndim == 1:
            output = np.expand_dims(output, axis=0)
            
        N = output.shape[0]#识别框的个数
        if N == 0:
            return []
        
        #排序识别框的得分
        conf_list=[]
        for i in range(int(N)):
            confidence = output[i,4]
            if confidence < 0.1:  # 忽略低得分
                    continue
            conf_list.append((i,confidence))
            
        conf_sort=sorted(conf_list,key=lambda x: x[1],reverse=True)
        if not NMS:
            #取前num个识别框
            LEN=self.num if self.num <len(conf_sort) else len(conf_sort)
            result=[]
            for t in range(LEN):
                #提取坐标，得分
                i=conf_sort[t][0]
                confidence = output[ i, 4]
                x_c = output[ i, 0]
                y_c = output[ i, 1]
                w = output[ i, 2]
                h = output[ i, 3]
                x1 = int(x_c - w/2)
                y1 = int(y_c - h/2)
                x2 = int(x_c + w/2)
                y2 = int(y_c + h/2)
                #翻转成原坐标
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                scale_x = img.shape[1] / self.input_size
                scale_y = img.shape[0] / self.input_size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                x1 = max(0, min(x1, img.shape[1]-1))
                x2 = max(0, min(x2, img.shape[1]-1))
                y1 = max(0, min(y1, img.shape[0]-1))
                y2 = max(0, min(y2, img.shape[0]-1))
                confidence = float(confidence)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                #返回字典列表，每个字典包含框的坐标和得分
                result.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"confidence":confidence})
            return result
        else:
            boxes_raw = []     # 用于 NMS：原始框
            confidences = []   # 用于 NMS：置信度
            result = []

            for t in range(len(conf_sort)):
                i = conf_sort[t][0]
                confidence = float(output[i, 4])

                x_c, y_c, w, h = output[i, 0:4]
                x1 = int((x_c - w / 2) * img.shape[1] / self.input_size)
                y1 = int((y_c - h / 2) * img.shape[0] / self.input_size)
                x2 = int((x_c + w / 2) * img.shape[1] / self.input_size)
                y2 = int((y_c + h / 2) * img.shape[0] / self.input_size)

                x1 = max(0, min(x1, img.shape[1] - 1))
                y1 = max(0, min(y1, img.shape[0] - 1))
                x2 = max(0, min(x2, img.shape[1] - 1))
                y2 = max(0, min(y2, img.shape[0] - 1))

                boxes_raw.append([x1, y1, x2 - x1, y2 - y1])  # 注意：NMSBoxes 要 xywh 格式
                confidences.append(confidence)

            # 应用 NMS
            indices = cv2.dnn.NMSBoxes(boxes_raw, confidences, score_threshold=0.3, nms_threshold=0.4)
            if len(indices) == 0:
                return []
            for idx in indices:
                i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                x, y, w, h = boxes_raw[i]
                result.append({
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h),
                    "confidence": float(confidences[i])
                })
                
            return result
    def solve_hands(self,img:np.ndarray)->List[Dict[str, Union[str, List[Dict[str, Union[int, float]]]]]]:
        if not self.check_hand:
            self.ensure_hands()#启用模型
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#只需要变成rgb就能使用
        results = self.hands.process(img_rgb)
        hands_detected = results.multi_hand_landmarks
        num_hands = len(hands_detected) if hands_detected else 0
        h, w = img.shape[:2]
        
        result=[]#最终输出的列表
        
        for i in range(num_hands):
            points={}#存储手部信息的字典
            points["label"]=results.multi_handedness[i].classification[0].label#left/right，字符串
            landmarks = results.multi_hand_landmarks[i]
            lm_list = []#存储手部21个节点内容的列表
            for idx, lm in enumerate(landmarks.landmark):
                lm_list.append({
                    "id": idx,#节点序号
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z  # 原始 z 值
                })
            points["landmarks"] = lm_list
            result.append(points)
            
        return result
    def similarity(self,vec1,vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)) 
    
    def solve_face(self,img:np.ndarray,result:List[dict],cut_size:int=20):
        if not self.check_face:
            self.ensure_face()#启用模型
        face_result=[]
        for i in result:#取识别框
            x1,y1,x2,y2=i.get("x1"),i.get("y1"),i.get("x2"),i.get("y2")
            y1_new = max(0, y1 - cut_size)
            x1_new = max(0, x1 - cut_size)
            y2_new = min(img.shape[0] - 1, y2 + cut_size)
            x2_new = min(img.shape[1] - 1, x2 + cut_size)
            
            face_crop = img[y1_new:y2_new, x1_new:x2_new]#切下来识别框
            
            if face_crop.size == 0 or (y2 - y1) < 20 or (x2 - x1) < 20:
                continue  # 太小的不处理
            #face预处理
            face_resized = cv2.resize(face_crop, (self.face_input, self.face_input))#调整大小适配模型
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)#将BGR转成用于模型的RGB
            face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            face_blob = np.expand_dims(face_transposed, axis=0)
            
            embedding_output = self.face_model.run(None, {self.face_model_name: face_blob})#按照输入张量运行模型
            embedding = embedding_output[0][0]
            embedding = embedding / np.linalg.norm(embedding)
            label=""
            flag_face=False
            for key,arr in self.face_dict.items():
                sim=self.similarity(embedding,arr)
                if sim >0.2 and flag_face == False:
                    label=key
                    flag_face=True
                    break
            if flag_face:
                face_result.append(label)
            else :face_result.append("unknow")
        return face_result
        

