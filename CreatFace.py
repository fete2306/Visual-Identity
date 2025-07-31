import cv2
import onnxruntime as ort
import numpy as np

model_path=r".\yolov5-face-master\weights\yolov5-blazeface.onnx"
model_path2=r".\Model\glintasia_r50.onnx"
model = ort.InferenceSession(model_path)
model2 = ort.InferenceSession(model_path2)
input_name2 = model2.get_inputs()[0].name#获取模型输入张量名，用于传入ort
input_size=640#人脸检测大小
check_size=112#人脸识别大小


face_name="HOST0"#当前人脸名

min_sim=0.8

face_path=r""#写入自己的人脸库路径（npz)
num=3#识别框数目
face_dict={}
face_num=0
face_list=[]
try:
    face_data = np.load(face_path,allow_pickle=True)
except Exception as e:
    print(e)
else:
    for key,arr in face_data.items():
        face_dict[key]=arr
        if key==face_name:
            if arr.size>0:
                for i in arr:
                    face_list.append(i)
    face_num=len(face_dict)#脸的数量


cap = cv2.VideoCapture(0)

th=0#计数
cut_size=20#额外增加剪贴的识别框大小
def similarity(vec1,vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)) 
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收视频帧，退出")
        break
    img_resized = cv2.resize(frame, (input_size, input_size))#调整大小适配模型
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)#将BGR转成用于模型的RGB
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    blob = np.expand_dims(img_transposed, axis=0)
    
    inputs = {model.get_inputs()[0].name: blob}
    outputs = model.run(None, inputs)
    output = outputs[0]
    output = np.squeeze(output, axis=0)
    
    
    N = output.shape[0]#识别框的个数

    boxes=[]
    conf_list=[]
    for i in range(output.shape[0]):
        confidence = output[i, 4]
        if confidence > 0.3:
            x_c, y_c, w, h = output[i, :4]
            x = int(x_c - w / 2)
            y = int(y_c - h / 2)
            boxes.append([x, y, int(w), int(h)])
            conf_list.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(boxes, conf_list, score_threshold=0.3, nms_threshold=0.4)
    flag_indices=False
    if len(indices)>1:flag_indices=True
    for idx in indices:
        if flag_indices == True:
            print("超过一张脸")
            break
        i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
        x, y, w, h = boxes[i]
        confidence=conf_list[i]
        

        x1 = int(x )
        y1 = int(y )
        x2 = int(x + w)
        y2 = int(y + h)
        #翻转成原坐标
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        scale_x = frame.shape[1] / input_size
        scale_y = frame.shape[0] / input_size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        x1 = max(0, min(x1, frame.shape[1]-1))
        x2 = max(0, min(x2, frame.shape[1]-1))
        y1 = max(0, min(y1, frame.shape[0]-1))
        y2 = max(0, min(y2, frame.shape[0]-1))
        confidence = float(confidence)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        y1_new = max(0, y1 - cut_size)
        x1_new = max(0, x1 - cut_size)
        y2_new = min(frame.shape[0] - 1, y2 + cut_size)
        x2_new = min(frame.shape[1] - 1, x2 + cut_size)
        
        face_crop = frame[y1_new:y2_new, x1_new:x2_new]#切下来识别框
        if face_crop.size == 0 or (y2 - y1) < 20 or (x2 - x1) < 20:
            continue  # 太小的不处理
        #face预处理
        face_resized = cv2.resize(face_crop, (check_size, check_size))#调整大小适配模型
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)#将BGR转成用于模型的RGB
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        face_blob = np.expand_dims(face_transposed, axis=0)
        
        
        embedding_output = model2.run(None, {input_name2: face_blob})#按照输入张量运行模型
        embedding = embedding_output[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        
        label=""      
        flag_face=False
        sim=0.0
        for key,arr in face_dict.items():
            if flag_face==True:
                break
            if arr.size>0:
                for i in range(len(arr)):
                    sim=float(similarity(embedding,arr[i]))
                    if sim >min_sim and flag_face == False:
                        label=key
                        flag_face=True
                        break
        if not flag_face:
            for i in face_list:
                sim=float(similarity(embedding,i))
                if sim >min_sim and flag_face == False:
                    label=face_name
                    flag_face=True
                    break
        if not flag_face:
            label=f"{face_name}"
            print(label+" 录入成功 "+f" {sim:.4f} "+f" 当前录入 {face_num}")
            face_num+=1
            face_list.append(embedding)
                
        if label:
            x0,y0=abs(x2+x1)//2,y1-10 if y1-10 >0 else 0
            cv2.putText(frame, label, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), thickness=1, lineType=None)
        # else:cv2.putText(frame, "unknow", (x2+10,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), thickness=1, lineType=None)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 0, 255 ), 2)
    cv2.imshow("Detection Result", frame)
    key=cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下 ESC 键退出（ASCII码27）
        if face_list:
            face_array=np.stack(face_list)
            face_dict[face_name]=face_array
            np.savez(face_path, **face_dict)#将字典展开存入
        break
    if key == ord('p'):
        none_dict={}
        np.savez(face_path, **none_dict)#将字典展开存入
cv2.destroyAllWindows()
        
    
        