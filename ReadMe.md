# Visual-Identity

一个基于深度学习的视觉识别系统，支持人脸检测、人脸识别和手势识别功能。

## 🚀 项目特性

- **人脸检测**: 基于YOLOv5-BlazeFace的高效人脸检测
- **人脸识别**: 使用GlintAsia-R50模型进行人脸特征提取和身份识别
- **手势识别**: 集成MediaPipe进行实时手势检测
- **Web API**: 提供RESTful API接口，支持图片上传和实时识别
- **Web界面**: 提供直观的Web界面进行实时视频流识别
- **CPU优化**: 支持CPU推理，无需GPU加速

## 📁 项目结构

```
Visual-Identity/
├── ApiMake.py              # FastAPI后端服务
├── CreatFace.py            # 人脸数据收集工具
├── RecognitionCpu.py       # 核心识别类
├── RecognitionForApi.py    # API专用识别模块
├── TestThread.py           # 多线程测试
├── Model/
│   └── glintasia_r50.onnx  # 人脸识别模型
├── img/                    # 测试图片
├── templates/
│   └── 视觉识别.html       # Web界面
└── yolov5-face-master/     # YOLOv5人脸检测模型
```

## 🛠️ 环境要求

- Python 3.7+
- OpenCV
- ONNX Runtime
- MediaPipe
- FastAPI
- NumPy

## 📦 安装说明

1. 克隆项目

```bash
git clone https://github.com/fete2306/Visual-Identity.git
cd Visual-Identity
```

2. 安装依赖

```bash
pip install opencv-python onnxruntime mediapipe fastapi uvicorn numpy
```

3. 下载模型文件（如果尚未包含）

```bash
# 确保Model目录下有glintasia_r50.onnx文件
# 确保yolov5-face-master/weights目录下有yolov5-blazeface.onnx文件
```

## 🚀 使用方法

### 1. 启动API服务

```bash
python ApiMake.py
```

服务将在 `http://localhost:8000` 启动

### 2. 使用Web界面

访问 `http://localhost:8000/cam` 打开Web界面，支持：

- 实时人脸检测
- 手势识别
- 人脸识别
- 相似度调节

### 3. API接口

#### 图片识别接口

```
POST /infer/
```

参数：

- `img`: 上传的图片文件
- `use_onnx`: 是否启用人脸检测 (默认: True)
- `use_hand`: 是否启用手势识别 (默认: False)
- `use_nms`: 是否启用非极大值抑制 (默认: False)
- `use_face`: 是否启用人脸识别 (默认: False)
- `min_sim`: 人脸识别相似度阈值 (默认: 0.8)

### 4. 人脸数据收集

使用 `CreatFace.py` 收集人脸数据：

```bash
python CreatFace.py
```

修改脚本中的 `face_path` 变量指定人脸库路径。

## 🔧 配置说明

### 人脸库配置

- 人脸库使用 `.npz` 格式存储
- 每个身份对应一个特征向量数组
- 支持多张人脸图片训练

### 模型配置

- 人脸检测模型：`yolov5-blazeface.onnx`
- 人脸识别模型：`glintasia_r50.onnx`
- 输入尺寸：640x640 (检测) / 112x112 (识别)

## 📊 性能特点

- **多种推理方式**: 支持CPU+GPU混合推理和CPU推理
- **多线程**: 支持并发处理多个请求
- **实时性**: 优化的推理流程，支持实时视频流处理
- **可扩展**: 模块化设计，易于扩展新功能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。

## 👨‍💻 作者

[fete2306](https://github.com/fete2306)

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！
