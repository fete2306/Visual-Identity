# Visual-Identity_r

本地人脸检测 / 手势识别 / 人脸识别接口服务。端侧做检测，后端做识别；端侧与后端局域网连通即可，断外网可用。

## 架构

- 端侧：人脸框检测、手部关键点检测，在调用方本地跑，不走网络。
- 后端（`app.py`）：只做人脸识别（特征提取 + 比对），通过 HTTP 接口对外提供。
- 前端与示例应用只调用上述两层，不自带识别逻辑。

## 核心接口

`Back.py`（本地调用）：

1. 人脸检测：SCRFD，`detect_face` 返回人脸框（默认 `Model/SCRFD/det_500m.onnx`，可换 `det_2.5g.onnx`）。
2. 手势识别：MediaPipe HandLandmarker，`detect_hands` 返回手势字典（`Model/hand_landmarker.task`，21 关键点）。
3. 人脸识别：ArcFace R50，`FaceRecognition` 负责特征提取与比对（`Model/buffalo_l/w600k_r50.onnx`，512 维余弦相似度，`min_sim` 默认 0.4）。

`app.py`（服务调用）：把 `FaceRecognition` 封装为 HTTP 接口（见下文“接口”一节），另挂载 `/front` 实现前端同源访问、`/readme` 托管本文档。

## 应用层

外层功能，与核心接口隔离：

- `Front/front.html`、`Front/front.js`：浏览器端。本地 BlazeFace 检测 + HandLandmarker 手势，识别调用同源 `/infer/batch`。
- `Front/front_py.py`：桌面端。本地检测 + 本地手势，识别调用远端 `/infer/batch`；支持列摄像头、单图、`--remote-only` 整帧直送。
- `dev/demo_hand_mouse.py`：手势鼠标示例。
- `dev/demo_attendance.py`、`dev/attendance_db.py`：人脸签到示例（sqlite 存取封装在 `attendance_db.py`）。
- `setup.bat`（环境配置，跑一次）、`start.bat`（运行入口）。

## 目录结构

目标结构：只收录未被 `.gitignore` / `.git/info/exclude` 屏蔽的文件（即仓库应有内容）。

```
Visual-Identity_r/
├── README.md               # 本文档（/readme 同域托管）
├── requirements.txt        # 依赖
├── setup.bat               # 环境配置（选 Python → 装依赖 → 补模型 → 签证书，跑一次）
├── start.bat               # 运行入口（HTTPS 部署 / 本地流程 / demo）
├── .gitignore              # 仓库级忽略
├── app.py                  # 后端（/infer/* + /front 同源挂载 + /readme）
├── Back.py                 # 解耦后端（detect_face / detect_hands / FaceRecognition）
├── Front/
│   ├── front.html          # 浏览器端页面
│   ├── front.js            # 浏览器端逻辑（含 Node 单图 CLI）
│   └── front_py.py         # 桌面端（本地检测 + 远端识别）
├── dev/
│   ├── attendance_db.py    # 签到示例的 sqlite 存取封装
│   ├── demo_attendance.py  # 人脸签到示例
│   └── demo_hand_mouse.py  # 手势鼠标示例
└── tools/
    ├── download_models.py  # 模型并行补全（大小 + SHA256 双校验）
    ├── gen_cert.py         # 自签证书签发
    └── get_lan_ip.py       # 取本机局域网 IP（setup.bat 签证书 SAN / start.bat 回显用）
```

以下目录/文件在运行时生成或由 `setup.bat` 补全，不随仓库提交，不在上表列出：

- `Model/`（`.gitignore` 忽略，`setup.bat` 自动下载）
- `Data/FaceEmbedding/`（`.gitignore` 忽略，人脸底库，磁盘唯一份，注意备份）
- `Front/vendor/`（`.gitignore` 忽略，`setup.bat` 按 tasks-vision 钉死版本自动下载）
- `certs/key.pem`、`certs/cert.pem`（缺失时 `setup.bat` 自动签发）
- `dev/attendance.db`（`.gitignore` 忽略，demo 运行时生成）
- `.virun`（`.gitignore` 忽略，`setup.bat` 写入的 Python 路径配置）
- `__pycache__/`、`.vscode/`（`.gitignore` 忽略）

本地文件（`.git/info/exclude`，仅本机存在，不进仓库）：`Https/`、`base.py`、`base1.py`（同名命中含 `dev/base1.py`）。

## 快速启动

先双击 `setup.bat`（可 `setup.bat <conda环境名>` 跳过输入，首次/换环境时跑一次）：

1. 输入 conda 环境名（回车 = 全局 Python；都没有则自动装 Python 3.10），配置存 `.virun`；
2. 依赖齐备则跳过 pip，否则前台安装；缺失模型后台多线程并行补全（大小 + SHA256 双校验），等待时实时回显进度；
3. 缺证书自动签发（`tools/gen_cert.py` 优先，openssl 备用；依赖来自 `requirements.txt`，此处不装包；SAN 含 localhost / 127.0.0.1 / 本机 IP）。

注意：跑 `setup.bat` 前先关闭 VPN / 虚拟网卡（如 FlClash 等 TUN 类）。挂着它们时取到的是虚拟网卡 IP（如 `198.18.0.1`），会写进证书 SAN，局域网其他设备就连不上本机了。

再双击 `start.bat` 选模式：`1` HTTPS 部署 / `2` 本地流程 / `3` 手势 demo / `4` 签到 demo。

HTTPS 浏览器首次打开点“继续前往”信任自签证书：
`https://<本机IP>:8060/front/front.html`，本文档同域：`https://<本机IP>:8060/readme`。

## 手动启动

```bat
:: HTTPS（前端全站 HTTPS）
python -m uvicorn app:app --host 0.0.0.0 --port 8060 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
:: 桌面前端：列摄像头 / 单图测试 / 摄像头实时 / 纯远端
python Front\front_py.py --list-cameras
python Front\front_py.py --image Data\test.jpg --server https://127.0.0.1:8060
python Front\front_py.py --server https://127.0.0.1:8060 --camera 0
python Front\front_py.py --server https://127.0.0.1:8060 --camera 0 --remote-only
:: dev demo
python dev\demo_hand_mouse.py    :: q退出 m开关 f切换左右手
python dev\demo_attendance.py    :: q退出
```

## 接口

- `POST /infer/batch?min_sim=0.4`：multipart，字段名为序号的人脸 crop（`"0".."k-1"`，最多 20）→ `{"time": float, "results": {"0": [name, sim], ...}}`。未注册返回 `stranger`；单张坏图记 `unknow`，不影响整批。
- `POST /infer/single`：单图（`img` 文件字段，可带 `min_sim`）。
- `POST /infer/save`：`img` + `name`，录入人脸。
- `GET /front/front.html`：浏览器前端（须 HTTPS 同源）。`GET /docs`：接口调试。`GET /readme`：本文档（有 markdown 库渲染，否则原样 `<pre>` 显示）。

## 模型与数据

- `Model/` 被 git 忽略，新机 `setup.bat` 自动并行下载：`det_500m.onnx` / `det_2.5g.onnx` / `w600k_r50.onnx`（约 166MB）/ `hand_landmarker.task`；`Front/vendor` 大文件（wasm 钉死 tasks-vision 版本、hand task）同样自动下载。手动检查：`python tools/download_models.py --root . --check-only`。
- 无 GPU 时 onnxruntime 自动回退 CPU。
- `Data/FaceEmbedding` 被 git 忽略且是唯一份，备份后再改动。
- `Front/vendor/` 整体被 git 忽略（mjs 与 wasm 同版本配套下载）；缺 wasm 则浏览器端检测/手势不可用。

## 证书

`certs/key.pem` + `cert.pem` 为自签证书（HTTPS 必备）。缺失由 `setup.bat` 自动签发（`tools/gen_cert.py` 优先，openssl 备用；SAN 含 localhost / 127.0.0.1 / 签发时本机 IP），也可手动放置或跑 `python tools/gen_cert.py --ip <局域网IP>`。`front_py.py` 默认校验 `certs/cert.pem`，对不上加 `--no-verify` 跳过。

## 常见问题

- 摄像头打不开：换 `--camera` 索引，先 `--list-cameras` 查看编号。
- 模型缺失/截断：重跑 `setup.bat`，或手动跑 `tools/download_models.py`。
- 识别率低：1 米内正脸、光线充足；`min_sim` 默认 0.4 可调。
- 断外网可用（端侧与后端局域网连通即可）。
- 浏览器取摄像头要求安全上下文（localhost 或 HTTPS）：局域网使用 HTTPS 部署；纯本地无网场景使用 `start.bat` 本地流程。
- demo 提示导包失败：以项目根为工作目录运行（`python dev\demo_*.py`）。
- 局域网打不开页面 / 证书 IP 不对：跑 `setup.bat` 时挂着 VPN，证书签到了虚拟网卡 IP。解决：关 VPN 后删掉 `certs\key.pem` + `certs\cert.pem`，重跑 `setup.bat`；或不断网直接 `python tools/gen_cert.py --ip <真实局域网IP>` 重签。
