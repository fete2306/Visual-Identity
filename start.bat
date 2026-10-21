@echo off
chcp 936 >nul
setlocal EnableExtensions
rem ================================================================
rem  start.bat - 纯功能入口 (配置见 setup.bat, Python 路径读 .virun)
rem  1) HTTPS - 局域网内部署 (前端全站 HTTPS)
rem  2) 本地 - Base流程 (摄像头直采识别, 无网络传输)
rem  3/4) dev demo (手势鼠标 / 人脸签到)
rem ================================================================
cd /d "%~dp0"
set "ROOT=%CD%"
set "PYTHON="
if exist "%ROOT%\.virun" for /f "usebackq delims=" %%v in ("%ROOT%\.virun") do set "PYTHON=%%v"
if not defined PYTHON goto :no_setup
if not exist "%PYTHON%" goto :no_setup
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 goto :no_setup
goto :menu

:no_setup
echo [错误] 未找到可用 Python 配置, 请先运行 setup.bat 完成配置。
pause
exit /b 1

:menu
echo [start] 使用 Python: %PYTHON%
echo   1) HTTPS - 局域网内部署 (证书缺失请先跑 setup.bat 签发, 推荐)
echo   2) 本地 - Base流程 (摄像头直采识别, 无网络传输)
echo   3) DEMO  - 手势鼠标 (右手食指相对位移, q退出)
echo   4) DEMO  - 人脸签到 (sqlite 落库, q退出)
choice /c 1234 /n /m "请输入 1-4: "
if errorlevel 4 goto :demo_att
if errorlevel 3 goto :demo_mouse
if errorlevel 2 goto :local
goto :https

:https
if not exist "%ROOT%\certs\key.pem" goto :no_cert
if not exist "%ROOT%\certs\cert.pem" goto :no_cert
call :get_ip
echo 正在启动 HTTPS 服务, 浏览器访问: https://%LOCAL_IP%:8060/front/front.html
echo 接口文档: https://%LOCAL_IP%:8060/docs (先信任自签证书)
"%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8060 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
goto :end

:no_cert
echo [错误] 缺少 certs\key.pem / cert.pem, 请先运行 setup.bat 自动签发。
pause
exit /b 1

:local
if not exist "%ROOT%\Back.py" (
  echo [提示] 本地流程需要 back.py ^(本地文件, 不在仓库中^)。
  pause
  exit /b 1
)
echo 启动本地 Base流程 (摄像头直采+识别, 无网络传输; q退出, s录入人脸) ...
"%PYTHON%" "%ROOT%\Back.py"
goto :end

:demo_mouse
echo 启动手势鼠标 demo (q退出, m开关鼠标, f切换左右手) ...
"%PYTHON%" "%ROOT%\dev\demo_hand_mouse.py"
goto :end

:demo_att
echo 启动签到 demo (q退出, 记录写入 dev\attendance.db) ...
"%PYTHON%" "%ROOT%\dev\demo_attendance.py"
goto :end

:end
pause
exit /b 0

:get_ip
set "LOCAL_IP="
if exist "%ROOT%/tools/get_lan_ip.py" for /f "delims=" %%a in ('"%PYTHON%" "%ROOT%/tools/get_lan_ip.py" 2^>nul') do (
  if not defined LOCAL_IP set "LOCAL_IP=%%a"
)
rem 脚本回环表示无可用内网(或被 Clash TUN 等虚拟网卡顶掉默认出口), 再用默认路由网卡兜底一次;
rem 不取 ipconfig 首个 IPv4: VMware VMnet1/8 常排在前面, 会拿到 192.168.x.1 这类虚拟地址。
if "%LOCAL_IP%"=="127.0.0.1" set "LOCAL_IP="
if not defined LOCAL_IP for /f "tokens=4" %%a in ('route print -4 2^>nul ^| findstr /r /c:"^ *0\.0\.0\.0 *0\.0\.0\.0"') do (
  if not defined LOCAL_IP set "LOCAL_IP=%%a"
)
set "LOCAL_IP=%LOCAL_IP: =%"
if not defined LOCAL_IP set "LOCAL_IP=127.0.0.1"
exit /b 0
