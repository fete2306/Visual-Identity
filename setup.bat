@echo off
chcp 65001 >nul
setlocal EnableExtensions
rem ================================================================
rem  setup.bat - 环境配置 (跑一次即可, 换环境时重跑)
rem  流程: 选Python(conda手动输入->全局->安装3.10) ->保存配置到.virun
rem        ->装依赖+模型后台并行补全 ->自签证书 ->检查
rem  用法: setup.bat [conda环境名]
rem  配置保存在 .virun (已加入 .gitignore), start.bat 只读它启动
rem ================================================================
cd /d "%~dp0"
set "ROOT=%CD%"
echo 项目目录: %ROOT%
echo.

set "CONDA_ENV="
if not "%~1"=="" set "CONDA_ENV=%~1"
set "PYTHON="

rem ============ 1. Python 定位 ============
if not defined CONDA_ENV set /p "CONDA_ENV=请输入 conda 环境名(直接回车则使用全局 Python): "
if defined CONDA_ENV call :use_conda
if not defined PYTHON call :use_global
if not defined PYTHON call :install_py310
if not defined PYTHON (
  echo [错误] 仍未找到可用 Python, 请手动安装 Python 3.10 后重试。
  pause
  exit /b 1
)
echo [1/4] 使用 Python: %PYTHON%
"%PYTHON%" --version
if errorlevel 1 (
  echo [错误] 该 Python 无法运行。
  pause
  exit /b 1
)
(echo %PYTHON%)>"%ROOT%\.virun"
echo [配置] Python 路径已保存到 .virun
echo.

rem ============ 2. 依赖 + 模型 (重叠执行) ============
rem 原理: 两步无依赖。模型下载内部已是多线程, 再用 start /b 丢后台(即 bash 的 `&`);
rem       后台命令是单条干净命令, 日志/退出码由 Python 自己落盘, 不拼接 & > 符号;
rem       前台跑 pip(输出可见、退出码直接可用), 最后等模型的 rc 文件。
echo [2/4] 模型后台并行下载中(大小+SHA256校验, 日志: %TEMP%\vi_models.log) ...
del "%TEMP%\vi_models.flag" 2>nul
start "vi-models" /b "%PYTHON%" "%ROOT%\tools\download_models.py" --root "%ROOT%" --workers 4 --log "%TEMP%\vi_models.log" --rc-file "%TEMP%\vi_models.flag"
echo [2/4] 安装依赖 (requirements.txt) ...
set "PIP_RC=0"
"%PYTHON%" -c "import fastapi,uvicorn,cv2,numpy,onnxruntime,mediapipe,requests,cryptography" >nul 2>nul
if not errorlevel 1 (
  echo [2/4] 依赖已齐备, 跳过 pip 安装。
  goto :pip_done
)
"%PYTHON%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 set "PIP_RC=1"
:pip_done
set "WAIT_N=0"
set "MLAST="
:wait_models
if exist "%TEMP%\vi_models.flag" goto :models_done
for /f "delims=" %%L in ('type "%TEMP%\vi_models.log" 2^>nul') do set "MCUR=%%L"
if defined MCUR if not "%MCUR%"=="%MLAST%" echo [模型] %MCUR%
set "MLAST=%MCUR%"
ping -n 2 127.0.0.1 >nul
set /a WAIT_N+=1
if %WAIT_N% gtr 1800 (
  echo [错误] 模型下载等待超时, 请查看 %TEMP%\vi_models.log 后重跑 setup.bat。
  pause
  exit /b 1
)
goto :wait_models
:models_done
set /p MODEL_RC=<"%TEMP%\vi_models.flag"
if not "%PIP_RC%"=="0" (
  echo [错误] 依赖安装失败, 请检查网络或 pip 换源后重试。
  pause
  exit /b 1
)
if not "%MODEL_RC%"=="0" (
  echo [错误] 模型补全失败, 日志: %TEMP%\vi_models.log
  type "%TEMP%\vi_models.log"
  pause
  exit /b 1
)
echo [2/4] 依赖与模型均就绪。
echo.

rem ============ 3. 自签证书 ============
echo [3/4] 检查自签证书 ...
call :ensure_certs
if errorlevel 1 (
  pause
  exit /b 1
)
echo.

rem ============ 4. 检查 ============
echo [4/4] 检查 ...
if not exist "%ROOT%\Front\vendor\tasks-vision\wasm\vision_wasm_internal.wasm" echo [警告] 缺少 Front\vendor wasm, 浏览器手势/检测不可用。
if not exist "%ROOT%\Front\vendor\hand_landmarker.task" echo [警告] 缺少 Front\vendor\hand_landmarker.task, 浏览器手势不可用。
if not exist "%ROOT%\Data\FaceEmbedding" echo [警告] 缺少 Data\FaceEmbedding (人脸底库), 录入功能不可用。
echo.
echo [完成] 配置就绪, 运行 start.bat 选择模式启动。
pause
exit /b 0

rem ================================================================
rem  子过程
rem ================================================================

:use_conda
set "CONDA_ENV=%CONDA_ENV: =%"
if "%CONDA_ENV%"=="" goto :conda_empty
where conda >nul 2>nul
if errorlevel 1 (
  echo [提示] 未找到 conda 命令, 跳过 conda 环境。
  goto :conda_empty
)
rem 直接定位该环境的 python.exe (不用 conda run, 输出唯一、无激活开销)
set "CONDA_BASE="
for /f "delims=" %%b in ('conda info --base 2^>nul') do if not defined CONDA_BASE set "CONDA_BASE=%%b"
if not defined CONDA_BASE (
  echo [提示] conda 不可用, 跳过 conda 环境。
  goto :conda_empty
)
if /i "%CONDA_ENV%"=="base" (
  set "PYTHON=%CONDA_BASE%\python.exe"
) else (
  set "PYTHON=%CONDA_BASE%\envs\%CONDA_ENV%\python.exe"
)
if not exist "%PYTHON%" (
  echo [提示] conda 环境 "%CONDA_ENV%" 不存在。可用环境如下:
  call conda env list
  goto :conda_empty
)
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 (
  echo [提示] 该 Python 无法运行: %PYTHON%
  goto :conda_empty
)
echo [conda] 已使用环境 "%CONDA_ENV%": %PYTHON%
exit /b 0
:conda_empty
set "CONDA_ENV="
set "PYTHON="
exit /b 0

:use_global
where python >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%p in ('where python 2^>nul') do (if not defined PYTHON set "PYTHON=%%p")
if not defined PYTHON exit /b 0
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 (
  set "PYTHON="
  exit /b 0
)
echo [global] 使用全局 Python: %PYTHON%
exit /b 0

:install_py310
echo [安装] 未找到可用 Python, 准备安装 Python 3.10 ...
where winget >nul 2>nul
if errorlevel 1 goto :no_winget
echo [安装] 使用 winget 安装 Python 3.10 ...
winget install -e --id Python.Python.3.10 --accept-source-agreements --accept-package-agreements
where python >nul 2>nul
if errorlevel 1 goto :no_winget
for /f "delims=" %%p in ('where python 2^>nul') do (if not defined PYTHON set "PYTHON=%%p")
if defined PYTHON (
  echo [安装] winget 安装成功: %PYTHON%
  exit /b 0
)
:no_winget
echo [安装] 改用官网安装包 (python-3.10.11-amd64.exe) ...
where curl >nul 2>nul
if errorlevel 1 (
  echo [错误] 无 winget 也无 curl, 请手动安装 Python 3.10 后重试。
  exit /b 0
)
set "PYSETUP=%TEMP%\python-3.10.11-amd64.exe"
curl -L -o "%PYSETUP%" "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
if errorlevel 1 (
  echo [错误] 安装包下载失败, 请手动安装 Python 3.10 后重试。
  exit /b 0
)
echo [安装] 静默安装中, 请稍候 ...
"%PYSETUP%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON%" (
  echo [错误] 安装后仍找不到 python.exe, 请手动安装后重试。
  set "PYTHON="
  exit /b 0
)
echo [安装] 安装成功: %PYTHON%
exit /b 0

:ensure_certs
if exist "%ROOT%\certs\key.pem" if exist "%ROOT%\certs\cert.pem" exit /b 0
echo [证书] 未找到自签证书, 正在自动签发 ...
if not exist "%ROOT%\certs" mkdir "%ROOT%\certs"
call :get_ip
rem ---- 首选 Python 方式 (tools/gen_cert.py, 不依赖 openssl, 最稳; 依赖见 requirements.txt, 此处不装包) ----
"%PYTHON%" -c "import cryptography" >nul 2>nul
if errorlevel 1 (
  echo [提示] 缺少 cryptography (requirements.txt 未装全), 改用 openssl 方式 ...
  goto :cert_openssl
)
"%PYTHON%" "%ROOT%\tools\gen_cert.py" --out-dir "%ROOT%\certs" --ip "%LOCAL_IP%"
if not errorlevel 1 if exist "%ROOT%\certs\key.pem" if exist "%ROOT%\certs\cert.pem" (
  echo [证书] 已签发 (Python, SAN 含 %LOCAL_IP%).
  exit /b 0
)
echo [提示] Python 签发失败, 改用 openssl 方式 ...
:cert_openssl
rem ---- 备用 openssl 方式 (conda 自带 openssl 缺默认 cnf, 置空 OPENSSL_CONF 绕过; 已验证可用) ----
where openssl >nul 2>nul
if errorlevel 1 (
  echo [错误] 两种签发方式均不可用, 请手动放置 certs\key.pem + cert.pem 后重试。
  exit /b 1
)
set "OPENSSL_CONF=NUL"
openssl req -x509 -newkey rsa:2048 -keyout "%ROOT%\certs\key.pem" -out "%ROOT%\certs\cert.pem" -days 825 -nodes -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:%LOCAL_IP%"
if not errorlevel 1 if exist "%ROOT%\certs\key.pem" if exist "%ROOT%\certs\cert.pem" (
  echo [证书] 已签发 (openssl, SAN 含 %LOCAL_IP%).
  exit /b 0
)
echo [错误] 证书签发失败, 请手动放置 certs\key.pem + cert.pem 后重试。
exit /b 1

:get_ip
set "LOCAL_IP="
if exist "%ROOT%\testIp.py" for /f "delims=" %%a in ('"%PYTHON%" "%ROOT%\testIp.py" 2^>nul') do (
  if not defined LOCAL_IP set "LOCAL_IP=%%a"
)
if not defined LOCAL_IP for /f "tokens=2 delims=:" %%a in ('ipconfig 2^>nul ^| findstr "IPv4"') do (
  if not defined LOCAL_IP set "LOCAL_IP=%%a"
)
set "LOCAL_IP=%LOCAL_IP: =%"
if not defined LOCAL_IP set "LOCAL_IP=127.0.0.1"
exit /b 0
