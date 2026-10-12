@echo off
chcp 65001 >nul
setlocal EnableExtensions
rem ================================================================
rem  Visual-Identity 一键启动
rem  流程: 选Python(conda手动输入->全局->安装3.10)
rem        ->装依赖->并行补全模型->选HTTP/HTTPS启动
rem  用法: start.bat [conda环境名]
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
echo [1/5] 使用 Python: %PYTHON%
"%PYTHON%" --version
if errorlevel 1 (
  echo [错误] 该 Python 无法运行。
  pause
  exit /b 1
)
echo.

rem ============ 2+3. 模型后台并行下载 + 依赖前台安装(两者重叠执行) ============
rem 原理: 两步无依赖。模型下载内部已是多线程, 再用 start /b 丢后台(即 bash 的 `&`);
rem       后台命令是单条干净命令, 日志/退出码由 Python 自己落盘, 不拼接 & > 符号;
rem       前台跑 pip(输出可见、退出码直接可用), 最后等模型的 rc 文件。
echo [3/5] 模型后台并行下载中(日志: %TEMP%\vi_models.log) ...
del "%TEMP%\vi_models.flag" 2>nul
start "vi-models" /b "%PYTHON%" "%ROOT%\tools\download_models.py" --root "%ROOT%" --workers 4 --log "%TEMP%\vi_models.log" --rc-file "%TEMP%\vi_models.flag"
echo [2/5] 安装依赖 (requirements.txt) ...
set "PIP_RC=0"
"%PYTHON%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 set "PIP_RC=1"
set "WAIT_N=0"
:wait_models
if exist "%TEMP%\vi_models.flag" goto :models_done
ping -n 2 127.0.0.1 >nul
set /a WAIT_N+=1
if %WAIT_N% gtr 1800 (
  echo [错误] 模型下载等待超时, 请查看 %TEMP%\vi_models.log 后重跑 start.bat。
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
echo [2/5]+[3/5] 依赖与模型均就绪。
if not exist "%ROOT%\Data\FaceEmbedding" echo [警告] 缺少 Data\FaceEmbedding (人脸底库), 录入功能不可用。
if not exist "%ROOT%\certs\key.pem" echo [警告] 缺少 certs 证书, HTTPS 模式不可用。
if not exist "%ROOT%\certs\cert.pem" echo [警告] 缺少 certs 证书, HTTPS 模式不可用。
echo.

rem ============ 4. 选择启动模式 ============
echo [4/5] 选择启动模式:
echo   1) HTTPS - 局域网访问 front 页面 (需 certs, 推荐)
echo   2) HTTP  - http://本机IP:8000/docs
choice /c 12 /n /m "请输入 1 或 2: "
if errorlevel 2 goto :http
goto :https

:https
if not exist "%ROOT%\certs\key.pem" (
  echo [错误] 缺少 certs\key.pem / cert.pem, 无法以 HTTPS 启动。
  pause
  exit /b 1
)
call :get_ip
echo 正在启动 HTTPS 服务, 浏览器访问: https://%LOCAL_IP%:8060/front/front.html
"%PYTHON%" -m uvicorn Https.app:app --host 0.0.0.0 --port 8060 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
goto :end

:http
call :get_ip
echo 正在启动 HTTP 服务, 浏览器访问: http://%LOCAL_IP%:8000/docs
"%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000
goto :end

:end
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
conda run -n "%CONDA_ENV%" python --version >nul 2>nul
if errorlevel 1 (
  echo [提示] conda 环境 "%CONDA_ENV%" 不存在。可用环境如下:
  conda env list
  goto :conda_empty
)
for /f "delims=" %%p in ('conda run -n "%CONDA_ENV%" python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%p"
if defined PYTHON echo [conda] 已切换到环境 "%CONDA_ENV%": %PYTHON%
exit /b 0
:conda_empty
set "CONDA_ENV="
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

:get_ip
set "LOCAL_IP="
for /f "delims=" %%a in ('"%PYTHON%" "%ROOT%\testIp.py" 2^>nul') do (
  if not defined LOCAL_IP set "LOCAL_IP=%%a"
)
if not defined LOCAL_IP set "LOCAL_IP=127.0.0.1"
exit /b 0
