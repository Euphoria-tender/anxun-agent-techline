@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================================
echo   安巡 Agent · 安装（CPU 版，任意电脑）
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [x] 没找到 python。请先装 Python 3.9 以上，安装时勾选 "Add to PATH"。
    echo     https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python -V 2^>^&1') do set PYVER=%%v
echo [1/3] Python %PYVER%

echo [2/3] 安装依赖（onnxruntime / opencv / numpy，约 80 MB）...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-cpu.txt
if errorlevel 1 (
    echo [x] 依赖安装失败。国内网络可加镜像：
    echo     python -m pip install -r requirements-cpu.txt -i https://mirrors.aliyun.com/pypi/simple/
    pause
    exit /b 1
)

echo [3/3] 检查模型权重...
if exist models\onnx\best.onnx (
    echo     已找到 models\onnx\ ，共 55 MB
) else (
    echo [x] 缺 models\onnx\best.onnx
    echo     从项目仓库里把 models\onnx\ 整个目录拷过来，
    echo     或者在装了 torch 的机器上运行：python src\anxun\export_onnx.py
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   装完了。双击 run.bat 启动，或命令行：
echo     python src\anxun\app.py
echo ============================================================
pause
