@echo off
chcp 65001 >nul
cd /d %~dp0
echo 安巡 Agent  正在启动...
echo   有显卡 -> 自动用 GPU；没有 -> ONNX CPU 版
echo   强制 CPU：run.bat --backend onnx
echo.
python src\anxun\app.py %*
pause
