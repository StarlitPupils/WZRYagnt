@echo off
chcp 65001 >nul
cd /d %~dp0..
echo ================================================
echo   王者荣耀 手动标注工具
echo   操作:
echo     m            切换 全屏 / 小地图(放大3倍)
echo     数字键 1-9,0 选类别
echo     鼠标左键拖拽 画框
echo     d            删最后一个框
echo     n / p        下一张 / 上一张
echo     s            保存
echo     ESC          退出
echo ================================================
echo.
echo 标注目录: temp\ann\s*.png
echo 输出: 每张图同目录 .txt (YOLO格式)
echo.
pause
venv\Scripts\python.exe -X utf8 scripts\annotate_manual.py
pause
