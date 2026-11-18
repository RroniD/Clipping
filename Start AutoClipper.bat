@echo off
rem Double-click to start AutoClipper: the review queue + live mode for every streamer on your list.
cd /d "%~dp0"
set "PATH=%PATH%;C:\Tools"
set PYTHONIOENCODING=utf-8
echo Starting AutoClipper...
start "AutoClipper - review queue (keep open)" cmd /k python -m clipper.queue_app
timeout /t 4 /nobreak >nul
start "AutoClipper - live mode (keep open)" cmd /k python -m clipper.live
start "" http://127.0.0.1:8000
echo.
echo Two windows opened: the review queue and live mode. Keep them open.
echo The queue opened in your browser. On your phone, use the address shown on the queue page.
echo To stop: close both windows.
timeout /t 10 >nul