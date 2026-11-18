@echo off
cd /d "%~dp0"
set "PATH=%PATH%;C:\Tools"
python -m clipper.doctor
echo.
pause