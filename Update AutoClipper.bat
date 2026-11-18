@echo off
rem Double-click to get the newest version of AutoClipper and its packages.
cd /d "%~dp0"
set "PATH=%PATH%;C:\Tools"
echo 1/3 Getting the newest version...
git pull
if errorlevel 1 echo    Couldn't update the code (no internet, or files you changed by hand). Your setup still works.
echo 2/3 Updating Python packages...
python -m pip install -q -r requirements.txt
if not exist ".venv-demucs\Scripts\python.exe" (
  echo    Installing music removal (one time, a few minutes)...
  python -m venv .venv-demucs
  ".venv-demucs\Scripts\python.exe" -m pip install -q demucs soundfile
)
echo 3/3 Checking everything...
python -m clipper.doctor
echo.
pause