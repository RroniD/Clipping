@echo off
rem Double-click, paste a Twitch VOD link, and the finished clips appear in the review queue.
cd /d "%~dp0"
set "PATH=%PATH%;C:\Tools"
set PYTHONIOENCODING=utf-8
set /p URL="Paste the VOD link (for example https://www.twitch.tv/videos/123456789) and press Enter: "
if "%URL%"=="" goto :eof
python run.py "%URL%" --debug
if errorlevel 1 goto failed
for /f "delims=" %%F in ('python -c "import sys; from clipper.fetch import parse_vod_url; from clipper.config import load_streamers; print(parse_vod_url(sys.argv[1])[1])" "%URL%"') do set VOD=%%F
for /d %%D in (out\*\%VOD%) do python -m clipper.edit render "%%D"
echo.
echo Done. Open the review queue to approve the clips.
pause
goto :eof
:failed
echo.
echo Something went wrong (see the messages above). Run "Check setup.bat" for help.
pause