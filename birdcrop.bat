@echo off
echo Got argument "%~1"
"%USERPROFILE%\Programme\WPy64-31160\python-3.11.6.amd64\python.exe" "%USERPROFILE%\Documents\Repositories\bird_crop\run_birdcrop.py" "%~1"
timeout /t 10 /nobreak > NUL
