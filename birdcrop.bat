REM -----------------------------------------------------------------------------
REM This batch file is intended for drag & drop use: drop an image folder or file onto it.
REM Adjust the PYTHON executable path and the run_birdcrop.py script path below for your system!
REM -----------------------------------------------------------------------------
@echo off
echo Got argument "%~1"
"%USERPROFILE%\Programme\WPy64-31160\python-3.11.6.amd64\python.exe" "%USERPROFILE%\Documents\Repositories\bird_crop\run_birdcrop.py" --conf=0.25 --model-size=large "%~1"
timeout /t 10 /nobreak > NUL
