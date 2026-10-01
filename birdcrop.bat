REM -----------------------------------------------------------------------------
REM This batch file is intended for drag & drop use: drop an image folder or file onto it.
REM Adjust the PYTHON executable path and the run_birdcrop.py script path below for your system!
REM -----------------------------------------------------------------------------
@echo off
echo Got argument "%~1"
"%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" "%USERPROFILE%\Documents\Repositories\bird_crop\run_birdcrop.py" --confidence=0.33 --model-size=large -- "%~1"
timeout /t 10 /nobreak > NUL
