@echo off
setlocal
set "JSS_PYTHON=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "JSS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
"%JSS_PYTHON%" -u "%~dp0run_all_types.py" %*