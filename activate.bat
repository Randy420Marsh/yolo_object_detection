@echo off

SET UV_LINK_MODE=copy

SET current_path=%CD%

cd %current_path%

setlocal enabledelayedexpansion

IF exist ./venv (cmd /k call .\venv\scripts\activate.bat)  ELSE (cmd /k uv venv venv --python 3.12 && cmd /k call .\venv\scripts\activate.bat)