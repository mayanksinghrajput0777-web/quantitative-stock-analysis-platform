@echo off
cd /d %~dp0
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% start.py check
if errorlevel 1 goto end
%PY% start.py setup
:end
pause
