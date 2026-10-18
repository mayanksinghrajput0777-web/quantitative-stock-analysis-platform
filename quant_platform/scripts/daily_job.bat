@echo off
REM Windows Task Scheduler: run Mon-Fri ~19:30 IST (the FII/DII file is published ~18:30-19:00)
cd /d %~dp0\..
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% start.py daily >> data\artifacts\daily_job.log 2>&1
