@echo off
REM NusaWatt forecast API server - fully detached, long-lived.
REM Usage: start_api.cmd
REM Launches uvicorn detached from this terminal so it survives.
setlocal

set "WORK_DIR=C:\Users\rakhaa\ayam\Code\rusun-twin\rusun-intelligence"
cd /d "%WORK_DIR%"

set "SUPABASE_DB_URL=postgresql://analytics_reader.placutyxqpnyliomtppf:readerpassword123@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

echo Launching uvicorn with SUPABASE_DB_URL=...pooler.supabase.com:6543/postgres

REM Fully detach: create a new session, redirect all I/O, no console window.
start "NusaWatt Forecast API" /B /I .\venv\Scripts\python.exe -m uvicorn app.forecast_api:app --host 127.0.0.1 --port 8000

echo [done] uvicorn launcher issued (PID may vary).
