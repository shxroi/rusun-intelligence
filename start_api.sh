#!/bin/bash
# NusaWatt forecast API - self-contained, env-baked, long-lived server.
# Usage: ./start_api.sh
# Launched with setsid/nohup so the foreground server survives the launcher exiting.
set -u
cd /c/Users/rakhaa/ayam/Code/rusun-twin/rusun-intelligence || exit 1
export SUPABASE_DB_URL='postgresql://analytics_reader.placutyxqpnyliomtppf:readerpassword123@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'
echo "starting uvicorn -> 127.0.0.1:8000 (SUPABASE_DB_URL set)"
exec /c/Users/rakhaa/ayam/Code/rusun-twin/rusun-intelligence/.venv/Scripts/python.exe -m uvicorn app.forecast_api:app --host 127.0.0.1 --port 8000
