#!/usr/bin/env bash
# cron:  30 19 * * 1-5  /path/to/quant_platform/scripts/daily_job.sh
cd "$(dirname "$0")/.." && source .venv/bin/activate && python run_pipeline.py daily >> data/artifacts/daily_job.log 2>&1
