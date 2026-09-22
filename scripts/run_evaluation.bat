@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0learning"
if "%*"=="" (
    set FILES=
    for %%f in (data\checkpoints\ensemble_5\*.pt) do set FILES=!FILES! "%%f"
    python evaluate_ensemble.py !FILES!
) else (
    python evaluate_ensemble.py %*
)
