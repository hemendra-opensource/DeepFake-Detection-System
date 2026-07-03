@echo off
REM ============================================================
REM  DeepFake Detection System — Quick Start Script (Windows)
REM  Usage: run.bat [dashboard|train|test|preprocess]
REM ============================================================

SET PROJECT_DIR=%~dp0
SET VENV_PYTHON=%PROJECT_DIR%venv\Scripts\python.exe
SET VENV_STREAMLIT=%PROJECT_DIR%venv\Scripts\streamlit.exe

IF NOT EXIST "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found.
    echo Run this first: py -3.11 -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

SET CMD=%1
IF "%CMD%"=="" SET CMD=dashboard

IF "%CMD%"=="dashboard" (
    echo [INFO] Starting Streamlit dashboard...
    "%VENV_STREAMLIT%" run app.py --server.port 8501 --server.headless true
    goto :end
)

IF "%CMD%"=="train" (
    echo [INFO] Starting training (model: %2, phases: %3)...
    "%VENV_PYTHON%" train.py --model %2 --phases %3
    goto :end
)

IF "%CMD%"=="test" (
    echo [INFO] Running all unit tests...
    "%VENV_PYTHON%" -m pytest tests/ -v
    goto :end
)

IF "%CMD%"=="preprocess" (
    echo [INFO] Running preprocessing pipeline...
    "%VENV_PYTHON%" train.py --preprocess --datasets %2
    goto :end
)

IF "%CMD%"=="evaluate" (
    echo [INFO] Running model evaluation...
    "%VENV_PYTHON%" train.py --evaluate
    goto :end
)

echo Usage: run.bat [dashboard^|train^|test^|preprocess^|evaluate]
echo.
echo  dashboard   - Launch the Streamlit dashboard (default)
echo  train       - Train models: run.bat train xceptionnet all
echo  test        - Run all unit tests
echo  preprocess  - Preprocess datasets: run.bat preprocess celeb_df
echo  evaluate    - Evaluate all trained models

:end
