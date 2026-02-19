@echo off
chcp 65001 >nul 2>&1

echo ============================================================
echo    SYNAPSE - Installation Script
echo ============================================================
echo.

REM ============================================================
REM Step 1: Check Python
REM ============================================================
echo [1/5] Checking Python...

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

python --version
echo       OK
echo.

REM ============================================================
REM Step 2: Create virtual environment
REM ============================================================
echo [2/5] Creating virtual environment...

if exist "venv" (
    echo       venv already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create venv!
        pause
        exit /b 1
    )
    echo       Created venv
)
echo.

REM ============================================================
REM Step 3: Activate and upgrade pip
REM ============================================================
echo [3/5] Activating environment...

call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
echo       OK
echo.

REM ============================================================
REM Step 4: Install PyTorch
REM ============================================================
echo [4/5] Installing PyTorch...
echo.

set CUDA_MODE=%1
if "%CUDA_MODE%"=="" set CUDA_MODE=cpu

if "%CUDA_MODE%"=="cuda" goto install_cuda
if "%CUDA_MODE%"=="cuda121" goto install_cuda
goto install_cpu

:install_cuda
echo       Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
goto done_torch

:install_cpu
echo       Installing PyTorch CPU...
echo       (Use 'setup.bat cuda' for GPU)
pip install torch torchvision torchaudio
goto done_torch

:done_torch
echo       PyTorch installed
echo.

REM ============================================================
REM Step 5: Install dependencies
REM ============================================================
echo [5/5] Installing SYNAPSE dependencies...

cd files
pip install -r requirements.txt
cd ..

echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.

REM Verification
echo Verifying installation...
echo.

cd files
python quickstart.py
cd ..

echo.
echo ============================================================
echo    Done! Next steps:
echo.
echo    1. run_life.bat   - Start AI life
echo    2. test.bat       - Run all tests
echo ============================================================
echo.

pause
