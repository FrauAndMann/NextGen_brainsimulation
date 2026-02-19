@echo off
REM ============================================================
REM SYNAPSE - Installation Script
REM ============================================================
REM
REM This script:
REM 1. Creates virtual environment
REM 2. Installs PyTorch (CPU or CUDA)
REM 3. Installs all dependencies
REM 4. Runs verification
REM
REM Usage:
REM   setup.bat          - Install with CPU PyTorch
REM   setup.bat cuda     - Install with CUDA PyTorch
REM   setup.bat cuda121  - Install with CUDA 12.1
REM
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    SYNAPSE - Self-Aware AI Installation
echo ============================================================
echo.

REM Detect CUDA argument
set CUDA_VERSION=%1
if "%CUDA_VERSION%"=="" set CUDA_VERSION=none

REM Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.9+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Python %PYTHON_VERSION% found

REM Check if already in venv
if defined VIRTUAL_ENV (
    echo.
    echo WARNING: Virtual environment already active!
    echo   Current: %VIRTUAL_ENV%
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 0
)

REM Create venv
echo.
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo       venv already exists, skipping creation
) else (
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment!
        echo Make sure you have 'python -m venv' available.
        pause
        exit /b 1
    )
    echo       Created: venv\
)

REM Activate venv
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo       Activated successfully

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip -q
echo       Done

REM Install PyTorch
echo.
echo [5/6] Installing PyTorch...
echo.

if "%CUDA_VERSION%"=="cuda" (
    echo       Installing PyTorch with CUDA (latest)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%CUDA_VERSION%"=="cuda121" (
    echo       Installing PyTorch with CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%CUDA_VERSION%"=="cuda118" (
    echo       Installing PyTorch with CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo       Installing PyTorch (CPU version)...
    echo       Tip: Use 'setup.bat cuda' for GPU support
    pip install torch torchvision torchaudio
)

if errorlevel 1 (
    echo.
    echo WARNING: PyTorch installation had issues, continuing...
)

REM Install other dependencies
echo.
echo [6/6] Installing SYNAPSE dependencies...
cd files

REM Install from requirements.txt
pip install -r requirements.txt -q

if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies may have failed to install
)

cd ..

echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.

REM Verification
echo Running verification...
echo.

cd files
python quickstart.py
set VERIFY_RESULT=%errorlevel%
cd ..

echo.
echo ============================================================

if %VERIFY_RESULT%==0 (
    echo.
    echo    SUCCESS! SYNAPSE is ready to use.
    echo.
    echo    Next steps:
    echo.
    echo      1. Activate environment:
    echo         venv\Scripts\activate
    echo.
    echo      2. Run continuous training:
    echo         cd files
    echo         python train_continuous.py
    echo.
    echo      3. Or start with dashboard:
    echo         run_life.bat
    echo.
) else (
    echo.
    echo    WARNING: Verification had issues.
    echo    Check the errors above.
    echo.
)

echo ============================================================
echo.

pause
