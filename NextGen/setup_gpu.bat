@echo off
chcp 65001 >nul 2>&1
cls
echo ============================================================
echo    SYNAPSE - GPU Setup (Automatic)
echo ============================================================
echo.

REM Step 1: Remove old venv
echo [1/6] Removing old venv...
if exist venv (
    rd /s /q venv
    echo       Removed!
) else (
    echo       No old venv found.
)

REM Step 2: Create new venv with Python 3.10
echo.
echo [2/6] Creating venv with Python 3.10...
if not exist "C:\Python310\python.exe" (
    echo ERROR: Python 3.10 not found!
    echo.
    echo Please install Python 3.10:
    echo https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    pause
    exit /b 1
)

C:\Python310\python.exe -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv!
    pause
    exit /b 1
)
echo       Created!

REM Step 3: Activate
echo.
echo [3/6] Activating venv...
call venv\Scripts\activate.bat
echo       Activated!

REM Step 4: Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Step 5: Install PyTorch with CUDA
echo.
echo [5/6] Installing PyTorch with CUDA 12.1...
echo       This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo CUDA 12.1 failed. Trying CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Step 6: Install other dependencies
echo.
echo [6/6] Installing other dependencies...
pip install transformers sentence-transformers chromadb numpy scipy matplotlib seaborn networkx opencv-python pillow tqdm pytest fastapi uvicorn websockets --quiet

echo.
echo ============================================================
echo    Verifying GPU Support
echo ============================================================
echo.

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED - will use CPU')"

echo.
echo ============================================================
if errorlevel 0 (
    echo    Setup Complete!
    echo.
    echo    Now run: run_life.bat
) else (
    echo    Setup may have issues. Check errors above.
)
echo ============================================================
echo.
pause
