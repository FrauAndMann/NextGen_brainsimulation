@echo off
chcp 65001 >nul 2>&1

echo ============================================================
echo    SYNAPSE - Test
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] Quickstart check...
echo.
cd files
python quickstart.py
cd ..
echo.

echo ============================================================
echo [2/3] Running 36 tests...
echo.
cd files
python -m pytest tests/ -v
cd ..
echo.

echo ============================================================
echo [3/3] Demo...
echo.
cd files
python demo.py
cd ..

echo.
echo ============================================================
echo    All tests complete!
echo ============================================================
echo.

pause
