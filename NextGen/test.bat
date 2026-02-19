@echo off
REM ============================================================
REM SYNAPSE - Quick Test
REM ============================================================
REM
REM Quick verification that everything works:
REM - Tests all 36 unit tests
REM - Shows demo output
REM - Verifies installation
REM
REM ============================================================

echo.
echo ============================================================
echo    SYNAPSE - Quick Test
echo ============================================================
echo.

REM Check venv
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] Running quickstart verification...
echo.
cd files
python quickstart.py
cd ..

echo.
echo ============================================================
echo.

echo [2/3] Running 36 unit tests...
echo.
cd files
python -m pytest tests/ -v --tb=short
cd ..

echo.
echo ============================================================
echo.

echo [3/3] Running demo...
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
