@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  Camo Foundry Windows x64 build script
echo ============================================================
echo.

if /I not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if /I not "%PROCESSOR_ARCHITEW6432%"=="AMD64" (
    echo This build script needs 64-bit Windows.
    pause
    exit /b 1
)

set "PYTHON_URL=https://www.python.org/ftp/python/3.13.14/python-3.13.14-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python-3.13.14-amd64.exe"
set "PYTHON_EXE="

echo Searching for a 64-bit Python...
py -3.13-64 -c "import struct,sys; sys.exit(0 if struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
if %ERRORLEVEL%==0 set "PYTHON_EXE=py -3.13-64"

if not defined PYTHON_EXE (
    py -3-64 -c "import struct,sys; sys.exit(0 if struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
    if %ERRORLEVEL%==0 set "PYTHON_EXE=py -3-64"
)

if not defined PYTHON_EXE (
    python -c "import struct,sys; sys.exit(0 if struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
    if %ERRORLEVEL%==0 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo No usable 64-bit Python found. Downloading Python 3.13.14 x64 from python.org...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'" || goto :fail
    echo Installing Python silently for the current user...
    start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Include_test=0 || goto :fail
    py -3.13-64 -c "import struct,sys; sys.exit(0 if struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
    if %ERRORLEVEL%==0 set "PYTHON_EXE=py -3.13-64"
)

if not defined PYTHON_EXE (
    echo Python installation did not expose a usable 64-bit Python command.
    echo Try closing and reopening Command Prompt, then run build.bat again.
    goto :fail
)

echo Using: %PYTHON_EXE%
%PYTHON_EXE% -c "import struct; print('Python architecture:', struct.calcsize('P')*8, 'bit')" || goto :fail

echo.
echo Creating clean virtual environment...
if exist ".venv" rmdir /s /q ".venv"
%PYTHON_EXE% -m venv .venv || goto :fail

set "VENV_PY=.venv\Scripts\python.exe"
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel || goto :fail
"%VENV_PY%" -m pip install -r requirements.txt || goto :fail

echo.
echo Building standalone Windows app...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
"%VENV_PY%" -m PyInstaller --noconfirm --clean --windowed --onefile --name CamoFoundry --collect-all PySide6 camo_foundry.py || goto :fail

echo.
echo ============================================================
echo Build complete: dist\CamoFoundry.exe
echo ============================================================
pause
exit /b 0

:fail
echo.
echo Build failed. Scroll up for the actual error.
pause
exit /b 1
