@echo off
setlocal
cd /d "%~dp0"
echo.
echo  HAIKU ZERO - setup
echo.

rem ---- pick a Python launcher -------------------------------------------------
set "PYLAUNCH="
py -3.11 --version >nul 2>nul && set "PYLAUNCH=py -3.11"
if not defined PYLAUNCH py -3 --version >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH python --version >nul 2>nul && set "PYLAUNCH=python"
if not defined PYLAUNCH (
    echo  [X] No Python found. Install Python 3.11 from python.org and tick "Add to PATH".
    pause
    exit /b 1
)
if not exist pytorch_with_gpu\Scripts\python.exe (
    echo  [1/3] creating the virtual environment in pytorch_with_gpu
    %PYLAUNCH% -m venv pytorch_with_gpu
    if errorlevel 1 goto :venvfail
) else (
    echo  [1/3] virtual environment already exists
)

set "PY=pytorch_with_gpu\Scripts\python.exe"

rem ---- dependencies ----------------------------------------------------------
echo  [2/3] installing PyTorch (CUDA 13.0 build) and the rest
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :pipfail
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pipfail

"%PY%" -c "import torch; print('      torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

rem ---- front end -------------------------------------------------------------
if not exist frontend\package.json (
    echo  [3/3] no React front end here (that one is private).
    echo        The basic desk sheet ships with this repo - run start_desk.bat later.
    goto :done
)
if exist frontend\dist\index.html (
    echo  [3/3] front end already built
    goto :done
)
where npm >nul 2>nul
if errorlevel 1 (
    echo  [3/3] npm not found - the API will work, the web UI will not.
    echo        Install Node.js 20+ and run: cd frontend ^&^& npm install ^&^& npm run build
    goto :done
)
echo  [3/3] building the front end
pushd frontend
call npm install
if errorlevel 1 goto :npmfail
call npm run build
if errorlevel 1 goto :npmfail
popd

:done
echo.
echo  Ready. Double-click run.bat to start the app.
echo.
pause
exit /b 0

:venvfail
echo  [X] Could not create the virtual environment.
pause
exit /b 1

:pipfail
echo  [X] Dependency install failed. If it is a CUDA mismatch, install the CPU build instead:
echo      "%PY%" -m pip install torch
pause
exit /b 1

:npmfail
echo  [X] Front-end build failed. The API still works; fix the front end later.
popd
pause
exit /b 1
