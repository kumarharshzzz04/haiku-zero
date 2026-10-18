@echo off
setlocal
cd /d "%~dp0"

set "PY=pytorch_with_gpu\Scripts\python.exe"
set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%/"
set "CURL=%SystemRoot%\System32\curl.exe"
set "PING=%SystemRoot%\System32\ping.exe"

if not exist "%PY%" (
    echo  No environment yet - running setup first.
    call setup.bat
)
if not exist "%PY%" exit /b 1

if not exist checkpoints\best.pt (
    echo.
    echo  [X] No model at checkpoints\best.pt
    echo      Train one with train.bat, or copy a checkpoint into checkpoints\.
    echo.
    pause
    exit /b 1
)

rem one server per port; if it is already up, just show it
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo  A server is already listening on port %PORT% - opening the browser to it.
    start "" "%URL%"
    exit /b 0
)

echo.
echo  Starting HAIKU ZERO on %URL%
start "HAIKU ZERO server" cmd /c ""%PY%" -m backend.server --port %PORT% %*"
echo  Waiting for it to answer...

set "TRIES=0"
:wait
"%CURL%" -s -f -o NUL --max-time 2 "%URL%api/health" >nul 2>nul
if not errorlevel 1 goto :up
set /a TRIES+=1
if %TRIES% geq 30 goto :slow
"%PING%" -n 2 127.0.0.1 >nul
goto :wait

:up
rem the server answers before the warm-up thread finishes loading the weights
"%PING%" -n 4 127.0.0.1 >nul
start "" "%URL%"
echo.
echo  Open in your browser. Close the "HAIKU ZERO server" window to stop the app.
exit /b 0

:slow
echo.
echo  The server is not answering after 30 s. Its window has the error - check it, then retry.
pause
exit /b 1
