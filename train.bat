@echo off
setlocal
cd /d "%~dp0"

set "PY=pytorch_with_gpu\Scripts\python.exe"
set "EPOCHS=%~1"
if "%EPOCHS%"=="" set "EPOCHS=5"

if not exist "%PY%" (
    echo  No environment yet - running setup first.
    call setup.bat
)
if not exist "%PY%" exit /b 1

if not exist data\poetry_pretrain_v2.txt (
    echo.
    echo  [X] Missing data\poetry_pretrain_v2.txt
    echo      Build it with:  "%PY%" scripts\data\fetch_poetry.py --out data\poetry_pretrain_v2.txt --max-chars 150e6 --max-books 3000
    echo.
    pause
    exit /b 1
)
if not exist data\haikus_classical.txt (
    echo.
    echo  [X] Missing data\haikus_classical.txt
    echo      Build the haiku corpora first: "%PY%" -m backend.prepare_data --raw-dir data\raw --out data\haikus_clean.txt
    echo.
    pause
    exit /b 1
)

echo.
echo  Stage 1 - English, %EPOCHS% epochs on 54M tokens of public-domain poetry
echo  (60 minutes at the default 5 epochs on an RTX 4050)
echo.
"%PY%" -m backend.train --corpus data\poetry_pretrain_v2.txt --epochs %EPOCHS% ^
    --run-name poetry-pretrain-v2 --ckpt-dir checkpoints_pretrain_v2 ^
    --log logs\training\training_log_pretrain_v2.csv --embed-dim 384 --layers 8 --ffn-dim 1536
if errorlevel 1 goto :fail

echo.
echo  Stage 2 - form, 25 epochs on the authentic haiku corpus (about 5 minutes)
echo.
"%PY%" -m backend.train --corpus data\haikus_classical.txt --epochs 25 --lr 1e-4 ^
    --init-from checkpoints_pretrain_v2\best.pt --run-name poetry-then-haiku-v2 ^
    --ckpt-dir checkpoints --log logs\training\training_log_poetry_v2.csv ^
    --embed-dim 384 --layers 8 --ffn-dim 1536
if errorlevel 1 goto :fail

echo.
echo  Done. checkpoints\best.pt is the new model.
echo  Check it:  "%PY%" scripts\eval\quality_floor.py
echo.
pause
exit /b 0

:fail
echo.
echo  [X] Training stopped on an error.
pause
exit /b 1
