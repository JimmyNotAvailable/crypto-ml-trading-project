@echo off
:: =====================================
:: CRYPTO ML PROJECT - SIMPLE LAUNCHER
:: =====================================
chcp 65001 >nul

echo.
echo CRYPTO ML TRADING PROJECT
echo ======================================
echo.

:: Chuyển về thư mục dự án
cd /d "e:\Code on PC\DoAnMLPython\crypto-project-clean"

echo [1] Kich hoat moi truong ao
echo [2] Cai dat packages
echo [3] Chay bot Discord
echo [4] Chay web dashboard (neu da cai plotly)
echo [5] Mo Python interactive
echo [6] Thoat
echo.
set /p choice=Chon (1-6): 

if "%choice%"=="1" goto ACTIVATE_ENV
if "%choice%"=="2" goto INSTALL_PACKAGES  
if "%choice%"=="3" goto RUN_BOT
if "%choice%"=="4" goto RUN_DASHBOARD
if "%choice%"=="5" goto PYTHON_SHELL
if "%choice%"=="6" goto EXIT

echo Lua chon khong hop le!
pause
goto START

:ACTIVATE_ENV
echo.
echo Kich hoat moi truong ao...
powershell -Command "& '.\crypto-venv\Scripts\Activate.ps1'"
if errorlevel 1 (
    echo Loi kich hoat! Hay tao moi truong ao truoc:
    echo python -m venv crypto-venv
    pause
    goto START
)
echo Moi truong ao da duoc kich hoat!
echo Ban co the chay cac lenh Python bay gio.
cmd /k
goto EXIT

:INSTALL_PACKAGES
echo.
echo Cai dat tat ca packages...
crypto-venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo Loi cai dat packages!
) else (
    echo Da cai dat thanh cong!
)
pause
goto START

:RUN_BOT
echo.
echo Chay Discord Bot...
crypto-venv\Scripts\python.exe app\bot.py
pause
goto START

:RUN_DASHBOARD
echo.
echo Chay Web Dashboard...
echo Truy cap: http://localhost:5000
crypto-venv\Scripts\python.exe examples\ml\web_dashboard.py
pause
goto START

:PYTHON_SHELL
echo.
echo Mo Python interactive...
crypto-venv\Scripts\python.exe
pause
goto START

:START
cls
echo.
echo CRYPTO ML TRADING PROJECT
echo ======================================
echo.
echo [1] Kich hoat moi truong ao
echo [2] Cai dat packages
echo [3] Chay bot Discord
echo [4] Chay web dashboard (neu da cai plotly)
echo [5] Mo Python interactive
echo [6] Thoat
echo.
set /p choice=Chon (1-6): 

if "%choice%"=="1" goto ACTIVATE_ENV
if "%choice%"=="2" goto INSTALL_PACKAGES  
if "%choice%"=="3" goto RUN_BOT
if "%choice%"=="4" goto RUN_DASHBOARD
if "%choice%"=="5" goto PYTHON_SHELL
if "%choice%"=="6" goto EXIT

echo Lua chon khong hop le!
pause
goto START

:EXIT
echo Tam biet!
exit