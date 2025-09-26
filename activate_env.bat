@echo off
chcp 65001 >nul
:: =====================================
:: CRYPTO ML PROJECT - ENVIRONMENT SETUP
:: =====================================
echo.
echo CRYPTO ML TRADING PROJECT
echo =====================================
echo.

:: Lưu thư mục hiện tại
set ORIGINAL_DIR=%CD%

:: Chuyển về thư mục dự án
cd /d "e:\Code on PC\DoAnMLPython\crypto-project-clean"

:: Kiểm tra môi trường ảo có tồn tại không
if not exist "crypto-venv\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy môi trường ảo crypto-venv
    echo 💡 Hãy tạo môi trường ảo trước:
    echo    python -m venv crypto-venv
    echo    crypto-venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

:: Kích hoạt môi trường ảo
echo Dang kich hoat moi truong ao...
call crypto-venv\Scripts\activate.bat

:: Kiểm tra kích hoạt thành công
if errorlevel 1 (
    echo Loi kich hoat moi truong ao
    pause
    exit /b 1
)

echo Da kich hoat moi truong ao crypto-venv
echo.

:: Hiển thị menu lựa chọn
::MENU
echo CHON HOAT DONG:
echo.
echo [1] Chay Web Dashboard (Flask)
echo [2] Chay Discord Bot
echo [3] Chay Data Collector
echo [4] Demo Machine Learning
echo [5] Cai dat/Cap nhat packages
echo [6] Chay Production Setup
echo [7] Mo Python Interactive
echo [8] Mo Command Prompt voi env
echo [9] Thoat
echo.
set /p choice=Nhap lua chon (1-9): 

if "%choice%"=="1" goto RUN_DASHBOARD
if "%choice%"=="2" goto RUN_BOT
if "%choice%"=="3" goto RUN_COLLECTOR
if "%choice%"=="4" goto RUN_ML_DEMO
if "%choice%"=="5" goto INSTALL_PACKAGES
if "%choice%"=="6" goto RUN_PRODUCTION
if "%choice%"=="7" goto PYTHON_INTERACTIVE
if "%choice%"=="8" goto OPEN_CMD
if "%choice%"=="9" goto EXIT

echo ❌ Lựa chọn không hợp lệ, vui lòng thử lại.
echo.
goto MENU

:RUN_DASHBOARD
echo.
echo 🌐 Đang khởi động Web Dashboard...
echo 📍 Truy cập: http://localhost:5000
echo.
python examples\ml\web_dashboard.py
goto MENU_RETURN

:RUN_BOT
echo.
echo 🤖 Đang khởi động Discord Bot...
echo.
python app\bot.py
goto MENU_RETURN

:RUN_COLLECTOR
echo.
echo 📊 Đang khởi động Data Collector...
echo.
python scripts\continuous_collector.py
goto MENU_RETURN

:RUN_ML_DEMO
echo.
echo 🧠 Chọn ML Demo:
echo [1] Demo Auto Training
echo [2] Demo New Models
echo [3] Demo Enterprise Services
echo [4] Demo Model Registry
echo.
set /p ml_choice=Nhập lựa chọn (1-4): 

if "%ml_choice%"=="1" (
    python demos\demo_auto_training.py
) else if "%ml_choice%"=="2" (
    python demos\demo_new_models.py
) else if "%ml_choice%"=="3" (
    python demos\demo_enterprise_services.py
) else if "%ml_choice%"=="4" (
    python demos\demo_model_registry.py
) else (
    echo ❌ Lựa chọn không hợp lệ
)
goto MENU_RETURN

:INSTALL_PACKAGES
echo.
echo 🔧 Đang cài đặt/cập nhật packages...
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ✅ Đã hoàn thành cài đặt packages
pause
goto MENU

:RUN_PRODUCTION
echo.
echo 📈 Đang chạy Production Setup...
echo.
python scripts\final_production_setup.py
goto MENU_RETURN

:PYTHON_INTERACTIVE
echo.
echo 🐍 Mở Python Interactive với môi trường dự án...
echo 💡 Bạn có thể import các module từ app, models, etc.
echo.
python -i -c "import sys; sys.path.append('.'); print('🚀 Python interactive với crypto-ml project')"
goto MENU_RETURN

:OPEN_CMD
echo.
echo 💻 Mở Command Prompt với môi trường ảo đã kích hoạt...
echo 💡 Môi trường ảo vẫn hoạt động trong cửa sổ mới
echo.
start cmd /k "title Crypto ML Environment && echo ✅ Môi trường ảo crypto-venv đã được kích hoạt"
goto MENU

:MENU_RETURN
echo.
echo Nhan phim bat ky de quay lai menu chinh...
pause >nul
echo.
goto MENU

:EXIT
echo.
echo 👋 Đang thoát...
echo 💡 Môi trường ảo sẽ được deactivate
echo.
call crypto-venv\Scripts\deactivate.bat 2>nul
cd /d "%ORIGINAL_DIR%"
exit /b 0