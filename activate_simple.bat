@echo off
:: =====================================
:: SIMPLE ENV ACTIVATION - CRYPTO ML
:: =====================================

:: Chuyển về thư mục dự án
cd /d "e:\Code on PC\DoAnMLPython\crypto-project-clean"

:: Kích hoạt môi trường ảo
echo 🔄 Kích hoạt môi trường ảo crypto-venv...
call crypto-venv\Scripts\activate.bat

:: Kiểm tra kích hoạt thành công
if errorlevel 1 (
    echo ❌ Lỗi kích hoạt môi trường ảo
    pause
    exit /b 1
)

echo ✅ Môi trường ảo đã được kích hoạt!
echo 💡 Bạn có thể chạy các lệnh Python với môi trường này
echo.
echo 📋 LỆNH HAY DÙNG:
echo    python app\bot.py                    # Chạy Discord bot
echo    python examples\ml\web_dashboard.py  # Chạy web dashboard
echo    python scripts\continuous_collector.py # Chạy data collector
echo    pip install [package]               # Cài package mới
echo    python -m pip list                  # Xem packages đã cài
echo.

:: Giữ cửa sổ mở với môi trường ảo
cmd /k