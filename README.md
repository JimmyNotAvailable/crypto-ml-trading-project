# 🚀 Crypto ML Trading Project

> **Dự án Machine Learning dự đoán giá cryptocurrency với interface tiếng Việt**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)

## 📋 Mô tả dự án

Dự án Machine Learning toàn diện để dự đoán giá cryptocurrency, bao gồm:

- 🤖 **Thuật toán ML**: Linear Regression, KNN, K-Means Clustering
- 📊 **Phân tích dữ liệu**: Feature engineering, time series analysis  
- 🌐 **Web Dashboard**: Flask interface để hiển thị kết quả
- 📈 **Trading Bot**: Tích hợp với bot Telegram
- 🔄 **Pipeline tự động**: Thu thập, xử lý và training dữ liệu

## 🛠️ Công nghệ sử dụng

- **Backend**: Python, Flask, SQLAlchemy
- **Machine Learning**: Scikit-learn, Pandas, Numpy
- **Database**: PostgreSQL, SQLite
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Docker Compose
- **Bot**: python-telegram-bot
- **Monitoring**: Prometheus, Grafana

## 📦 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Docker & Docker Compose (tùy chọn)
- PostgreSQL (hoặc SQLite)

### Cài đặt môi trường

```bash
# 1. Clone repository
git clone https://github.com/JimmyNotAvailable/crypto-ml-trading-project.git
cd crypto-ml-trading-project

# 2. Tạo virtual environment
python -m venv crypto-venv
crypto-venv\Scripts\activate  # Windows
# source crypto-venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình environment
cp .env.example .env
# Chỉnh sửa file .env với thông tin của bạn

# 5. Khởi tạo database
python scripts/setup_database.py

# 6. Chạy data pipeline
python scripts/data_collection/collect_crypto_data.py
```

## 🚀 Sử dụng

### 1. Ví dụ Machine Learning cơ bản

```python
from examples.ml.basic_usage import vi_du_1_hoi_quy_tuyen_tinh

# Chạy ví dụ dự đoán giá với Linear Regression
vi_du_1_hoi_quy_tuyen_tinh()
```

### 2. Web Dashboard

```bash
# Khởi động web server
python web/app.py

# Truy cập: http://localhost:5000
```

### 3. Trading Bot

```bash
# Khởi động Telegram bot
python app/bot.py
```

## 📊 Cấu trúc dự án

```
crypto-project/
├── app/                    # Core application
│   ├── ml/                # ML algorithms & models
│   ├── services/          # Business logic
│   └── bot.py            # Telegram bot
├── examples/              # ML examples (Tiếng Việt)
│   └── ml/               # Ví dụ thuật toán ML
├── scripts/              # Utility scripts
│   ├── data_collection/  # Thu thập dữ liệu
│   └── training/         # Training models
├── web/                  # Web interface
├── data/                 # Datasets & cache
├── models/               # Trained models
├── config/               # Configuration files
└── docs/                 # Documentation
```

## 🎓 Ví dụ ML (Tiếng Việt)

Dự án bao gồm các ví dụ học máy chi tiết bằng tiếng Việt:

- **Hồi quy tuyến tính**: Dự đoán giá crypto
- **KNN Classification**: Dự đoán xu hướng tăng/giảm  
- **KNN Regression**: Dự đoán giá phi tuyến
- **K-Means Clustering**: Phân tích chế độ thị trường

```bash
# Chạy tất cả ví dụ
python examples/ml/basic_usage.py
```

## 📈 Performance

- **Độ chính xác**: R² > 0.95 cho dự đoán giá
- **Tốc độ**: < 100ms cho mỗi prediction
- **Throughput**: 1000+ predictions/second

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📄 License

Dự án được phân phối dưới MIT License. Xem file [LICENSE](LICENSE) để biết chi tiết.

## 👨‍💻 Tác giả

- **Jimmy** - *Initial work* - [JimmyNotAvailable](https://github.com/JimmyNotAvailable)

## 🙏 Cảm ơn

- [Binance API](https://binance-docs.github.io/apidocs/) cho dữ liệu crypto
- [Scikit-learn](https://scikit-learn.org/) cho ML algorithms
- [Flask](https://flask.palletsprojects.com/) cho web framework

---

⭐ Nếu dự án hữu ích, đừng quên để lại một star!
