#!/usr/bin/env python3
"""
🎯 HƯỚNG DẪN TÍCH HỢP WEB - TIẾNG VIỆT
=====================================

Hướng dẫn chi tiết cách tích hợp các mô hình ML vào web interface,
cách hiểu thuật toán và chuẩn bị cho việc viết tay theo lý thuyết.
"""

import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Thêm thư mục gốc dự án vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel

def huong_dan_1_hieu_thuat_toan_ml():
    """Hướng dẫn 1: Hiểu rõ cách thuật toán ML hoạt động"""
    print("🧠 HƯỚNG DẪN 1: HIỂU RÕ THUẬT TOÁN ML")
    print("=" * 60)
    
    print(f"\n📚 LÝ THUYẾT CƠ BẢN CÁC THUẬT TOÁN:")
    
    print(f"\n🔸 1. LINEAR REGRESSION (HỒI QUY TUYẾN TÍNH):")
    print(f"   📊 Mục đích: Dự đoán giá trị liên tục (giá Bitcoin)")
    print(f"   🧮 Công thức: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ")
    print(f"   🎯 Cách hoạt động:")
    print(f"      • Tìm đường thẳng 'phù hợp nhất' với dữ liệu")
    print(f"      • Sử dụng phương pháp bình phương tối thiểu")
    print(f"      • Tối ưu hóa để giảm sai số dự đoán")
    print(f"   💡 Ưu điểm: Đơn giản, nhanh, dễ hiểu")
    print(f"   ⚠️ Nhược điểm: Chỉ hiệu quả với quan hệ tuyến tính")
    
    print(f"\n🔸 2. K-NEAREST NEIGHBORS (KNN):")
    print(f"   📊 Mục đích: Phân loại xu hướng (Bullish/Bearish) hoặc dự đoán")
    print(f"   🧮 Nguyên lý: 'Những điểm gần nhau có tính chất giống nhau'")
    print(f"   🎯 Cách hoạt động:")
    print(f"      • Tính khoảng cách đến tất cả điểm trong tập train")
    print(f"      • Chọn K điểm gần nhất")
    print(f"      • Phân loại: Bỏ phiếu đa số")
    print(f"      • Hồi quy: Lấy trung bình")
    print(f"   💡 Ưu điểm: Không cần giả định về dữ liệu")
    print(f"   ⚠️ Nhược điểm: Chậm khi dữ liệu lớn, nhạy cảm với nhiễu")
    
    print(f"\n🔸 3. K-MEANS CLUSTERING:")
    print(f"   📊 Mục đích: Phân nhóm thị trường (Bull/Bear/Sideways)")
    print(f"   🧮 Nguyên lý: Nhóm các điểm tương tự lại với nhau")
    print(f"   🎯 Cách hoạt động:")
    print(f"      • Chọn K tâm cụm ban đầu ngẫu nhiên")
    print(f"      • Gán mỗi điểm vào cụm gần nhất")
    print(f"      • Cập nhật tâm cụm = trung bình điểm trong cụm")
    print(f"      • Lặp lại cho đến khi hội tụ")
    print(f"   💡 Ưu điểm: Đơn giản, phát hiện pattern ẩn")
    print(f"   ⚠️ Nhược điểm: Cần biết trước số cụm K")
    
    print(f"\n🔬 VÍ DỤ CỤ THỂ VỚI DỮ LIỆU CRYPTO:")
    
    # Tải dữ liệu để minh họa
    try:
        datasets = load_prepared_datasets('ml_datasets_top3')
        du_lieu_mau = datasets['X_train'].head(5)
        gia_mau = datasets['y_train']['price'].head(5)
        
        print(f"\n📊 Dữ liệu đầu vào mẫu (5 mẫu đầu tiên):")
        print(f"   Đặc trưng: {list(du_lieu_mau.columns)[:5]}...")
        print(f"   Kích thước: {du_lieu_mau.shape}")
        
        print(f"\n🎯 Giá Bitcoin tương ứng:")
        for i, (idx, gia) in enumerate(gia_mau.items()):
            print(f"   Mẫu {i+1}: ${gia:.2f}")
        
        # Minh họa Linear Regression
        print(f"\n🔍 MINH HỌA LINEAR REGRESSION:")
        mo_hinh_lr = LinearRegressionModel(target_type='price')
        mo_hinh_lr.train(datasets)
        
        # Lấy hệ số mô hình
        if hasattr(mo_hinh_lr.model, 'coef_') and hasattr(mo_hinh_lr.model, 'intercept_'):
            print(f"   📈 Phương trình dự đoán:")
            print(f"      Giá = {mo_hinh_lr.model.intercept_:.2f} + ")
            
            # Hiển thị một vài hệ số quan trọng nhất
            he_so = mo_hinh_lr.model.coef_
            ten_dac_trung = du_lieu_mau.columns
            
            # Sắp xếp theo độ lớn hệ số
            chi_so_sap_xep = np.argsort(np.abs(he_so))[-5:]  # Top 5
            
            for i, idx in enumerate(chi_so_sap_xep):
                dau = "+" if he_so[idx] >= 0 else ""
                print(f"             {dau}{he_so[idx]:.6f} × {ten_dac_trung[idx]}")
                if i == 2:  # Chỉ hiển thị 3 đầu tiên
                    print(f"             + ... (còn {len(he_so)-3} đặc trưng khác)")
                    break
        
        # Minh họa KNN
        print(f"\n🔍 MINH HỌA K-NEAREST NEIGHBORS:")
        mo_hinh_knn = KNNClassifier(n_neighbors=5)
        mo_hinh_knn.train(datasets)
        
        # Dự đoán mẫu đầu tiên
        mau_du_doan = du_lieu_mau.iloc[[0]]
        du_doan_knn = mo_hinh_knn.predict(mau_du_doan)[0]
        xac_suat = mo_hinh_knn.predict_proba(mau_du_doan)[0]
        
        print(f"   🎯 Dự đoán xu hướng cho mẫu đầu tiên:")
        print(f"      Kết quả: {du_doan_knn}")
        print(f"      Xác suất:")
        
        cac_lop = mo_hinh_knn.model.classes_
        for lop, xs in zip(cac_lop, xac_suat):
            print(f"         {lop}: {xs:.3f} ({xs*100:.1f}%)")
        
        print(f"   💡 Giải thích: KNN tìm 5 điểm gần nhất và bỏ phiếu")
        
        # Minh họa K-Means
        print(f"\n🔍 MINH HỌA K-MEANS CLUSTERING:")
        mo_hinh_kmeans = KMeansClusteringModel(n_clusters=3)
        mo_hinh_kmeans.train(datasets)
        
        # Dự đoán cụm cho mẫu đầu tiên
        cum_du_doan = mo_hinh_kmeans.predict(mau_du_doan)[0]
        print(f"   🎯 Cụm thị trường cho mẫu đầu tiên: Cụm {cum_du_doan}")
        
        # Hiển thị tâm cụm
        if hasattr(mo_hinh_kmeans.model, 'cluster_centers_'):
            tam_cum = mo_hinh_kmeans.model.cluster_centers_
            print(f"   🎯 Có {len(tam_cum)} cụm thị trường được phát hiện:")
            for i, tam in enumerate(tam_cum):
                print(f"      Cụm {i}: Đặc trưng trung bình = {tam[:3].round(3)}...")
        
        print(f"\n🎉 Đã minh họa thành công cách hoạt động của 3 thuật toán!")
        
        return {
            'mo_hinh_lr': mo_hinh_lr,
            'mo_hinh_knn': mo_hinh_knn,
            'mo_hinh_kmeans': mo_hinh_kmeans,
            'du_lieu_mau': du_lieu_mau
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong minh họa thuật toán: {e}")
        return None

def huong_dan_2_tich_hop_web():
    """Hướng dẫn 2: Tích hợp ML vào web interface"""
    print("\n🌐 HƯỚNG DẪN 2: TÍCH HỢP ML VÀO WEB")
    print("=" * 60)
    
    print(f"\n📋 KIẾN TRÚC WEB ML SYSTEM:")
    print(f"""
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Frontend      │    │    Backend       │    │   ML Models     │
    │   (HTML/JS)     │◄──►│   (Flask/API)    │◄──►│  (Trained)      │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
           │                        │                        │
           │                        │                        │
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Dashboard     │    │    Database      │    │   Real-time     │
    │   (Charts)      │    │   (SQLite)       │    │   Data Feed     │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    """)
    
    print(f"\n🔧 CÁC THÀNH PHẦN CHÍNH:")
    
    print(f"\n🔸 1. BACKEND API (Flask):")
    print(f"   📊 /api/du-doan-moi - Tạo dự đoán mới")
    print(f"   📈 /api/bieu-do-gia - Dữ liệu biểu đồ giá")
    print(f"   📊 /api/bieu-do-tin-cay - Biểu đồ độ tin cậy")
    print(f"   ⚖️ /api/so-sanh-mo-hinh - So sánh hiệu suất")
    print(f"   🤖 /api/bot-trading - Quản lý bot giao dịch")
    
    print(f"\n🔸 2. FRONTEND DASHBOARD:")
    print(f"   📈 Biểu đồ dự đoán giá real-time")
    print(f"   📊 Cards hiển thị metrics quan trọng")
    print(f"   🤖 Giao diện quản lý bot trading")
    print(f"   📋 Bảng so sánh hiệu suất mô hình")
    print(f"   🔄 Cập nhật tự động mỗi 30 giây")
    
    print(f"\n🔸 3. DATABASE (SQLite):")
    print(f"   💾 Lưu trữ lịch sử dự đoán")
    print(f"   📊 Theo dõi hiệu suất mô hình")
    print(f"   💰 Ghi nhận giao dịch bot")
    print(f"   📈 Phân tích xu hướng theo thời gian")
    
    # Tạo ví dụ mã API endpoint
    print(f"\n💻 VÍ DỤ CODE ENDPOINT API:")
    print(f"""
@app.route('/api/du-doan-moi')
def api_du_doan_moi():
    try:
        # 1. Lấy dữ liệu thị trường mới nhất
        du_lieu_thi_truong = lay_du_lieu_moi_nhat()
        
        # 2. Sử dụng ML để dự đoán
        ket_qua = bot_giao_dich.phan_tich_thi_truong(du_lieu_thi_truong)
        
        # 3. Lưu vào database
        luu_du_doan_vao_db(ket_qua)
        
        # 4. Trả về JSON cho frontend
        return jsonify({{
            'trang_thai': 'thanh_cong',
            'gia_du_doan': ket_qua['gia_du_doan'],
            'do_tin_cay': ket_qua['do_tin_cay'],
            'xu_huong': ket_qua['xu_huong_chinh']
        }})
    except Exception as e:
        return jsonify({{'trang_thai': 'loi', 'thong_bao': str(e)}})
    """)
    
    print(f"\n💻 VÍ DỤ CODE FRONTEND (JavaScript):")
    print(f"""
// Gọi API để lấy dự đoán mới
function cap_nhat_du_doan() {{
    fetch('/api/du-doan-moi')
        .then(response => response.json())
        .then(data => {{
            if (data.trang_thai === 'thanh_cong') {{
                // Cập nhật giao diện
                document.getElementById('gia-du-doan').textContent = 
                    '$' + data.gia_du_doan.toFixed(2);
                document.getElementById('do-tin-cay').textContent = 
                    (data.do_tin_cay * 100).toFixed(1) + '%';
                
                // Cập nhật biểu đồ
                cap_nhat_bieu_do();
            }}
        }});
}}

// Tự động cập nhật mỗi 30 giây
setInterval(cap_nhat_du_doan, 30000);
    """)
    
    print(f"\n🎯 QUY TRÌNH HOẠT ĐỘNG:")
    print(f"   1️⃣ User truy cập dashboard")
    print(f"   2️⃣ Frontend load dữ liệu từ API")
    print(f"   3️⃣ Backend chạy ML models để dự đoán")
    print(f"   4️⃣ Kết quả được lưu vào database")
    print(f"   5️⃣ Frontend hiển thị biểu đồ và metrics")
    print(f"   6️⃣ Tự động cập nhật theo thời gian thực")
    
    print(f"\n🔧 CÀI ĐẶT VÀ TRIỂN KHAI:")
    print(f"   pip install flask plotly pandas numpy")
    print(f"   python web_dashboard.py")
    print(f"   Truy cập: http://localhost:5000")
    
    return True

def huong_dan_3_chuan_bi_viet_tay():
    """Hướng dẫn 3: Chuẩn bị để viết thuật toán bằng tay theo lý thuyết"""
    print("\n✍️ HƯỚNG DẪN 3: CHUẨN BỊ VIẾT THUẬT TOÁN BẰNG TAY")
    print("=" * 60)
    
    print(f"\n📚 TẠI SAO CẦN VIẾT BẰNG TAY THEO LÝ THUYẾT:")
    print(f"   🧠 Hiểu sâu cách thuật toán hoạt động")
    print(f"   🔧 Tùy chỉnh theo nhu cầu cụ thể")
    print(f"   🎯 Tối ưu hóa cho dữ liệu crypto")
    print(f"   📈 Cải thiện hiệu suất và độ chính xác")
    print(f"   💡 Phát triển thuật toán mới")
    
    print(f"\n📖 KIẾN THỨC TOÁN HỌC CẦN THIẾT:")
    
    print(f"\n🔸 1. CHO LINEAR REGRESSION:")
    print(f"   📊 Đại số tuyến tính:")
    print(f"      • Phép nhân ma trận")
    print(f"      • Ma trận nghịch đảo")
    print(f"      • Phép chuyển vị (transpose)")
    print(f"   📈 Giải tích:")
    print(f"      • Đạo hàm riêng")
    print(f"      • Gradient descent")
    print(f"      • Cost function (MSE)")
    print(f"   🧮 Công thức Normal Equation:")
    print(f"      θ = (X^T × X)^(-1) × X^T × y")
    
    print(f"\n🔸 2. CHO K-NEAREST NEIGHBORS:")
    print(f"   📏 Khoảng cách:")
    print(f"      • Euclidean: √[(x₁-x₂)² + (y₁-y₂)²]")
    print(f"      • Manhattan: |x₁-x₂| + |y₁-y₂|")
    print(f"      • Cosine similarity")
    print(f"   📊 Thống kê:")
    print(f"      • Trung bình có trọng số")
    print(f"      • Bỏ phiếu đa số")
    print(f"      • Xử lý tie-breaking")
    
    print(f"\n🔸 3. CHO K-MEANS CLUSTERING:")
    print(f"   📊 Tối ưu hóa:")
    print(f"      • Lloyd's algorithm")
    print(f"      • Within-cluster sum of squares (WCSS)")
    print(f"      • Elbow method để chọn K")
    print(f"   🎯 Khởi tạo:")
    print(f"      • Random initialization")
    print(f"      • K-means++ (smarter init)")
    print(f"      • Xử lý convergence")
    
    print(f"\n💻 TEMPLATE CODE ĐỂ BẮT ĐẦU:")
    
    print(f"\n🔸 LINEAR REGRESSION BẰNG TAY:")
    print(f"""
class LinearRegressionTuViet:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.lich_su_loss = []
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        # Khởi tạo weights và bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Tính cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            self.lich_su_loss.append(cost)
            
            # Tính gradients
            dw = (2/len(X)) * np.dot(X.T, (y_pred - y))
            db = (2/len(X)) * np.sum(y_pred - y)
            
            # Cập nhật parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    """)
    
    print(f"\n🔸 KNN BẰNG TAY:")
    print(f"""
class KNNTuViet:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Tính khoảng cách đến tất cả điểm train
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sắp xếp và lấy K gần nhất
            distances.sort()
            k_nearest = distances[:self.k]
            
            # Dự đoán (classification: vote, regression: mean)
            k_labels = [label for _, label in k_nearest]
            prediction = self._aggregate(k_labels)
            predictions.append(prediction)
        
        return predictions
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _aggregate(self, labels):
        # Cho classification: return most common
        # Cho regression: return mean
        return np.mean(labels)  # Đơn giản hóa
    """)
    
    print(f"\n🔸 K-MEANS BẰNG TAY:")
    print(f"""
class KMeansTuViet:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        # Khởi tạo centroids ngẫu nhiên
        n_features = X.shape[1]
        self.centroids = np.random.randn(self.k, n_features)
        
        for iteration in range(self.max_iters):
            # Gán mỗi điểm vào cluster gần nhất
            distances = self._calculate_distances(X)
            new_labels = np.argmin(distances, axis=1)
            
            # Kiểm tra convergence
            if hasattr(self, 'labels') and np.array_equal(new_labels, self.labels):
                break
            
            self.labels = new_labels
            
            # Cập nhật centroids
            for k in range(self.k):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels == k], axis=0)
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    """)
    
    print(f"\n🎯 LỘ TRÌNH HỌC VÀ PHÁT TRIỂN:")
    print(f"   1️⃣ Hiểu rõ lý thuyết từ ví dụ có sẵn")
    print(f"   2️⃣ Viết lại từng thuật toán bằng tay")
    print(f"   3️⃣ Test với dữ liệu crypto thực tế")
    print(f"   4️⃣ So sánh với thư viện sklearn")
    print(f"   5️⃣ Tối ưu hóa cho crypto trading")
    print(f"   6️⃣ Phát triển thuật toán hybrid mới")
    
    print(f"\n📖 TÀI LIỆU THAM KHẢO NÂNG CAO:")
    print(f"   📚 'Pattern Recognition and Machine Learning' - Bishop")
    print(f"   📚 'The Elements of Statistical Learning' - Hastie")
    print(f"   📚 'Hands-On Machine Learning' - Aurélien Géron")
    print(f"   🌐 Coursera Machine Learning - Andrew Ng")
    print(f"   🌐 Khan Academy - Linear Algebra")
    
    return True

def huong_dan_4_ket_qua_web():
    """Hướng dẫn 4: Format kết quả để hiển thị trên web"""
    print("\n📱 HƯỚNG DẪN 4: FORMAT KẾT QUẢ CHO WEB")
    print("=" * 60)
    
    print(f"\n🎨 NGUYÊN TẮC THIẾT KẾ WEB INTERFACE:")
    print(f"   👀 Thông tin quan trọng nổi bật")
    print(f"   📊 Biểu đồ trực quan dễ hiểu")
    print(f"   🎯 Hành động rõ ràng (Mua/Bán/Giữ)")
    print(f"   ⚡ Cập nhật real-time")
    print(f"   📱 Responsive cho mobile")
    
    print(f"\n📋 CẤU TRÚC HIỂN THỊ KẾT QUẢ:")
    
    # Tạo ví dụ kết quả ML formatted cho web
    ket_qua_mau = {
        'du_doan_gia': {
            'gia_hien_tai': 45230.50,
            'gia_du_doan': 46150.75,
            'thay_doi_phan_tram': 2.03,
            'do_tin_cay': 0.847,
            'thoi_gian': '2024-01-15 14:30:25'
        },
        'phan_loai_xu_huong': {
            'xu_huong_chinh': 'Bullish',
            'xac_suat': {
                'Bullish': 0.753,
                'Bearish': 0.142,
                'Sideways': 0.105
            }
        },
        'phan_tich_thi_truong': {
            'cum_thi_truong': 1,
            'mo_ta_cum': 'Thị trường tăng trưởng',
            'do_manh': 'Cao'
        },
        'tin_hieu_giao_dich': {
            'hanh_dong': 'MUA',
            'ly_do': 'Tín hiệu tăng mạnh với độ tin cậy cao',
            'kich_thuoc_vi_the': 0.15,
            'muc_rui_ro': 'Trung bình'
        }
    }
    
    print(f"\n🔸 1. DASHBOARD CARDS (Thẻ tổng quan):")
    print(f"   💰 Card Giá dự đoán:")
    print(f"      Hiển thị: ${ket_qua_mau['du_doan_gia']['gia_du_doan']:,.2f}")
    print(f"      Màu sắc: {'🟢 Xanh' if ket_qua_mau['du_doan_gia']['thay_doi_phan_tram'] > 0 else '🔴 Đỏ'}")
    print(f"      Badge: +{ket_qua_mau['du_doan_gia']['thay_doi_phan_tram']:.2f}%")
    
    print(f"   🧠 Card Độ tin cậy:")
    print(f"      Hiển thị: {ket_qua_mau['du_doan_gia']['do_tin_cay']:.1%}")
    print(f"      Progress bar: {ket_qua_mau['du_doan_gia']['do_tin_cay']*100:.0f}%")
    print(f"      Màu: {'🟢 Xanh' if ket_qua_mau['du_doan_gia']['do_tin_cay'] > 0.7 else '🟡 Vàng' if ket_qua_mau['du_doan_gia']['do_tin_cay'] > 0.5 else '🔴 Đỏ'}")
    
    print(f"   📈 Card Xu hướng:")
    print(f"      Hiển thị: {ket_qua_mau['phan_loai_xu_huong']['xu_huong_chinh']}")
    print(f"      Icon: {'📈' if ket_qua_mau['phan_loai_xu_huong']['xu_huong_chinh'] == 'Bullish' else '📉'}")
    print(f"      Xác suất: {ket_qua_mau['phan_loai_xu_huong']['xac_suat']['Bullish']:.1%}")
    
    print(f"\n🔸 2. TÍN HIỆU GIAO DỊCH (Alert/Notification):")
    hanh_dong = ket_qua_mau['tin_hieu_giao_dich']['hanh_dong']
    mau_sac_hanh_dong = {
        'MUA': '🟢 bg-success',
        'BÁN': '🔴 bg-danger', 
        'GIỮ': '🟡 bg-warning'
    }
    
    print(f"   🎯 Alert chính:")
    print(f"      '{mau_sac_hanh_dong[hanh_dong]} {hanh_dong}'")
    print(f"      Lý do: '{ket_qua_mau['tin_hieu_giao_dich']['ly_do']}'")
    print(f"      Kích thước: {ket_qua_mau['tin_hieu_giao_dich']['kich_thuoc_vi_the']:.0%} tài khoản")
    
    print(f"\n🔸 3. BIỂU ĐỒ VÀ VISUALIZATIONS:")
    print(f"   📊 Biểu đồ giá:")
    print(f"      • Line chart: Giá thực tế vs Dự đoán")
    print(f"      • Màu sắc: Xanh (thực tế), Đỏ đứt nét (dự đoán)")
    print(f"      • Tooltip: Hiển thị chi tiết khi hover")
    
    print(f"   📈 Biểu đồ độ tin cậy:")
    print(f"      • Bar chart theo thời gian")
    print(f"      • Ngưỡng 70% (xanh), 50% (vàng)")
    print(f"      • Cảnh báo khi < 50%")
    
    print(f"   🥧 Pie chart xu hướng:")
    print(f"      • Phân bố xác suất Bullish/Bearish/Sideways")
    print(f"      • Màu sắc: Xanh/Đỏ/Xám")
    
    print(f"\n💻 VÍ DỤ HTML TEMPLATE:")
    template_html = '''
<!-- Card dự đoán giá -->
<div class="card bg-primary text-white">
    <div class="card-body">
        <h5><i class="fas fa-chart-line"></i> Dự đoán Giá</h5>
        <h2>${{ ket_qua.gia_du_doan | number_format(2) }}</h2>
        <span class="badge bg-{{ 'success' if ket_qua.thay_doi > 0 else 'danger' }}">
            {{ ket_qua.thay_doi_phan_tram | show_change }}%
        </span>
        <small class="d-block mt-2">
            Độ tin cậy: {{ ket_qua.do_tin_cay | percentage }}
        </small>
    </div>
</div>

<!-- Alert tín hiệu giao dịch -->
<div class="alert alert-{{ 'success' if tin_hieu.hanh_dong == 'MUA' else 'danger' if tin_hieu.hanh_dong == 'BÁN' else 'warning' }}">
    <h6><i class="fas fa-exclamation-triangle"></i> Tín hiệu: {{ tin_hieu.hanh_dong }}</h6>
    <p>{{ tin_hieu.ly_do }}</p>
    <small>Kích thước đề xuất: {{ tin_hieu.kich_thuoc_vi_the | percentage }}</small>
</div>
    '''
    print(template_html)
    
    print(f"\n🔸 4. MOBILE RESPONSIVE:")
    print(f"   📱 Stack cards vertically on mobile")
    print(f"   🖥️ Grid layout on desktop")
    print(f"   👆 Touch-friendly buttons")
    print(f"   ⚡ Fast loading với lazy loading")
    
    print(f"\n🔸 5. REAL-TIME UPDATES:")
    print(f"   🔄 WebSocket hoặc polling mỗi 30s")
    print(f"   🎨 Smooth animations khi update")
    print(f"   🔔 Push notifications cho tín hiệu quan trọng")
    print(f"   💾 Cache dữ liệu để load nhanh")
    
    print(f"\n🎯 CHECKLIST HOÀN THIỆN WEB:")
    print(f"   ✅ Dashboard tổng quan")
    print(f"   ✅ Biểu đồ interactive")
    print(f"   ✅ Alerts và notifications")
    print(f"   ✅ Mobile responsive")
    print(f"   ✅ Real-time updates")
    print(f"   ✅ Lịch sử dự đoán")
    print(f"   ✅ So sánh mô hình")
    print(f"   ✅ Export dữ liệu")
    print(f"   ✅ Settings và cấu hình")
    print(f"   ✅ Help và documentation")
    
    return ket_qua_mau

def main():
    """Chạy tất cả hướng dẫn tích hợp web"""
    print("🎯 HƯỚNG DẪN TÍCH HỢP WEB & HIỂU THUẬT TOÁN ML")
    print("=" * 70)
    print("Hướng dẫn chi tiết cách hiểu thuật toán, tích hợp web,")
    print("và chuẩn bị để viết bằng tay theo lý thuyết.")
    print("")
    print("🎯 NỘI DUNG HƯỚNG DẪN:")
    print("   🧠 1. Hiểu rõ cách thuật toán ML hoạt động")
    print("   🌐 2. Tích hợp ML vào web interface")
    print("   ✍️ 3. Chuẩn bị viết thuật toán bằng tay")
    print("   📱 4. Format kết quả cho web hiển thị")
    
    cac_huong_dan = [
        ("Hiểu thuật toán ML", huong_dan_1_hieu_thuat_toan_ml),
        ("Tích hợp web", huong_dan_2_tich_hop_web),
        ("Chuẩn bị viết tay", huong_dan_3_chuan_bi_viet_tay),
        ("Format kết quả web", huong_dan_4_ket_qua_web)
    ]
    
    ket_qua = {}
    thanh_cong = 0
    
    for i, (ten_huong_dan, ham_huong_dan) in enumerate(cac_huong_dan, 1):
        try:
            print(f"\n🔄 Đang chạy Hướng dẫn {i}: {ten_huong_dan}...")
            result = ham_huong_dan()
            ket_qua[ham_huong_dan.__name__] = result
            
            if result is not None:
                thanh_cong += 1
                print(f"✅ Hướng dẫn {i} hoàn thành thành công!")
            else:
                print(f"⚠️ Hướng dẫn {i} có vấn đề!")
                
        except Exception as e:
            print(f"❌ Hướng dẫn {i} thất bại: {e}")
            ket_qua[ham_huong_dan.__name__] = None
    
    # Tổng kết
    tong_so = len(cac_huong_dan)
    print(f"\n{'='*70}")
    print("🎯 TỔNG KẾT HƯỚNG DẪN TÍCH HỢP")
    print(f"{'='*70}")
    print(f"✅ Hoàn thành: {thanh_cong}/{tong_so}")
    
    if thanh_cong == tong_so:
        print("\n🎉 TẤT CẢ HƯỚNG DẪN ĐÃ HOÀN THÀNH!")
        print("🚀 Bạn đã sẵn sàng để tích hợp ML vào web!")
        
        print(f"\n📚 KIẾN THỨC ĐÃ THÀNH THẠO:")
        print(f"   🧠 Hiểu rõ cách hoạt động của Linear Regression, KNN, K-Means")
        print(f"   🌐 Tích hợp ML vào Flask web application")
        print(f"   ✍️ Chuẩn bị để implement thuật toán bằng tay")
        print(f"   📱 Thiết kế web interface cho ML results")
        print(f"   📊 Tạo dashboard và biểu đồ interactive")
        print(f"   🔄 Xây dựng system real-time updates")
        
        print(f"\n🎯 BƯỚC TIẾP THEO:")
        print(f"   1. Chạy web dashboard: python web_dashboard.py")
        print(f"   2. Truy cập: http://localhost:5000")
        print(f"   3. Test các tính năng ML trong web")
        print(f"   4. Bắt đầu viết thuật toán bằng tay")
        print(f"   5. Tối ưu hóa cho dữ liệu crypto")
        print(f"   6. Deploy lên production server")
        
        print(f"\n🔗 TÀI NGUYÊN HỖ TRỢ:")
        print(f"   📁 examples/ml/production_examples.py - Bot trading")
        print(f"   🌐 examples/ml/web_dashboard.py - Web dashboard")
        print(f"   📊 examples/ml/basic_usage.py - ML examples")
        print(f"   🔧 examples/ml/advanced_pipeline.py - Advanced pipeline")
        
        print(f"\n💡 LƯU Ý QUAN TRỌNG:")
        print(f"   🚨 Đây là môi trường học tập, không phải tư vấn đầu tư")
        print(f"   📊 Luôn test kỹ lưỡng trước khi sử dụng thực tế")
        print(f"   💰 Quản lý rủi ro cẩn thận trong trading")
        print(f"   🔄 Cập nhật mô hình thường xuyên")
        print(f"   📚 Tiếp tục học hỏi và cải thiện")
        
    else:
        print(f"\n⚠️ MỘT SỐ HƯỚNG DẪN CẦN KHẮC PHỤC")
        print("🔍 Xem chi tiết lỗi ở trên để điều chỉnh.")
    
    return ket_qua

if __name__ == "__main__":
    main()