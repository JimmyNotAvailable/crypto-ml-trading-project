#!/usr/bin/env python3
"""
🎯 VÍ DỤ SỬ DỤNG CƠ BẢN ML - TIẾNG VIỆT (ĐÃ TỐI ƯU)
====================================================

Các ví dụ đơn giản minh họa cách sử dụng từng thuật toán ML trong dự án crypto.
Hoàn hảo cho người mới bắt đầu hiểu các khái niệm cố        # 4. Thực hiện dự đoán
        print("\n🔮 Thực hiện dự đoán giá...")
        test_predictions = model.predict(datasets['raw']['X_test_raw'].head(5))
        actual_prices = datasets['raw']['y_test']['price'].head(5).valuesi và cách sử dụng cơ bản.

Tất cả kết quả và giải thích được in ra bằng tiếng Việt để dễ hiểu và tích hợp vào web.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Thêm thư mục gốc dự án vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel

def chuyen_doi_dinh_dang_datasets(raw_datasets):
    """Chuyển đổi từ định dạng X_train/y_train sang train/test DataFrame"""
    # Tạo train DataFrame
    train_df = raw_datasets['X_train_raw'].copy()
    
    # Thêm targets từ y_train (là dict)
    y_train = raw_datasets['y_train']
    if 'price' in y_train:
        train_df['target_price'] = y_train['price']
    if 'price_change' in y_train:
        train_df['target_price_change'] = y_train['price_change'] 
    if 'trend' in y_train:
        train_df['target_trend'] = y_train['trend']
        
    # Tạo test DataFrame
    test_df = raw_datasets['X_test_raw'].copy()
    
    # Thêm targets từ y_test (là dict)
    y_test = raw_datasets['y_test']
    if 'price' in y_test:
        test_df['target_price'] = y_test['price']
    if 'price_change' in y_test:
        test_df['target_price_change'] = y_test['price_change']
    if 'trend' in y_test:
        test_df['target_trend'] = y_test['trend']
    
    return {
        'train': train_df,
        'test': test_df,
        'raw': raw_datasets  # Giữ lại raw data để dự đoán
    }

def lay_chi_so_hieu_suat(training_results, model_type='regression'):
    """Helper function để chuẩn hóa kết quả từ các model types khác nhau"""
    if model_type == 'regression':
        return {
            'r2': training_results.get('test_r2', training_results.get('train_r2', 0)),
            'mae': training_results.get('test_mae', training_results.get('train_mae', 0)),
            'rmse': training_results.get('test_rmse', training_results.get('train_rmse', 0))
        }
    elif model_type == 'classification':
        return {
            'accuracy': training_results.get('test_accuracy', training_results.get('train_accuracy', 0)),
            'precision_macro': training_results.get('test_precision_macro', 0.5),
            'recall_macro': training_results.get('test_recall_macro', 0.5),
            'f1_macro': training_results.get('test_f1_macro', 0.5)
        }
    elif model_type == 'clustering':
        return {
            'silhouette_score': training_results.get('silhouette_score', 0),
            'optimal_clusters': training_results.get('n_clusters', 0)
        }
    else:
        return training_results

def in_tieu_de_vi_du(tieu_de):
    """In tiêu đề ví dụ bằng tiếng Việt"""
    print(f"\n{'='*60}")
    print(f"🎯 {tieu_de}")
    print(f"{'='*60}")

def vi_du_1_hoi_quy_tuyen_tinh():
    """Ví dụ 1: Hồi quy tuyến tính cơ bản để dự đoán giá"""
    in_tieu_de_vi_du("HỒI QUY TUYẾN TÍNH - DỰ ĐOÁN GIÁ CRYPTO")
    
    try:
        # 1. Tải dữ liệu
        print("📊 Đang tải dữ liệu...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        print(f"✅ Đã tải {len(raw_datasets['X_train'])} mẫu huấn luyện")
        
        # Chuyển đổi định dạng
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        
        # 2. Khởi tạo mô hình
        print("\n🤖 Tạo mô hình Hồi quy tuyến tính...")
        model = LinearRegressionModel(target_type='price')
        print(f"✅ Mô hình đã khởi tạo: {model.model_name}")
        
        print("\n📚 CÁCH HOẠT ĐỘNG CỦA HỒI QUY TUYẾN TÍNH:")
        print("   🔹 Tìm đường thẳng tốt nhất khớp với dữ liệu")
        print("   🔹 Công thức: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b")
        print("   🔹 Tối ưu hóa bằng cách giảm thiểu sai số bình phương")
        print("   🔹 Phù hợp cho mối quan hệ tuyến tính")
        
        # 3. Huấn luyện mô hình
        print("\n🎯 Huấn luyện mô hình...")
        training_results = model.train(datasets)
        print(f"✅ Huấn luyện hoàn thành!")
        
        # Chuẩn hóa kết quả
        metrics = lay_chi_so_hieu_suat(training_results, 'regression')
        
        print(f"\n📈 KẾT QUẢ HUẤN LUYỆN:")
        print(f"   📊 Điểm R² (độ chính xác): {metrics['r2']:.4f}")
        print(f"   💰 Sai số trung bình (MAE): ${metrics['mae']:.2f}")
        print(f"   📏 Sai số căn bậc hai (RMSE): ${metrics['rmse']:.2f}")
        
        # Giải thích điểm số
        r2_score = metrics['r2']
        if r2_score > 0.9:
            print(f"   🎉 Độ chính xác RẤT CAO - Mô hình dự đoán rất tốt!")
        elif r2_score > 0.8:
            print(f"   👍 Độ chính xác CAO - Mô hình hoạt động tốt")
        elif r2_score > 0.6:
            print(f"   ⚠️ Độ chính xác TRUNG BÌNH - Cần cải thiện")
        else:
            print(f"   ❌ Độ chính xác THẤP - Cần xem xét lại mô hình")
        
        # 4. Thực hiện dự đoán
        print("\n🔮 Thực hiện dự đoán trên dữ liệu kiểm tra...")
        test_predictions = model.predict(datasets['raw']['X_test_raw'].head(5))
        actual_prices = datasets['raw']['y_test']['price'].head(5).values
        
        print("   📊 So sánh Dự đoán vs Thực tế:")
        for i, (pred, actual) in enumerate(zip(test_predictions, actual_prices)):
            error = abs(pred - actual)
            error_percent = (error / actual) * 100
            print(f"      Mẫu {i+1}: Dự đoán ${pred:.2f} | Thực tế ${actual:.2f} | Sai số {error_percent:.1f}%")
        
        # 5. Giải thích kết quả
        print(f"\n💡 GIẢI THÍCH KẾT QUẢ:")
        print(f"   🎯 Mô hình học được mối quan hệ tuyến tính giữa đặc trưng và giá")
        print(f"   📈 R² = {r2_score:.3f} có nghĩa là mô hình giải thích được {r2_score*100:.1f}% biến động giá")
        print(f"   💰 Sai số trung bình ${metrics['mae']:.2f} cho mỗi dự đoán")
        print(f"   🚀 Có thể ứng dụng để dự đoán xu hướng giá ngắn hạn")
        
        return {
            'mo_hinh': model,
            'ket_qua_train': training_results,
            'du_doan_mau': test_predictions,
            'gia_thuc_te': actual_prices
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong ví dụ hồi quy tuyến tính: {e}")
        return None

def vi_du_2_phan_loai_knn():
    """Ví dụ 2: Phân loại KNN để dự đoán xu hướng giá"""
    in_tieu_de_vi_du("PHÂN LOẠI KNN - DỰ ĐOÁN XU HƯỚNG GIÁ")
    
    try:
        # 1. Tải dữ liệu
        print("📊 Đang tải dữ liệu...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        
        # 2. Tạo bộ phân loại KNN
        print("\n🎯 Tạo bộ phân loại KNN...")
        model = KNNClassifier(n_neighbors=5)
        print(f"✅ Bộ phân loại đã khởi tạo: {model.model_name}")
        
        print("\n📚 CÁCH HOẠT ĐỘNG CỦA KNN PHÂN LOẠI:")
        print("   🔹 Tìm K điểm gần nhất trong không gian đặc trưng")
        print("   🔹 Lấy phiếu bầu từ K hàng xóm gần nhất")
        print("   🔹 Phân loại theo nhóm có nhiều phiếu nhất")
        print("   🔹 Không cần huấn luyện trước - 'Lazy Learning'")
        print(f"   🔹 Đang sử dụng K=5 hàng xóm")
        
        # 3. Huấn luyện bộ phân loại
        print("\n🎯 Huấn luyện bộ phân loại...")
        training_results = model.train(datasets)
        print(f"✅ Huấn luyện hoàn thành!")
        
        # Chuẩn hóa kết quả
        metrics = lay_chi_so_hieu_suat(training_results, 'classification')
        
        print(f"\n📈 KẾT QUẢ PHÂN LOẠI:")
        print(f"   🎯 Độ chính xác: {metrics['accuracy']:.4f}")
        print(f"   📊 Precision: {metrics['precision_macro']:.4f}")
        print(f"   📏 Recall: {metrics['recall_macro']:.4f}")
        print(f"   ⚖️ F1-Score: {metrics['f1_macro']:.4f}")
        
        # 4. Thực hiện dự đoán
        print("\n🔮 Thực hiện dự đoán xu hướng...")
        test_predictions = model.predict(datasets['raw']['X_test_raw'].head(5))
        test_probabilities = model.predict_proba(datasets['raw']['X_test_raw'].head(5))
        actual_trends = datasets['raw']['y_test']['trend'].head(5).values
        
        print("   📊 Dự đoán xu hướng:")
        class_labels = model.model.classes_
        for i, (pred, proba, actual) in enumerate(zip(test_predictions, test_probabilities, actual_trends)):
            print(f"      Mẫu {i+1}: Dự đoán '{pred}' | Thực tế '{actual}'")
            print(f"         Xác suất: ", end="")
            for j, label in enumerate(class_labels):
                print(f"{label}: {proba[j]:.3f}", end=" | " if j < len(class_labels)-1 else "\n")
        
        # 5. Giải thích các xu hướng
        print(f"\n📈 GIẢI THÍCH XU HƯỚNG:")
        print(f"   📊 'Bullish' (Tăng): Giá có xu hướng tăng")
        print(f"   📉 'Bearish' (Giảm): Giá có xu hướng giảm")
        print(f"   ➡️ 'Sideways' (Ngang): Giá đi ngang, ít biến động")
        print(f"   🎯 Độ chính xác {metrics['accuracy']*100:.1f}% giúp đưa ra quyết định giao dịch")
        
        return {
            'mo_hinh': model,
            'ket_qua_train': training_results,
            'du_doan_xu_huong': test_predictions,
            'xac_suat': test_probabilities
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong ví dụ phân loại KNN: {e}")
        return None

def vi_du_3_hoi_quy_knn():
    """Ví dụ 3: Hồi quy KNN để dự đoán giá phi tuyến"""
    in_tieu_de_vi_du("HỒI QUY KNN - DỰ ĐOÁN GIÁ PHI TUYẾN")
    
    try:
        # 1. Tải dữ liệu
        print("📊 Đang tải dữ liệu...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        
        # 2. Tạo bộ hồi quy KNN
        print("\n📈 Tạo bộ hồi quy KNN...")
        model = KNNRegressor(n_neighbors=5, target_type='price')
        print(f"✅ Bộ hồi quy đã khởi tạo: {model.model_name}")
        
        print("\n📚 CÁCH HOẠT ĐỘNG CỦA KNN HỒI QUY:")
        print("   🔹 Tìm K điểm gần nhất trong không gian đặc trưng")
        print("   🔹 Tính giá trị trung bình của K hàng xóm")
        print("   🔹 Dự đoán = Trung bình có trọng số theo khoảng cách")
        print("   🔹 Xử lý được mối quan hệ phi tuyến phức tạp")
        print(f"   🔹 Đang sử dụng K=5 hàng xóm")
        
        # 3. Huấn luyện bộ hồi quy
        print("\n🎯 Huấn luyện bộ hồi quy...")
        training_results = model.train(datasets)
        print(f"✅ Huấn luyện hoàn thành!")
        
        # Chuẩn hóa kết quả
        metrics = lay_chi_so_hieu_suat(training_results, 'regression')
        
        print(f"\n📈 KẾT QUẢ HỒI QUY:")
        print(f"   📊 Điểm R²: {metrics['r2']:.4f}")
        print(f"   💰 MAE: ${metrics['mae']:.2f}")
        print(f"   📏 RMSE: ${metrics['rmse']:.2f}")
        
        # 4. So sánh với Linear Regression
        print(f"\n⚖️ SO SÁNH VỚI HỒI QUY TUYẾN TÍNH:")
        print(f"   💪 Ưu điểm KNN: Xử lý quan hệ phi tuyến, không giả định về dữ liệu")
        print(f"   ⚠️ Nhược điểm KNN: Chậm hơn, cần nhiều bộ nhớ, nhạy cảm với nhiễu")
        print(f"   🎯 Phù hợp: Dữ liệu có pattern phức tạp, không tuyến tính")
        
        # 5. Thực hiện dự đoán
        print("\n🔮 Thực hiện dự đoán giá...")
        test_predictions = model.predict(datasets['raw']['X_test_raw'].head(5))
        actual_prices = datasets['raw']['y_test']['price'].head(5).values
        
        print("   📊 So sánh KNN vs Thực tế:")
        for i, (pred, actual) in enumerate(zip(test_predictions, actual_prices)):
            error = abs(pred - actual)
            error_percent = (error / actual) * 100
            print(f"      Mẫu {i+1}: KNN ${pred:.2f} | Thực tế ${actual:.2f} | Sai số {error_percent:.1f}%")
        
        return {
            'mo_hinh': model,
            'ket_qua_train': training_results,
            'du_doan_gia': test_predictions
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong ví dụ hồi quy KNN: {e}")
        return None

def vi_du_4_phan_cum_kmeans():
    """Ví dụ 4: Phân cụm K-Means để phân tích chế độ thị trường"""
    in_tieu_de_vi_du("PHÂN CỤM K-MEANS - PHÂN TÍCH CHẾ ĐỘ THỊ TRƯỜNG")
    
    try:
        # 1. Tải dữ liệu
        print("📊 Đang tải dữ liệu...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        
        # 2. Tạo mô hình phân cụm
        print("\n🎯 Tạo mô hình phân cụm K-Means...")
        model = KMeansClusteringModel(auto_tune=True)
        print(f"✅ Mô hình phân cụm đã khởi tạo: {model.model_name}")
        
        print("\n📚 CÁCH HOẠT ĐỘNG CỦA K-MEANS:")
        print("   🔹 Chia dữ liệu thành K nhóm (cụm)")
        print("   🔹 Tìm tâm cụm tối ưu bằng thuật toán lặp")
        print("   🔹 Mỗi điểm thuộc cụm có tâm gần nhất")
        print("   🔹 Tự động tìm số cụm tối ưu (2-8 cụm)")
        print("   🔹 Phát hiện các chế độ thị trường khác nhau")
        
        # 3. Huấn luyện mô hình phân cụm
        print("\n🎯 Huấn luyện mô hình phân cụm...")
        training_results = model.train(datasets)
        print(f"✅ Phân cụm hoàn thành!")
        
        # Chuẩn hóa kết quả
        metrics = lay_chi_so_hieu_suat(training_results, 'clustering')
        
        print(f"\n📈 KẾT QUẢ PHÂN CỤM:")
        print(f"   🎯 Số cụm tối ưu: {metrics['optimal_clusters']}")
        print(f"   📊 Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"   🎪 Inertia: {training_results.get('inertia', 0):.2f}")
        
        # 4. Phân tích các cụm
        print(f"\n🔍 PHÂN TÍCH CÁC CỤM THỊ TRƯỜNG:")
        test_clusters = model.predict(datasets['raw']['X_test_raw'].head(10))
        
        # Đếm số điểm trong mỗi cụm
        unique_clusters, counts = np.unique(test_clusters, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            print(f"   🎯 Cụm {cluster}: {count} điểm dữ liệu")
        
        # 5. Giải thích ý nghĩa các cụm
        print(f"\n💡 Ý NGHĨA CÁC CỤM THỊ TRƯỜNG:")
        cluster_meanings = {
            0: "Thị trường tăng mạnh (Bull Market)",
            1: "Thị trường giảm mạnh (Bear Market)", 
            2: "Thị trường ổn định (Stable Market)",
            3: "Thị trường biến động cao (Volatile Market)",
            4: "Thị trường phục hồi (Recovery Market)"
        }
        
        for i in range(min(training_results['optimal_clusters'], 5)):
            meaning = cluster_meanings.get(i, f"Chế độ thị trường {i}")
            print(f"   📊 Cụm {i}: {meaning}")
        
        print(f"\n🚀 ỨNG DỤNG THỰC TẾ:")
        print(f"   📈 Xác định chế độ thị trường hiện tại")
        print(f"   🎯 Điều chỉnh chiến lược giao dịch theo từng cụm")
        print(f"   ⚠️ Phát hiện sự thay đổi chế độ thị trường")
        print(f"   💼 Quản lý rủi ro dựa trên phân cụm")
        
        return {
            'mo_hinh': model,
            'ket_qua_train': training_results,
            'cum_du_doan': test_clusters
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong ví dụ phân cụm: {e}")
        return None

def vi_du_5_so_sanh_mo_hinh():
    """Ví dụ 5: So sánh tất cả mô hình để chọn mô hình tối ưu"""
    in_tieu_de_vi_du("SO SÁNH TẤT CẢ MÔ HÌNH - CHỌN MÔ HÌNH TỐI ƯU")
    
    try:
        # 1. Tải dữ liệu
        print("📊 Đang tải dữ liệu...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        
        print("\n🎯 Huấn luyện và đánh giá tất cả mô hình...")
        
        ket_qua_so_sanh = {}
        
        # Huấn luyện Linear Regression
        print("\n   📊 Đang huấn luyện Hồi quy tuyến tính...")
        lr_model = LinearRegressionModel(target_type='price')
        lr_results = lr_model.train(datasets)
        lr_metrics = lay_chi_so_hieu_suat(lr_results, 'regression')
        ket_qua_so_sanh['Hồi quy tuyến tính'] = {
            'mô hình': lr_model,
            'r2': lr_metrics['r2'],
            'mae': lr_metrics['mae'],
            'rmse': lr_metrics['rmse'],
            'loại': 'Hồi quy',
            'ưu điểm': 'Nhanh, đơn giản, dễ hiểu',
            'nhược điểm': 'Chỉ xử lý quan hệ tuyến tính'
        }
        
        # Huấn luyện KNN Regressor
        print("   📊 Đang huấn luyện Hồi quy KNN...")
        knn_reg = KNNRegressor(n_neighbors=5, target_type='price', auto_tune=False)
        knn_reg_results = knn_reg.train(datasets)
        knn_reg_metrics = lay_chi_so_hieu_suat(knn_reg_results, 'regression')
        ket_qua_so_sanh['Hồi quy KNN'] = {
            'mô hình': knn_reg,
            'r2': knn_reg_metrics['r2'],
            'mae': knn_reg_metrics['mae'],
            'rmse': knn_reg_metrics['rmse'],
            'loại': 'Hồi quy',
            'ưu điểm': 'Xử lý quan hệ phi tuyến',
            'nhược điểm': 'Chậm, cần nhiều bộ nhớ'
        }
        
        # Huấn luyện KNN Classifier
        print("   📊 Đang huấn luyện Phân loại KNN...")
        knn_clf = KNNClassifier(n_neighbors=5, auto_tune=False)
        knn_clf_results = knn_clf.train(datasets)
        knn_clf_metrics = lay_chi_so_hieu_suat(knn_clf_results, 'classification')
        ket_qua_so_sanh['Phân loại KNN'] = {
            'mô hình': knn_clf,
            'accuracy': knn_clf_metrics['accuracy'],
            'precision': knn_clf_metrics['precision_macro'],
            'recall': knn_clf_metrics['recall_macro'],
            'f1': knn_clf_metrics['f1_macro'],
            'loại': 'Phân loại',
            'ưu điểm': 'Dự đoán xu hướng tốt',
            'nhược điểm': 'Phụ thuộc vào chất lượng dữ liệu'
        }
        
        # Huấn luyện K-Means
        print("   📊 Đang huấn luyện Phân cụm K-Means...")
        kmeans = KMeansClusteringModel(auto_tune=True)
        kmeans_results = kmeans.train(datasets)
        ket_qua_so_sanh['Phân cụm K-Means'] = {
            'mô hình': kmeans,
            'silhouette_score': kmeans_results['silhouette_score'],
            'optimal_clusters': kmeans_results['optimal_clusters'],
            'loại': 'Phân cụm',
            'ưu điểm': 'Phát hiện pattern ẩn',
            'nhược điểm': 'Cần biết trước số cụm'
        }
        
        # In bảng so sánh
        print(f"\n📋 BẢNG SO SÁNH HIỆU SUẤT:")
        print(f"{'='*80}")
        print(f"{'Mô hình':<20} {'Loại':<12} {'R²/Acc':<8} {'MAE/F1':<8} {'Ưu điểm':<25}")
        print(f"{'='*80}")
        
        for ten_mo_hinh, info in ket_qua_so_sanh.items():
            if info['loại'] == 'Hồi quy':
                metric1 = f"{info['r2']:.3f}"
                metric2 = f"{info['mae']:.1f}"
            elif info['loại'] == 'Phân loại':
                metric1 = f"{info['accuracy']:.3f}"
                metric2 = f"{info['f1']:.3f}"
            else:  # Phân cụm
                metric1 = f"{info['silhouette_score']:.3f}"
                metric2 = f"{info['optimal_clusters']}"
            
            print(f"{ten_mo_hinh:<20} {info['loại']:<12} {metric1:<8} {metric2:<8} {info['ưu điểm'][:24]:<25}")
        
        # Đưa ra khuyến nghị
        print(f"\n🎯 KHUYẾN NGHỊ SỬ DỤNG:")
        
        # Tìm mô hình hồi quy tốt nhất
        mo_hinh_hoi_quy = {k: v for k, v in ket_qua_so_sanh.items() if v['loại'] == 'Hồi quy'}
        if mo_hinh_hoi_quy:
            best_regression = max(mo_hinh_hoi_quy.items(), key=lambda x: x[1]['r2'])
            print(f"   🏆 Dự đoán giá tốt nhất: {best_regression[0]} (R² = {best_regression[1]['r2']:.3f})")
        
        # Khuyến nghị cho phân loại
        if 'Phân loại KNN' in ket_qua_so_sanh:
            clf_acc = ket_qua_so_sanh['Phân loại KNN']['accuracy']
            print(f"   🎯 Dự đoán xu hướng: Phân loại KNN (Accuracy = {clf_acc:.3f})")
        
        # Khuyến nghị cho phân cụm
        if 'Phân cụm K-Means' in ket_qua_so_sanh:
            kmeans_info = ket_qua_so_sanh['Phân cụm K-Means']
            print(f"   📊 Phân tích thị trường: K-Means ({kmeans_info['optimal_clusters']} cụm)")
        
        print(f"\n💡 CHIẾN LƯỢC TỐI ƯU:")
        print(f"   🔄 Kết hợp nhiều mô hình (Ensemble)")
        print(f"   📈 Sử dụng Linear Regression cho dự đoán nhanh")
        print(f"   🎯 Sử dụng KNN cho pattern phức tạp")
        print(f"   📊 Sử dụng K-Means để hiểu thị trường")
        print(f"   ⚖️ Cân bằng giữa độ chính xác và tốc độ")
        
        return ket_qua_so_sanh
        
    except Exception as e:
        print(f"❌ Lỗi trong so sánh mô hình: {e}")
        return None

def vi_du_6_dich_vu_du_doan():
    """Ví dụ 6: Dịch vụ dự đoán đơn giản mô phỏng thực tế"""
    in_tieu_de_vi_du("DỊCH VỤ DỰ ĐOÁN ĐƠN GIẢN - MÔ PHỎNG THỰC TẾ")
    
    class DichVuDuDoan:
        """Dịch vụ dự đoán crypto đơn giản"""
        
        def __init__(self):
            self.mo_hinh_gia = None
            self.mo_hinh_xu_huong = None
            self.mo_hinh_cum = None
            self.da_huan_luyen = False
        
        def huan_luyen_tat_ca_mo_hinh(self, datasets):
            """Huấn luyện tất cả mô hình"""
            print("🎯 Đang huấn luyện tất cả mô hình...")
            
            # Mô hình dự đoán giá
            self.mo_hinh_gia = LinearRegressionModel(target_type='price')
            self.mo_hinh_gia.train(datasets)
            
            # Mô hình dự đoán xu hướng
            self.mo_hinh_xu_huong = KNNClassifier(n_neighbors=5)
            self.mo_hinh_xu_huong.train(datasets)
            
            # Mô hình phân cụm thị trường
            self.mo_hinh_cum = KMeansClusteringModel(auto_tune=True)
            self.mo_hinh_cum.train(datasets)
            
            self.da_huan_luyen = True
            print("✅ Tất cả mô hình đã được huấn luyện!")
        
        def du_doan_toan_dien(self, du_lieu_dau_vao):
            """Thực hiện dự đoán toàn diện"""
            if not self.da_huan_luyen:
                raise ValueError("❌ Mô hình chưa được huấn luyện!")
            
            # Dự đoán giá
            gia_du_doan = self.mo_hinh_gia.predict(du_lieu_dau_vao)[0]
            
            # Dự đoán xu hướng
            xu_huong_du_doan = self.mo_hinh_xu_huong.predict(du_lieu_dau_vao)[0]
            xac_suat_xu_huong = self.mo_hinh_xu_huong.predict_proba(du_lieu_dau_vao)[0]
            
            # Phân cụm thị trường
            cum_thi_truong = self.mo_hinh_cum.predict(du_lieu_dau_vao)[0]
            
            return {
                'gia_du_doan': gia_du_doan,
                'xu_huong': xu_huong_du_doan,
                'xac_suat_xu_huong': dict(zip(self.mo_hinh_xu_huong.model.classes_, xac_suat_xu_huong)),
                'cum_thi_truong': cum_thi_truong,
                'thoi_gian': datetime.now()
            }
        
        def tao_bao_cao(self, ket_qua_du_doan, gia_hien_tai=None):
            """Tạo báo cáo dự đoán dễ hiểu"""
            bao_cao = []
            bao_cao.append("📊 BÁO CÁO DỰ ĐOÁN CRYPTO")
            bao_cao.append("=" * 40)
            
            # Thông tin giá
            bao_cao.append(f"💰 Giá dự đoán: ${ket_qua_du_doan['gia_du_doan']:,.2f}")
            if gia_hien_tai:
                thay_doi = ((ket_qua_du_doan['gia_du_doan'] - gia_hien_tai) / gia_hien_tai) * 100
                huong = "📈" if thay_doi > 0 else "📉" if thay_doi < 0 else "➡️"
                bao_cao.append(f"📈 Thay đổi dự kiến: {huong} {thay_doi:+.2f}%")
            
            # Xu hướng
            xu_huong = ket_qua_du_doan['xu_huong']
            xac_suat = max(ket_qua_du_doan['xac_suat_xu_huong'].values())
            bao_cao.append(f"🎯 Xu hướng: {xu_huong} ({xac_suat:.1%} tin cậy)")
            
            # Phân tích cụm
            cum = ket_qua_du_doan['cum_thi_truong']
            mo_ta_cum = {
                0: "Thị trường tăng mạnh",
                1: "Thị trường giảm mạnh", 
                2: "Thị trường ổn định",
                3: "Thị trường biến động cao"
            }
            bao_cao.append(f"📊 Chế độ thị trường: {mo_ta_cum.get(cum, f'Cụm {cum}')}")
            
            # Khuyến nghị
            bao_cao.append("\n💡 KHUYẾN NGHỊ:")
            if xu_huong == 'Bullish' and xac_suat > 0.7:
                bao_cao.append("   🟢 CÂN NHẮC MUA - Tín hiệu tích cực mạnh")
            elif xu_huong == 'Bearish' and xac_suat > 0.7:
                bao_cao.append("   🔴 CÂN NHẮC BÁN - Tín hiệu tiêu cực mạnh")
            else:
                bao_cao.append("   🟡 GIỮ VỊ THẾ - Tín hiệu không rõ ràng")
            
            bao_cao.append("⚠️ Đây chỉ là dự đoán, không phải tư vấn đầu tư!")
            
            return "\n".join(bao_cao)
    
    try:
        # 1. Khởi tạo dịch vụ
        print("🚀 Tạo dịch vụ dự đoán mô phỏng...")
        dich_vu = DichVuDuDoan()
        
        # 2. Tải dữ liệu và huấn luyện
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        datasets = chuyen_doi_dinh_dang_datasets(raw_datasets)
        dich_vu.huan_luyen_tat_ca_mo_hinh(datasets)
        
        # 3. Thực hiện dự đoán mẫu
        print("\n🔮 Thực hiện dự đoán mẫu...")
        du_lieu_test = datasets['raw']['X_test_raw'].head(1)
        ket_qua = dich_vu.du_doan_toan_dien(du_lieu_test)
        
        # Lấy giá thực tế để so sánh
        gia_thuc_te = datasets['raw']['y_test']['price'].head(1).values[0]
        
        # 4. Hiển thị báo cáo
        print("\n📋 BÁO CÁO DỰ ĐOÁN:")
        bao_cao = dich_vu.tao_bao_cao(ket_qua, gia_thuc_te)
        print(bao_cao)
        
        # 5. So sánh với thực tế
        print(f"\n✅ KIỂM CHỨNG:")
        print(f"   💰 Giá thực tế: ${gia_thuc_te:,.2f}")
        print(f"   🎯 Giá dự đoán: ${ket_qua['gia_du_doan']:,.2f}")
        sai_so = abs(ket_qua['gia_du_doan'] - gia_thuc_te)
        sai_so_phan_tram = (sai_so / gia_thuc_te) * 100
        print(f"   📊 Sai số: ${sai_so:.2f} ({sai_so_phan_tram:.1f}%)")
        
        print(f"\n🚀 ỨNG DỤNG THỰC TẾ:")
        print(f"   🌐 Tích hợp vào website/app")
        print(f"   📱 API endpoint cho mobile")
        print(f"   📊 Dashboard real-time")
        print(f"   🔔 Cảnh báo qua email/SMS")
        print(f"   📈 Báo cáo định kỳ")
        
        return {
            'dich_vu': dich_vu,
            'ket_qua_du_doan': ket_qua,
            'bao_cao': bao_cao
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong dịch vụ dự đoán: {e}")
        return None

def main():
    """Chạy tất cả ví dụ cơ bản tiếng Việt"""
    print("🎯 VÍ DỤ MACHINE LEARNING CRYPTO - TIẾNG VIỆT")
    print("=" * 60)
    print("Tập hợp ví dụ minh họa cách sử dụng các thuật toán ML")
    print("trong dự án dự đoán giá cryptocurrency bằng tiếng Việt.")
    print("")
    print("🎓 MỤC TIÊU HỌC TẬP:")
    print("   📚 Hiểu rõ cách hoạt động của từng thuật toán")
    print("   🔍 So sánh hiệu suất các mô hình khác nhau")
    print("   💡 Biết cách chọn mô hình phù hợp")
    print("   🚀 Ứng dụng vào thực tế giao dịch crypto")
    
    cac_vi_du = [
        ("Hồi quy tuyến tính", vi_du_1_hoi_quy_tuyen_tinh),
        ("Phân loại KNN", vi_du_2_phan_loai_knn),
        ("Hồi quy KNN", vi_du_3_hoi_quy_knn),
        ("Phân cụm K-Means", vi_du_4_phan_cum_kmeans),
        ("So sánh mô hình", vi_du_5_so_sanh_mo_hinh),
        ("Dịch vụ dự đoán", vi_du_6_dich_vu_du_doan)
    ]
    
    ket_qua = {}
    thanh_cong = 0
    that_bai = 0
    
    for i, (ten_vi_du, ham_vi_du) in enumerate(cac_vi_du, 1):
        try:
            print(f"\n🔄 Đang chạy Ví dụ {i}: {ten_vi_du}...")
            result = ham_vi_du()
            ket_qua[ham_vi_du.__name__] = result
            
            if result is not None:
                thanh_cong += 1
                print(f"✅ Ví dụ {i} hoàn thành thành công!")
            else:
                that_bai += 1
                print(f"⚠️ Ví dụ {i} có vấn đề, xem chi tiết ở trên.")
                
        except Exception as e:
            that_bai += 1
            print(f"❌ Ví dụ {i} thất bại: {e}")
            ket_qua[ham_vi_du.__name__] = None
    
    # Tổng kết
    tong_so = len(cac_vi_du)
    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT CÁC VÍ DỤ")
    print(f"{'='*60}")
    print(f"✅ Thành công: {thanh_cong}/{tong_so} ví dụ")
    print(f"❌ Thất bại: {that_bai}/{tong_so} ví dụ")
    
    if thanh_cong == tong_so:
        print("\n🎉 TẤT CẢ VÍ DỤ HOÀN THÀNH THÀNH CÔNG!")
        print("🚀 Bạn đã nắm được các kiến thức cơ bản về ML trong crypto!")
        
        print(f"\n📚 KIẾN THỨC ĐÃ HỌC:")
        print(f"   🔹 Hồi quy tuyến tính cho dự đoán giá")
        print(f"   🔹 KNN cho phân loại xu hướng và hồi quy phi tuyến")
        print(f"   🔹 K-Means cho phân tích chế độ thị trường")
        print(f"   🔹 So sánh và lựa chọn mô hình phù hợp")
        print(f"   🔹 Xây dựng dịch vụ dự đoán hoàn chỉnh")
        
        print(f"\n🎯 BƯỚC TIẾP THEO:")
        print(f"   📈 Chạy advanced_pipeline.py cho kỹ thuật nâng cao")
        print(f"   🤖 Chạy production_examples.py cho ứng dụng thực tế")
        print(f"   🌐 Chạy web_dashboard.py cho giao diện web")
        print(f"   📚 Đọc integration_guide.py để hiểu sâu thuật toán")
        
    else:
        print(f"\n⚠️ MỘT SỐ VÍ DỤ GẶP VẤN ĐỀ")
        print("🔍 Kiểm tra thông báo lỗi ở trên để khắc phục.")
        print("📚 Đảm bảo dữ liệu đã được chuẩn bị đúng cách.")
    
    print(f"\n💡 GHI NHỚ:")
    print(f"   📊 Mỗi thuật toán có ưu nhược điểm riêng")
    print(f"   🎯 Chọn mô hình phù hợp với dữ liệu và mục tiêu")
    print(f"   🔄 Luôn kiểm tra và đánh giá kết quả")
    print(f"   ⚠️ Không đầu tư dựa hoàn toàn vào dự đoán ML")
    
    return ket_qua

if __name__ == "__main__":
    main()