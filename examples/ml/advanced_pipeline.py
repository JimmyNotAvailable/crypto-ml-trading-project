#!/usr/bin/env python3
"""
🚀 PIPELINE ML NÂNG CAO - TIẾNG VIỆT
===================================

Các ví dụ nâng cao minh họa cách xây dựng pipeline ML production-ready
cho dự đoán crypto với các tính năng nâng cao, tất cả bằng tiếng Việt.

Phù hợp cho việc tích hợp vào web interface và hiểu rõ cách hoạt động của ML.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc dự án vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
from app.ml.model_registry import ModelRegistry

class PipelineMLNangCao:
    """Pipeline ML nâng cao cho dự đoán crypto"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.mo_hinh = {}
        self.ket_qua = {}
    
    def chay_phan_tich_tong_hop(self, ten_dataset='ml_datasets_top3'):
        """Chạy pipeline phân tích ML toàn diện"""
        print("🚀 PIPELINE ML NÂNG CAO - PHÂN TÍCH TOÀN DIỆN")
        print("=" * 60)
        
        # 1. Tải và kiểm tra dữ liệu
        print("\n📊 Bước 1: Tải và Kiểm tra Dữ liệu")
        datasets = self._tai_va_kiem_tra_du_lieu(ten_dataset)
        
        # 2. Lựa chọn mô hình tự động
        print("\n🤖 Bước 2: Lựa chọn Mô hình Tự động")
        mo_hinh_tot_nhat = self._chon_mo_hinh_tu_dong(datasets)
        
        # 3. Tối ưu hóa siêu tham số
        print("\n⚙️ Bước 3: Tối ưu hóa Siêu tham số")
        mo_hinh_toi_uu = self._toi_uu_sieu_tham_so(mo_hinh_tot_nhat, datasets)
        
        # 4. Học tập Ensemble
        print("\n🎯 Bước 4: Học tập Ensemble (Kết hợp mô hình)")
        ket_qua_ensemble = self._tao_ensemble(mo_hinh_toi_uu, datasets)
        
        # 5. Phân tích hiệu suất
        print("\n📈 Bước 5: Phân tích Hiệu suất Chi tiết")
        phan_tich_hieu_suat = self._phan_tich_hieu_suat(datasets)
        
        # 6. Quản lý phiên bản mô hình
        print("\n💾 Bước 6: Quản lý Phiên bản Mô hình")
        ket_qua_registry = self._quan_ly_phien_ban_mo_hinh()
        
        # 7. Mô phỏng triển khai production
        print("\n🚀 Bước 7: Mô phỏng Triển khai Production")
        ket_qua_trien_khai = self._mo_phong_trien_khai_production(datasets)
        
        return {
            'datasets': datasets,
            'mo_hinh_tot_nhat': mo_hinh_tot_nhat,
            'mo_hinh_toi_uu': mo_hinh_toi_uu,
            'ket_qua_ensemble': ket_qua_ensemble,
            'phan_tich_hieu_suat': phan_tich_hieu_suat,
            'ket_qua_registry': ket_qua_registry,
            'ket_qua_trien_khai': ket_qua_trien_khai
        }
    
    def _tai_va_kiem_tra_du_lieu(self, ten_dataset):
        """Tải và kiểm tra chất lượng dữ liệu"""
        try:
            datasets = load_prepared_datasets(ten_dataset)
            
            print(f"✅ Đã tải dataset: {ten_dataset}")
            print(f"   📊 Mẫu huấn luyện: {len(datasets['X_train'])} mẫu")
            print(f"   📊 Mẫu kiểm tra: {len(datasets['X_test'])} mẫu")
            print(f"   📊 Số đặc trưng: {datasets['X_train'].shape[1]}")
            
            # Kiểm tra chất lượng dữ liệu
            print("\n🔍 Kiểm tra Chất lượng Dữ liệu:")
            
            # Giá trị thiếu
            missing_train = datasets['X_train'].isnull().sum().sum()
            missing_test = datasets['X_test'].isnull().sum().sum()
            print(f"   ❌ Giá trị thiếu - Huấn luyện: {missing_train}, Kiểm tra: {missing_test}")
            
            if missing_train == 0 and missing_test == 0:
                print(f"   ✅ Dữ liệu sạch - Không có giá trị thiếu!")
            else:
                print(f"   ⚠️ Cần xử lý giá trị thiếu trước khi tiếp tục")
            
            # Tương quan đặc trưng
            high_corr_features = self._tim_dac_trung_tuong_quan_cao(datasets['X_train'])
            print(f"   🔗 Cặp đặc trưng tương quan cao (>95%): {len(high_corr_features)}")
            
            if len(high_corr_features) > 0:
                print(f"   ⚠️ Có thể cần loại bỏ đặc trưng dư thừa")
                for feature1, feature2, corr in high_corr_features[:3]:  # Hiển thị 3 cặp đầu
                    print(f"      {feature1} ↔ {feature2}: {corr:.3f}")
            else:
                print(f"   ✅ Không có đa cộng tuyến nghiêm trọng")
            
            # Phân bố dữ liệu
            price_stats = datasets['y_train']['price'].describe()
            print(f"   💰 Phạm vi giá: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
            print(f"   💰 Giá trung bình: ${price_stats['mean']:.2f} ± ${price_stats['std']:.2f}")
            
            # Đánh giá độ ổn định dữ liệu
            cv = price_stats['std'] / price_stats['mean']  # Coefficient of variation
            if cv < 0.1:
                print(f"   📊 Dữ liệu ổn định (CV: {cv:.3f})")
            elif cv < 0.3:
                print(f"   📊 Dữ liệu biến động vừa (CV: {cv:.3f})")
            else:
                print(f"   📊 Dữ liệu biến động cao (CV: {cv:.3f})")
            
            return datasets
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            raise
    
    def _tim_dac_trung_tuong_quan_cao(self, X, nguong=0.95):
        """Tìm các cặp đặc trưng có tương quan cao"""
        corr_matrix = X.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > nguong:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return high_corr_pairs
    
    def _chon_mo_hinh_tu_dong(self, datasets):
        """Chạy lựa chọn mô hình tự động"""
        try:
            print("🔄 Đang kiểm tra các mô hình khác nhau...")
            
            mo_hinh_can_test = {
                'hoi_quy_tuyen_tinh': LinearRegressionModel(target_type='price'),
                'hoi_quy_knn': KNNRegressor(target_type='price', auto_tune=False),
                'phan_loai_knn': KNNClassifier(auto_tune=False)
            }
            
            ket_qua = {}
            
            for ten, mo_hinh in mo_hinh_can_test.items():
                print(f"   🧪 Đang test {ten}...")
                try:
                    ket_qua_huan_luyen = mo_hinh.train(datasets)
                    ket_qua[ten] = ket_qua_huan_luyen
                    
                    # In kết quả ngay
                    if 'r2' in ket_qua_huan_luyen['test_metrics']:
                        r2 = ket_qua_huan_luyen['test_metrics']['r2']
                        mae = ket_qua_huan_luyen['test_metrics']['mae']
                        print(f"      ✅ R²: {r2:.4f}, MAE: ${mae:.2f}")
                    elif 'accuracy' in ket_qua_huan_luyen['test_metrics']:
                        acc = ket_qua_huan_luyen['test_metrics']['accuracy']
                        prec = ket_qua_huan_luyen['test_metrics']['precision']
                        print(f"      ✅ Accuracy: {acc:.3f}, Precision: {prec:.3f}")
                        
                except Exception as e:
                    print(f"      ❌ Thất bại: {e}")
                    continue
            
            print(f"✅ Đã test {len(ket_qua)} mô hình")
            
            # Tìm mô hình tốt nhất cho từng loại
            mo_hinh_tot_nhat = {}
            
            # Mô hình hồi quy tốt nhất
            mo_hinh_hoi_quy = {k: v for k, v in ket_qua.items() 
                              if any(x in k.lower() for x in ['hoi_quy', 'regression'])}
            if mo_hinh_hoi_quy:
                mo_hinh_hoi_quy_tot_nhat = max(mo_hinh_hoi_quy.keys(), 
                                              key=lambda x: mo_hinh_hoi_quy[x]['test_metrics']['r2'])
                mo_hinh_tot_nhat['hoi_quy'] = {
                    'ten': mo_hinh_hoi_quy_tot_nhat,
                    'chi_so': mo_hinh_hoi_quy[mo_hinh_hoi_quy_tot_nhat]['test_metrics']
                }
                r2_score = mo_hinh_hoi_quy[mo_hinh_hoi_quy_tot_nhat]['test_metrics']['r2']
                print(f"   🏆 Hồi quy tốt nhất: {mo_hinh_hoi_quy_tot_nhat} (R² = {r2_score:.4f})")
            
            # Mô hình phân loại tốt nhất
            mo_hinh_phan_loai = {k: v for k, v in ket_qua.items() 
                                if 'phan_loai' in k.lower() or 'classifier' in k.lower()}
            if mo_hinh_phan_loai:
                mo_hinh_phan_loai_tot_nhat = max(mo_hinh_phan_loai.keys(), 
                                               key=lambda x: mo_hinh_phan_loai[x]['test_metrics']['accuracy'])
                mo_hinh_tot_nhat['phan_loai'] = {
                    'ten': mo_hinh_phan_loai_tot_nhat,
                    'chi_so': mo_hinh_phan_loai[mo_hinh_phan_loai_tot_nhat]['test_metrics']
                }
                acc_score = mo_hinh_phan_loai[mo_hinh_phan_loai_tot_nhat]['test_metrics']['accuracy']
                print(f"   🏆 Phân loại tốt nhất: {mo_hinh_phan_loai_tot_nhat} (Accuracy = {acc_score:.3f})")
            
            self.ket_qua['lua_chon_tu_dong'] = ket_qua
            return mo_hinh_tot_nhat
            
        except Exception as e:
            print(f"❌ Lỗi trong lựa chọn tự động: {e}")
            return {}
    
    def _toi_uu_sieu_tham_so(self, mo_hinh_tot_nhat, datasets):
        """Tối ưu hóa siêu tham số cho các mô hình tốt nhất"""
        mo_hinh_toi_uu = {}
        
        try:
            print("\n🔧 Tối ưu hóa Siêu tham số KNN:")
            
            # Tối ưu KNN Regressor
            print("   📊 Tối ưu hóa KNN Hồi quy...")
            knn_regressor = KNNRegressor(target_type='price', auto_tune=True)
            ket_qua_knn = knn_regressor.train(datasets)
            
            mo_hinh_toi_uu['knn_hoi_quy'] = {
                'mo_hinh': knn_regressor,
                'ket_qua': ket_qua_knn,
                'tham_so_tot_nhat': getattr(knn_regressor, 'best_params', None)
            }
            
            print(f"   ✅ KNN Hồi quy đã tối ưu:")
            print(f"      🎯 K tối ưu: {knn_regressor.n_neighbors}")
            print(f"      📊 Điểm R²: {ket_qua_knn['test_metrics']['r2']:.4f}")
            print(f"      💰 MAE: ${ket_qua_knn['test_metrics']['mae']:.2f}")
            
            # Giải thích tại sao chọn K này
            k_value = knn_regressor.n_neighbors
            if k_value <= 3:
                print(f"      💡 K nhỏ ({k_value}) → Nhạy cảm với dữ liệu địa phương")
            elif k_value <= 7:
                print(f"      💡 K vừa ({k_value}) → Cân bằng giữa bias và variance")
            else:
                print(f"      💡 K lớn ({k_value}) → Mô hình mượt, ít overfitting")
            
            # Tối ưu KNN Classifier
            print("   🎯 Tối ưu hóa KNN Phân loại...")
            knn_classifier = KNNClassifier(auto_tune=True)
            ket_qua_clf = knn_classifier.train(datasets)
            
            mo_hinh_toi_uu['knn_phan_loai'] = {
                'mo_hinh': knn_classifier,
                'ket_qua': ket_qua_clf,
                'tham_so_tot_nhat': getattr(knn_classifier, 'best_params', None)
            }
            
            print(f"   ✅ KNN Phân loại đã tối ưu:")
            print(f"      🎯 K tối ưu: {knn_classifier.n_neighbors}")
            print(f"      📊 Accuracy: {ket_qua_clf['test_metrics']['accuracy']:.3f}")
            print(f"      📈 Precision: {ket_qua_clf['test_metrics']['precision']:.3f}")
            
            # Tối ưu K-Means
            print("   🎯 Tối ưu hóa K-Means Clustering...")
            kmeans = KMeansClusteringModel(auto_tune=True, max_clusters=10)
            ket_qua_cluster = kmeans.train(datasets)
            
            mo_hinh_toi_uu['kmeans'] = {
                'mo_hinh': kmeans,
                'ket_qua': ket_qua_cluster
            }
            
            print(f"   ✅ K-Means đã tối ưu:")
            print(f"      🎯 Số cụm tối ưu: {ket_qua_cluster['n_clusters']}")
            print(f"      📊 Điểm Silhouette: {ket_qua_cluster['silhouette_score']:.3f}")
            
            print(f"\n💡 HIỂU VỀ TỐI ƯU HÓA SIÊU THAM SỐ:")
            print(f"   🔹 Auto-tuning giúp tìm tham số tốt nhất tự động")
            print(f"   🔹 Tránh overfitting và underfitting")
            print(f"   🔹 Cải thiện hiệu suất đáng kể so với tham số mặc định")
            print(f"   🔹 Tiết kiệm thời gian thử nghiệm thủ công")
            
            return mo_hinh_toi_uu
            
        except Exception as e:
            print(f"❌ Lỗi trong tối ưu hóa siêu tham số: {e}")
            return {}
    
    def _tao_ensemble(self, mo_hinh_toi_uu, datasets):
        """Tạo ensemble (kết hợp) dự đoán từ nhiều mô hình"""
        try:
            print("\n🎯 Tạo Ensemble - Kết hợp Dự đoán:")
            
            print(f"\n📚 ENSEMBLE LÀ GÌ?")
            print(f"   🔹 Kết hợp nhiều mô hình để tăng độ chính xác")
            print(f"   🔹 Giảm rủi ro từ một mô hình đơn lẻ")
            print(f"   🔹 Như xin ý kiến nhiều chuyên gia rồi tổng hợp")
            
            # Thu thập các mô hình hồi quy
            mo_hinh_hoi_quy = []
            ten_mo_hinh = []
            
            # Thêm Linear Regression
            linear_model = LinearRegressionModel(target_type='price')
            ket_qua_linear = linear_model.train(datasets)
            mo_hinh_hoi_quy.append(linear_model)
            ten_mo_hinh.append('Hồi quy tuyến tính')
            
            # Thêm KNN Regressor đã tối ưu
            if 'knn_hoi_quy' in mo_hinh_toi_uu:
                mo_hinh_hoi_quy.append(mo_hinh_toi_uu['knn_hoi_quy']['mo_hinh'])
                ten_mo_hinh.append('KNN Hồi quy')
            
            if len(mo_hinh_hoi_quy) < 2:
                print("   ⚠️ Cần ít nhất 2 mô hình để tạo ensemble")
                return {}
            
            print(f"   🔗 Đang kết hợp {len(ten_mo_hinh)} mô hình: {', '.join(ten_mo_hinh)}")
            
            # Thực hiện dự đoán từ tất cả mô hình
            test_data = datasets['X_test']
            cac_du_doan = []
            
            for mo_hinh in mo_hinh_hoi_quy:
                du_doan = mo_hinh.predict(test_data)
                cac_du_doan.append(du_doan)
            
            # Ensemble đơn giản - trung bình
            ensemble_don_gian = np.mean(cac_du_doan, axis=0)
            
            # Ensemble có trọng số - dựa trên điểm R²
            trong_so = []
            for mo_hinh in mo_hinh_hoi_quy:
                if hasattr(mo_hinh, 'training_history') and mo_hinh.training_history:
                    r2 = mo_hinh.training_history['test_metrics']['r2']
                    trong_so.append(r2)
                else:
                    trong_so.append(0.5)  # Trọng số mặc định
            
            trong_so = np.array(trong_so) / np.sum(trong_so)  # Chuẩn hóa
            ensemble_co_trong_so = np.average(cac_du_doan, axis=0, weights=trong_so)
            
            # Đánh giá hiệu suất ensemble
            gia_thuc_te = datasets['y_test']['price'].values
            
            # Ensemble đơn giản
            mae_don_gian = np.mean(np.abs(ensemble_don_gian - gia_thuc_te))
            r2_don_gian = 1 - np.sum((gia_thuc_te - ensemble_don_gian)**2) / np.sum((gia_thuc_te - np.mean(gia_thuc_te))**2)
            
            # Ensemble có trọng số
            mae_co_trong_so = np.mean(np.abs(ensemble_co_trong_so - gia_thuc_te))
            r2_co_trong_so = 1 - np.sum((gia_thuc_te - ensemble_co_trong_so)**2) / np.sum((gia_thuc_te - np.mean(gia_thuc_te))**2)
            
            ket_qua_ensemble = {
                'cac_mo_hinh': ten_mo_hinh,
                'trong_so': trong_so,
                'ensemble_don_gian': {
                    'du_doan': ensemble_don_gian,
                    'mae': mae_don_gian,
                    'r2': r2_don_gian
                },
                'ensemble_co_trong_so': {
                    'du_doan': ensemble_co_trong_so,
                    'mae': mae_co_trong_so,
                    'r2': r2_co_trong_so
                }
            }
            
            print(f"   ✅ Ensemble đã được tạo với {len(ten_mo_hinh)} mô hình:")
            for ten, trong_so_mo_hinh in zip(ten_mo_hinh, trong_so):
                print(f"      {ten}: {trong_so_mo_hinh:.3f}")
            
            print(f"\n   📊 Hiệu suất Ensemble:")
            print(f"      Trung bình đơn giản - R²: {r2_don_gian:.4f}, MAE: ${mae_don_gian:.2f}")
            print(f"      Trung bình có trọng số - R²: {r2_co_trong_so:.4f}, MAE: ${mae_co_trong_so:.2f}")
            
            # So sánh với từng mô hình riêng lẻ
            print(f"\n   ⚖️ So sánh với Mô hình Riêng lẻ:")
            for i, (mo_hinh, ten) in enumerate(zip(mo_hinh_hoi_quy, ten_mo_hinh)):
                mae_rieng_le = np.mean(np.abs(cac_du_doan[i] - gia_thuc_te))
                r2_rieng_le = 1 - np.sum((gia_thuc_te - cac_du_doan[i])**2) / np.sum((gia_thuc_te - np.mean(gia_thuc_te))**2)
                print(f"      {ten} - R²: {r2_rieng_le:.4f}, MAE: ${mae_rieng_le:.2f}")
            
            # Kết luận
            if r2_co_trong_so > max(r2_don_gian, max([1 - np.sum((gia_thuc_te - pred)**2) / np.sum((gia_thuc_te - np.mean(gia_thuc_te))**2) for pred in cac_du_doan])):
                print(f"\n   🏆 ENSEMBLE CÓ TRỌNG SỐ THẮNG! - Cải thiện hiệu suất tổng thể")
            else:
                print(f"\n   📊 Một số mô hình riêng lẻ có thể tốt hơn ensemble")
            
            print(f"\n💡 LỢI ÍCH CỦA ENSEMBLE:")
            print(f"   ✅ Giảm rủi ro overfitting")
            print(f"   ✅ Tăng độ ổn định dự đoán")
            print(f"   ✅ Kết hợp điểm mạnh của nhiều thuật toán")
            print(f"   ✅ Thường có hiệu suất tốt hơn mô hình đơn lẻ")
            
            return ket_qua_ensemble
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo ensemble: {e}")
            return {}
    
    def _phan_tich_hieu_suat(self, datasets):
        """Phân tích hiệu suất chi tiết"""
        try:
            print("\n📈 Phân tích Hiệu suất Chi tiết:")
            
            ket_qua_phan_tich = {}
            
            # 1. Phân tích tầm quan trọng đặc trưng
            print("\n   🔍 Phân tích Tầm quan trọng Đặc trưng:")
            linear_model = LinearRegressionModel(target_type='price')
            linear_model.train(datasets)
            
            tam_quan_trong_dac_trung = linear_model.get_feature_importance()
            top_features = tam_quan_trong_dac_trung.head(10)
            
            print("      🏆 Top 10 Đặc trưng Quan trọng nhất:")
            for _, row in top_features.iterrows():
                print(f"        {row['feature']}: {row['abs_coefficient']:.4f}")
            
            ket_qua_phan_tich['tam_quan_trong_dac_trung'] = tam_quan_trong_dac_trung
            
            # Giải thích ý nghĩa
            print(f"\n      💡 Ý nghĩa:")
            print(f"         🔹 Số càng lớn = đặc trưng càng quan trọng")
            print(f"         🔹 Giúp hiểu yếu tố nào ảnh hưởng giá nhất")
            print(f"         🔹 Có thể loại bỏ đặc trưng ít quan trọng")
            
            # 2. Phân tích lỗi dự đoán
            print("\n   📊 Phân tích Lỗi Dự đoán:")
            du_doan = linear_model.predict(datasets['X_test'])
            thuc_te = datasets['y_test']['price'].values
            loi = du_doan - thuc_te
            
            thong_ke_loi = {
                'loi_trung_binh': np.mean(loi),
                'do_lech_chuan_loi': np.std(loi),
                'loi_lon_nhat': np.max(np.abs(loi)),
                'loi_95_phan_tram': np.percentile(np.abs(loi), 95)
            }
            
            print(f"      📈 Thống kê Lỗi:")
            print(f"         Lỗi trung bình: ${thong_ke_loi['loi_trung_binh']:.2f}")
            print(f"         Độ lệch chuẩn: ${thong_ke_loi['do_lech_chuan_loi']:.2f}")
            print(f"         Lỗi lớn nhất: ${thong_ke_loi['loi_lon_nhat']:.2f}")
            print(f"         Lỗi 95%: ${thong_ke_loi['loi_95_phan_tram']:.2f}")
            
            # Đánh giá chất lượng lỗi
            if abs(thong_ke_loi['loi_trung_binh']) < 10:
                print(f"      ✅ Lỗi trung bình thấp - Mô hình không bias")
            else:
                print(f"      ⚠️ Lỗi trung bình cao - Mô hình có thể bias")
            
            ket_qua_phan_tich['phan_tich_loi'] = thong_ke_loi
            
            # 3. Phân tích chế độ thị trường
            print("\n   🎯 Phân tích Chế độ Thị trường:")
            clustering = KMeansClusteringModel(auto_tune=True)
            ket_qua_cluster = clustering.train(datasets)
            
            phan_tich_che_do = {}
            for cluster_id, info in ket_qua_cluster['cluster_analysis'].items():
                phan_tich_che_do[cluster_id] = {
                    'kich_thuoc': info['size'],
                    'loai_thi_truong': info['market_interpretation'],
                    'loi_du_doan_tb': None  # Sẽ tính toán lỗi theo từng chế độ
                }
            
            print(f"      📊 Đã xác định {len(phan_tich_che_do)} chế độ thị trường:")
            for cluster_id, info in phan_tich_che_do.items():
                print(f"        {cluster_id}: {info['loai_thi_truong']} ({info['kich_thuoc']} mẫu)")
            
            ket_qua_phan_tich['che_do_thi_truong'] = phan_tich_che_do
            
            # 4. Phân tích hiệu suất theo thời gian
            print("\n   ⏰ Phân tích Hiệu suất theo Thời gian:")
            
            # Mô phỏng chia theo thời gian
            kich_thuoc_test = len(datasets['X_test'])
            cac_giai_doan = ['Đầu kỳ', 'Giữa kỳ', 'Cuối kỳ']
            kich_thuoc_giai_doan = kich_thuoc_test // 3
            
            hieu_suat_theo_thoi_gian = {}
            for i, giai_doan in enumerate(cac_giai_doan):
                start_idx = i * kich_thuoc_giai_doan
                end_idx = start_idx + kich_thuoc_giai_doan if i < 2 else kich_thuoc_test
                
                du_doan_giai_doan = du_doan[start_idx:end_idx]
                thuc_te_giai_doan = thuc_te[start_idx:end_idx]
                
                mae_giai_doan = np.mean(np.abs(du_doan_giai_doan - thuc_te_giai_doan))
                r2_giai_doan = 1 - np.sum((thuc_te_giai_doan - du_doan_giai_doan)**2) / np.sum((thuc_te_giai_doan - np.mean(thuc_te_giai_doan))**2)
                
                hieu_suat_theo_thoi_gian[giai_doan] = {
                    'mae': mae_giai_doan,
                    'r2': r2_giai_doan,
                    'so_mau': end_idx - start_idx
                }
                
                print(f"      {giai_doan} - R²: {r2_giai_doan:.4f}, MAE: ${mae_giai_doan:.2f}")
            
            ket_qua_phan_tich['hieu_suat_thoi_gian'] = hieu_suat_theo_thoi_gian
            
            # Đánh giá xu hướng
            r2_scores = [hieu_suat_theo_thoi_gian[gd]['r2'] for gd in cac_giai_doan]
            if r2_scores[-1] > r2_scores[0]:
                print(f"      📈 Xu hướng: Hiệu suất CẢI THIỆN theo thời gian")
            elif r2_scores[-1] < r2_scores[0]:
                print(f"      📉 Xu hướng: Hiệu suất GIẢM theo thời gian")
            else:
                print(f"      📊 Xu hướng: Hiệu suất ỔN ĐỊNH theo thời gian")
            
            print(f"\n💡 TÓM TẮT PHÂN TÍCH:")
            print(f"   📊 Đã phân tích tầm quan trọng {len(top_features)} đặc trưng hàng đầu")
            print(f"   📈 Đã đánh giá chất lượng lỗi dự đoán")
            print(f"   🎯 Đã xác định {len(phan_tich_che_do)} chế độ thị trường")
            print(f"   ⏰ Đã phân tích hiệu suất qua {len(cac_giai_doan)} giai đoạn thời gian")
            
            return ket_qua_phan_tich
            
        except Exception as e:
            print(f"❌ Lỗi trong phân tích hiệu suất: {e}")
            return {}

def vi_du_1_pipeline_toan_dien():
    """Ví dụ 1: Pipeline nâng cao toàn diện"""
    print("🚀 VÍ DỤ 1: PIPELINE NÂNG CAO TOÀN DIỆN")
    print("=" * 60)
    
    pipeline = PipelineMLNangCao()
    ket_qua = pipeline.chay_phan_tich_tong_hop()
    
    return ket_qua

def vi_du_2_ensemble_tuy_chinh():
    """Ví dụ 2: Ensemble tùy chỉnh với các chiến lược khác nhau"""
    print("\n🚀 VÍ DỤ 2: CHIẾN LƯỢC ENSEMBLE TÙY CHỈNH")
    print("=" * 60)
    
    try:
        # Tải dữ liệu
        datasets = load_prepared_datasets('ml_datasets_top3')
        
        # Huấn luyện nhiều mô hình
        cac_mo_hinh = {
            'tuyen_tinh': LinearRegressionModel(target_type='price'),
            'knn': KNNRegressor(target_type='price', n_neighbors=5, auto_tune=False)
        }
        
        cac_du_doan = {}
        diem_r2 = {}
        
        print("📊 Huấn luyện các mô hình riêng lẻ...")
        for ten_mo_hinh, mo_hinh in cac_mo_hinh.items():
            ket_qua = mo_hinh.train(datasets)
            cac_du_doan[ten_mo_hinh] = mo_hinh.predict(datasets['X_test'])
            diem_r2[ten_mo_hinh] = ket_qua['test_metrics']['r2']
            print(f"   ✅ {ten_mo_hinh}: R² = {diem_r2[ten_mo_hinh]:.4f}")
        
        # Các chiến lược ensemble
        thuc_te = datasets['y_test']['price'].values
        
        print(f"\n🎯 CHIẾN LƯỢC ENSEMBLE:")
        
        # 1. Trung bình đơn giản
        tb_don_gian = np.mean(list(cac_du_doan.values()), axis=0)
        mae_tb_don_gian = np.mean(np.abs(tb_don_gian - thuc_te))
        r2_tb_don_gian = 1 - np.sum((thuc_te - tb_don_gian)**2) / np.sum((thuc_te - np.mean(thuc_te))**2)
        
        # 2. Trung bình có trọng số theo R²
        trong_so = np.array(list(diem_r2.values()))
        trong_so = trong_so / np.sum(trong_so)
        tb_co_trong_so = np.average(list(cac_du_doan.values()), axis=0, weights=trong_so)
        mae_co_trong_so = np.mean(np.abs(tb_co_trong_so - thuc_te))
        r2_co_trong_so = 1 - np.sum((thuc_te - tb_co_trong_so)**2) / np.sum((thuc_te - np.mean(thuc_te))**2)
        
        # 3. Chọn mô hình tốt nhất (dynamic)
        ten_mo_hinh_tot_nhat = max(diem_r2.keys(), key=lambda x: diem_r2[x])
        du_doan_tot_nhat = cac_du_doan[ten_mo_hinh_tot_nhat]
        mae_tot_nhat = np.mean(np.abs(du_doan_tot_nhat - thuc_te))
        r2_tot_nhat = diem_r2[ten_mo_hinh_tot_nhat]
        
        print("\n📈 KẾT QUẢ ENSEMBLE:")
        print(f"   Trung bình đơn giản: R² = {r2_tb_don_gian:.4f}, MAE = ${mae_tb_don_gian:.2f}")
        print(f"   Trung bình có trọng số: R² = {r2_co_trong_so:.4f}, MAE = ${mae_co_trong_so:.2f}")
        print(f"   Mô hình tốt nhất ({ten_mo_hinh_tot_nhat}): R² = {r2_tot_nhat:.4f}, MAE = ${mae_tot_nhat:.2f}")
        
        # Tìm chiến lược tốt nhất
        cac_chien_luoc = {
            'Trung bình đơn giản': (r2_tb_don_gian, mae_tb_don_gian),
            'Trung bình có trọng số': (r2_co_trong_so, mae_co_trong_so),
            f'Mô hình tốt nhất ({ten_mo_hinh_tot_nhat})': (r2_tot_nhat, mae_tot_nhat)
        }
        
        chien_luoc_tot_nhat = max(cac_chien_luoc.keys(), key=lambda x: cac_chien_luoc[x][0])
        print(f"\n🏆 CHIẾN LƯỢC TỐT NHẤT: {chien_luoc_tot_nhat}")
        
        print(f"\n💡 NHẬN XÉT VỀ ENSEMBLE:")
        if r2_co_trong_so > max(r2_tb_don_gian, r2_tot_nhat):
            print(f"   ✅ Trọng số theo hiệu suất hoạt động tốt nhất")
            print(f"   📊 Nên sử dụng weighted ensemble trong production")
        elif r2_tb_don_gian > r2_tot_nhat:
            print(f"   ✅ Trung bình đơn giản hiệu quả")
            print(f"   📊 Cả hai mô hình đều có giá trị tương đương")
        else:
            print(f"   ✅ Mô hình {ten_mo_hinh_tot_nhat} đủ tốt")
            print(f"   📊 Có thể không cần ensemble phức tạp")
        
        return {
            'mo_hinh_rieng_le': diem_r2,
            'cac_chien_luoc': cac_chien_luoc,
            'chien_luoc_tot_nhat': chien_luoc_tot_nhat
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong ensemble tùy chỉnh: {e}")
        return None

def main():
    """Chạy tất cả ví dụ nâng cao tiếng Việt"""
    print("🚀 VÍ DỤ PIPELINE ML NÂNG CAO - TIẾNG VIỆT")
    print("=" * 60)
    print("Minh họa các kỹ thuật ML nâng cao production-ready")
    print("với các tính năng tiên tiến cho dự đoán crypto.")
    print("")
    print("🎓 KIẾN THỨC NÂNG CAO:")
    print("   🔧 Tối ưu hóa siêu tham số tự động")
    print("   🎯 Ensemble learning và kết hợp mô hình")
    print("   📊 Phân tích hiệu suất chi tiết")
    print("   💾 Quản lý phiên bản mô hình")
    print("   🚀 Mô phỏng triển khai production")
    
    cac_vi_du = [
        ("Pipeline toàn diện", vi_du_1_pipeline_toan_dien),
        ("Ensemble tùy chỉnh", vi_du_2_ensemble_tuy_chinh)
    ]
    
    ket_qua = {}
    thanh_cong = 0
    
    for i, (ten_vi_du, ham_vi_du) in enumerate(cac_vi_du, 1):
        try:
            print(f"\n🔄 Đang chạy Ví dụ Nâng cao {i}: {ten_vi_du}...")
            result = ham_vi_du()
            ket_qua[ham_vi_du.__name__] = result
            
            if result is not None:
                thanh_cong += 1
                print(f"✅ Ví dụ Nâng cao {i} hoàn thành thành công!")
            else:
                print(f"⚠️ Ví dụ Nâng cao {i} có vấn đề!")
                
        except Exception as e:
            print(f"❌ Ví dụ Nâng cao {i} thất bại: {e}")
            ket_qua[ham_vi_du.__name__] = None
    
    # Tổng kết
    tong_so = len(cac_vi_du)
    print(f"\n{'='*60}")
    print("🎯 TỔNG KẾT VÍ DỤ NÂNG CAO")
    print(f"{'='*60}")
    print(f"✅ Thành công: {thanh_cong}/{tong_so}")
    
    if thanh_cong == tong_so:
        print("\n🎉 TẤT CẢ VÍ DỤ NÂNG CAO HOÀN THÀNH!")
        print("🚀 Bạn đã nắm vững kỹ thuật ML production-ready!")
        
        print(f"\n📚 KIẾN THỨC ĐÃ NÂNG CAO:")
        print(f"   🔧 Pipeline ML toàn diện với tự động hóa")
        print(f"   🤖 Lựa chọn và tối ưu mô hình tự động")
        print(f"   🎯 Ensemble methods nâng cao")
        print(f"   📊 Phân tích hiệu suất sâu")
        print(f"   🚀 Chuẩn bị cho production deployment")
        
        print(f"\n🎯 ÁP DỤNG VÀO WEB:")
        print(f"   1. Tích hợp kết quả tiếng Việt vào giao diện web")
        print(f"   2. Hiển thị so sánh mô hình trực quan")
        print(f"   3. Dashboard theo dõi hiệu suất real-time")
        print(f"   4. API dự đoán với ensemble models")
        
    else:
        print(f"\n⚠️ MỘT SỐ VÍ DỤ CẦN KHẮC PHỤC")
        print("🔍 Xem thông báo lỗi ở trên để điều chỉnh.")
    
    print(f"\n💡 CHUẨN BỊ CHO VIỆC VIẾT TAY:")
    print(f"   📚 Hiểu rõ logic từng thuật toán")
    print(f"   🔢 Nắm vững công thức toán học")
    print(f"   💻 Biết cách implement từ đầu")
    print(f"   🎯 Tối ưu hóa performance")
    
    return ket_qua

if __name__ == "__main__":
    main()