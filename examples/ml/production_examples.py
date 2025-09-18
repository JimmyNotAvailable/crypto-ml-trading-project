#!/usr/bin/env python3
"""
🏭 VÍ DỤ PRODUCTION THỰC TẾ - TIẾNG VIỆT
======================================

Các ví dụ production thực tế minh họa cách triển khai ML models
trong môi trường giao dịch crypto thực tế, hoàn toàn bằng tiếng Việt
để dễ tích hợp vào web interface và hiểu rõ cách hoạt động.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc dự án vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
from app.ml.model_registry import ModelRegistry

class BotGiaoDichTuDong:
    """Bot giao dịch tự động sử dụng dự đoán ML"""
    
    def __init__(self, so_du_ban_dau=10000):
        self.so_du = so_du_ban_dau
        self.so_du_ban_dau = so_du_ban_dau
        self.vi_the = {}  # Danh sách các vị thế đang mở
        self.lich_su_giao_dich = []
        self.cac_mo_hinh = {}
        self.quan_ly_rui_ro = {
            'kich_thuoc_vi_the_toi_da': 0.2,  # 20% số dư mỗi lệnh
            'stop_loss': 0.05,                # Cắt lỗ 5%
            'take_profit': 0.15,              # Chốt lời 15%
            'do_tin_cay_toi_thieu': 0.7       # Độ tin cậy tối thiểu
        }
        
        self._tai_cac_mo_hinh()
    
    def _tai_cac_mo_hinh(self):
        """Tải và huấn luyện các mô hình ML"""
        try:
            datasets = load_prepared_datasets('ml_datasets_top3')
            
            # Huấn luyện và lưu trữ các mô hình
            self.cac_mo_hinh['du_doan_gia'] = LinearRegressionModel(target_type='price')
            self.cac_mo_hinh['du_doan_gia'].train(datasets)
            
            self.cac_mo_hinh['phan_loai_xu_huong'] = KNNClassifier(n_neighbors=5)
            self.cac_mo_hinh['phan_loai_xu_huong'].train(datasets)
            
            self.cac_mo_hinh['phan_tich_thi_truong'] = KMeansClusteringModel(auto_tune=True)
            self.cac_mo_hinh['phan_tich_thi_truong'].train(datasets)
            
            print("✅ Các mô hình bot giao dịch đã được tải thành công")
            
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            self.cac_mo_hinh = {}
    
    def phan_tich_thi_truong(self, du_lieu_thi_truong):
        """Phân tích thị trường toàn diện sử dụng tất cả mô hình ML"""
        try:
            # Chuẩn bị đặc trưng (một hàng DataFrame)
            features_df = pd.DataFrame([du_lieu_thi_truong])
            
            # Dự đoán giá
            gia_du_doan = self.cac_mo_hinh['du_doan_gia'].predict(features_df)[0]
            gia_hien_tai = du_lieu_thi_truong.get('close', 0)
            thay_doi_gia_phan_tram = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
            
            # Phân loại xu hướng
            xac_suat_xu_huong = self.cac_mo_hinh['phan_loai_xu_huong'].predict_proba(features_df)[0]
            cac_lop_xu_huong = self.cac_mo_hinh['phan_loai_xu_huong'].model.classes_
            du_doan_xu_huong = dict(zip(cac_lop_xu_huong, xac_suat_xu_huong))
            
            # Chế độ thị trường
            cum_thi_truong = self.cac_mo_hinh['phan_tich_thi_truong'].predict(features_df)[0]
            
            # Tính điểm tin cậy (đơn giản hóa)
            do_tin_cay = min(
                self.cac_mo_hinh['du_doan_gia'].training_history['test_metrics']['r2'],
                max(xac_suat_xu_huong)
            )
            
            return {
                'gia_hien_tai': gia_hien_tai,
                'gia_du_doan': gia_du_doan,
                'thay_doi_gia_phan_tram': thay_doi_gia_phan_tram,
                'du_doan_xu_huong': du_doan_xu_huong,
                'cum_thi_truong': cum_thi_truong,
                'do_tin_cay': do_tin_cay,
                'thoi_gian': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Lỗi trong phân tích thị trường: {e}")
            return None
    
    def tao_tin_hieu_giao_dich(self, ket_qua_phan_tich):
        """Tạo tín hiệu giao dịch dựa trên phân tích ML"""
        if not ket_qua_phan_tich or ket_qua_phan_tich['do_tin_cay'] < self.quan_ly_rui_ro['do_tin_cay_toi_thieu']:
            return {
                'hanh_dong': 'GIỮ', 
                'ly_do': 'Độ tin cậy thấp hoặc phân tích thất bại',
                'do_tin_cay': ket_qua_phan_tich.get('do_tin_cay', 0) if ket_qua_phan_tich else 0
            }
        
        thay_doi_gia = ket_qua_phan_tich['thay_doi_gia_phan_tram']
        du_doan_xu_huong = ket_qua_phan_tich['du_doan_xu_huong']
        
        # Tìm xu hướng mạnh nhất
        xu_huong_manh_nhat = max(du_doan_xu_huong.keys(), key=lambda x: du_doan_xu_huong[x])
        do_tin_cay_xu_huong = du_doan_xu_huong[xu_huong_manh_nhat]
        
        # Logic giao dịch
        if thay_doi_gia > 5 and xu_huong_manh_nhat == 'Bullish' and do_tin_cay_xu_huong > 0.7:
            return {
                'hanh_dong': 'MUA',
                'ly_do': f'Tín hiệu tăng mạnh: Dự kiến tăng {thay_doi_gia:.2f}%',
                'kich_thuoc_vi_the': min(self.quan_ly_rui_ro['kich_thuoc_vi_the_toi_da'], do_tin_cay_xu_huong),
                'gia_muc_tieu': ket_qua_phan_tich['gia_du_doan'],
                'do_tin_cay': ket_qua_phan_tich['do_tin_cay'],
                'xu_huong_chinh': xu_huong_manh_nhat,
                'suc_manh_xu_huong': do_tin_cay_xu_huong
            }
        elif thay_doi_gia < -5 and xu_huong_manh_nhat == 'Bearish' and do_tin_cay_xu_huong > 0.7:
            return {
                'hanh_dong': 'BÁN',
                'ly_do': f'Tín hiệu giảm mạnh: Dự kiến giảm {abs(thay_doi_gia):.2f}%',
                'kich_thuoc_vi_the': min(self.quan_ly_rui_ro['kich_thuoc_vi_the_toi_da'], do_tin_cay_xu_huong),
                'gia_muc_tieu': ket_qua_phan_tich['gia_du_doan'],
                'do_tin_cay': ket_qua_phan_tich['do_tin_cay'],
                'xu_huong_chinh': xu_huong_manh_nhat,
                'suc_manh_xu_huong': do_tin_cay_xu_huong
            }
        else:
            return {
                'hanh_dong': 'GIỮ',
                'ly_do': f'Tín hiệu yếu: {thay_doi_gia:.2f}% thay đổi, {xu_huong_manh_nhat} với độ tin cậy {do_tin_cay_xu_huong:.2f}',
                'xu_huong_chinh': xu_huong_manh_nhat,
                'suc_manh_xu_huong': do_tin_cay_xu_huong
            }
    
    def thuc_hien_giao_dich(self, tin_hieu, du_lieu_thi_truong):
        """Thực hiện giao dịch dựa trên tín hiệu"""
        if tin_hieu['hanh_dong'] == 'GIỮ':
            return {'trang_thai': 'khong_hanh_dong', 'thong_bao': tin_hieu['ly_do']}
        
        gia_hien_tai = du_lieu_thi_truong['close']
        gia_tri_vi_the = self.so_du * tin_hieu.get('kich_thuoc_vi_the', 0.1)
        
        giao_dich = {
            'thoi_gian': datetime.now(),
            'hanh_dong': tin_hieu['hanh_dong'],
            'gia': gia_hien_tai,
            'so_luong': gia_tri_vi_the / gia_hien_tai if tin_hieu['hanh_dong'] == 'MUA' else gia_tri_vi_the,
            'ly_do': tin_hieu['ly_do'],
            'do_tin_cay': tin_hieu.get('do_tin_cay', 0),
            'gia_muc_tieu': tin_hieu.get('gia_muc_tieu'),
            'stop_loss': gia_hien_tai * (1 - self.quan_ly_rui_ro['stop_loss']) if tin_hieu['hanh_dong'] == 'MUA' else gia_hien_tai * (1 + self.quan_ly_rui_ro['stop_loss']),
            'take_profit': gia_hien_tai * (1 + self.quan_ly_rui_ro['take_profit']) if tin_hieu['hanh_dong'] == 'MUA' else gia_hien_tai * (1 - self.quan_ly_rui_ro['take_profit'])
        }
        
        if tin_hieu['hanh_dong'] == 'MUA':
            chi_phi = giao_dich['so_luong'] * gia_hien_tai
            if chi_phi <= self.so_du:
                self.so_du -= chi_phi
                ma_vi_the = f"BTC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.vi_the[ma_vi_the] = giao_dich
                self.lich_su_giao_dich.append(giao_dich)
                return {
                    'trang_thai': 'thuc_hien', 
                    'giao_dich': giao_dich,
                    'thong_bao': f'Đã mua {giao_dich["so_luong"]:.6f} BTC với giá ${gia_hien_tai:.2f}'
                }
            else:
                return {
                    'trang_thai': 'khong_du_tien', 
                    'can_thiet': chi_phi, 
                    'co_san': self.so_du,
                    'thong_bao': f'Không đủ tiền: Cần ${chi_phi:.2f}, chỉ có ${self.so_du:.2f}'
                }
        
        elif tin_hieu['hanh_dong'] == 'BÁN':
            # Mô phỏng bán vị thế có sẵn
            if self.vi_the:
                # Bán vị thế cũ nhất
                ma_vi_the = list(self.vi_the.keys())[0]
                vi_the = self.vi_the.pop(ma_vi_the)
                
                # Tính lãi/lỗ
                gia_mua = vi_the['gia']
                gia_tri_ban = vi_the['so_luong'] * gia_hien_tai
                lai_lo = gia_tri_ban - (vi_the['so_luong'] * gia_mua)
                
                self.so_du += gia_tri_ban
                giao_dich['lai_lo'] = lai_lo
                giao_dich['gia_mua_truoc'] = gia_mua
                giao_dich['phan_tram_lai_lo'] = (lai_lo / (vi_the['so_luong'] * gia_mua)) * 100
                self.lich_su_giao_dich.append(giao_dich)
                
                return {
                    'trang_thai': 'thuc_hien', 
                    'giao_dich': giao_dich, 
                    'lai_lo': lai_lo,
                    'thong_bao': f'Đã bán {vi_the["so_luong"]:.6f} BTC với P&L: ${lai_lo:+.2f} ({giao_dich["phan_tram_lai_lo"]:+.2f}%)'
                }
            else:
                return {
                    'trang_thai': 'khong_co_vi_the', 
                    'thong_bao': 'Không có vị thế nào để bán'
                }
    
    def lay_tom_tat_hieu_suat(self):
        """Lấy tóm tắt hiệu suất giao dịch"""
        if not self.lich_su_giao_dich:
            return {'thong_bao': 'Chưa có giao dịch nào được thực hiện'}
        
        tong_giao_dich = len(self.lich_su_giao_dich)
        giao_dich_co_loi = sum(1 for gd in self.lich_su_giao_dich if gd.get('lai_lo', 0) > 0)
        
        tong_lai_lo = sum(gd.get('lai_lo', 0) for gd in self.lich_su_giao_dich)
        ty_le_thang = giao_dich_co_loi / tong_giao_dich if tong_giao_dich > 0 else 0
        roi = ((self.so_du - self.so_du_ban_dau) / self.so_du_ban_dau) * 100
        
        return {
            'so_du_ban_dau': self.so_du_ban_dau,
            'so_du_hien_tai': self.so_du,
            'tong_lai_lo': tong_lai_lo,
            'roi_phan_tram': roi,
            'tong_giao_dich': tong_giao_dich,
            'giao_dich_co_loi': giao_dich_co_loi,
            'ty_le_thang': ty_le_thang,
            'vi_the_mo': len(self.vi_the)
        }

def vi_du_1_mo_phong_bot_giao_dich():
    """Ví dụ 1: Mô phỏng bot giao dịch với dự đoán ML"""
    print("🤖 VÍ DỤ 1: MÔ PHỎNG BOT GIAO DỊCH TỰ ĐỘNG")
    print("=" * 60)
    
    try:
        # Khởi tạo bot giao dịch
        bot = BotGiaoDichTuDong(so_du_ban_dau=10000)
        
        # Tải dữ liệu test để mô phỏng
        datasets = load_prepared_datasets('ml_datasets_top3')
        du_lieu_test = datasets['test'].head(20)  # Mô phỏng 20 phiên giao dịch
        
        print(f"🚀 Bắt đầu mô phỏng giao dịch với ${bot.so_du_ban_dau:,.2f}")
        print(f"📊 Mô phỏng {len(du_lieu_test)} phiên giao dịch...")
        
        print(f"\n📚 CÁCH HOẠT ĐỘNG CỦA BOT:")
        print(f"   🔹 Sử dụng 3 mô hình ML: Dự đoán giá + Phân loại xu hướng + Phân tích thị trường")
        print(f"   🔹 Áp dụng quản lý rủi ro: Stop loss 5%, Take profit 15%")
        print(f"   🔹 Chỉ giao dịch khi độ tin cậy > 70%")
        print(f"   🔹 Tối đa 20% tài khoản mỗi lệnh")
        
        # Mô phỏng các phiên giao dịch
        for i, (_, du_lieu_thi_truong) in enumerate(du_lieu_test.iterrows()):
            print(f"\n📈 Phiên giao dịch {i+1}:")
            gia_hien_tai = du_lieu_thi_truong['close']
            print(f"   💰 Giá hiện tại: ${gia_hien_tai:.2f}")
            print(f"   💼 Số dư tài khoản: ${bot.so_du:.2f}")
            
            # Phân tích thị trường
            ket_qua_phan_tich = bot.phan_tich_thi_truong(du_lieu_thi_truong.to_dict())
            
            if ket_qua_phan_tich:
                print(f"   🔮 Giá dự đoán: ${ket_qua_phan_tich['gia_du_doan']:.2f} ({ket_qua_phan_tich['thay_doi_gia_phan_tram']:+.2f}%)")
                print(f"   📊 Độ tin cậy: {ket_qua_phan_tich['do_tin_cay']:.3f}")
                
                # Hiển thị xu hướng mạnh nhất
                xu_huong_manh_nhat = max(ket_qua_phan_tich['du_doan_xu_huong'].keys(), 
                                        key=lambda x: ket_qua_phan_tich['du_doan_xu_huong'][x])
                do_manh = ket_qua_phan_tich['du_doan_xu_huong'][xu_huong_manh_nhat]
                print(f"   📈 Xu hướng: {xu_huong_manh_nhat} ({do_manh:.3f})")
                
                # Tạo tín hiệu giao dịch
                tin_hieu = bot.tao_tin_hieu_giao_dich(ket_qua_phan_tich)
                print(f"   🎯 Tín hiệu: {tin_hieu['hanh_dong']} - {tin_hieu['ly_do']}")
                
                # Thực hiện giao dịch
                if tin_hieu['hanh_dong'] != 'GIỮ':
                    ket_qua = bot.thuc_hien_giao_dich(tin_hieu, du_lieu_thi_truong.to_dict())
                    if ket_qua['trang_thai'] == 'thuc_hien':
                        giao_dich = ket_qua['giao_dich']
                        print(f"   ✅ {ket_qua['thong_bao']}")
                        print(f"      🎯 Mục tiêu: ${giao_dich['gia_muc_tieu']:.2f}")
                        print(f"      🛡️ Stop loss: ${giao_dich['stop_loss']:.2f}")
                        print(f"      🎁 Take profit: ${giao_dich['take_profit']:.2f}")
                        
                        if 'lai_lo' in ket_qua:
                            mau_sac = "💚" if ket_qua['lai_lo'] > 0 else "❤️"
                            print(f"      {mau_sac} P&L: ${ket_qua['lai_lo']:+.2f}")
                    else:
                        print(f"   ❌ {ket_qua['thong_bao']}")
                else:
                    print(f"   ⏸️ Giữ nguyên vị thế")
            else:
                print("   ❌ Phân tích thị trường thất bại")
        
        # Tóm tắt hiệu suất
        hieu_suat = bot.lay_tom_tat_hieu_suat()
        
        print(f"\n📊 KẾT QUẢ MÔ PHỎNG BOT GIAO DỊCH:")
        print(f"{'='*50}")
        print(f"💰 Số dư ban đầu: ${hieu_suat['so_du_ban_dau']:,.2f}")
        print(f"💰 Số dư cuối cùng: ${hieu_suat['so_du_hien_tai']:,.2f}")
        print(f"📈 Tổng P&L: ${hieu_suat['tong_lai_lo']:+,.2f}")
        print(f"📊 ROI: {hieu_suat['roi_phan_tram']:+.2f}%")
        print(f"🎯 Tổng giao dịch: {hieu_suat['tong_giao_dich']}")
        print(f"✅ Giao dịch có lời: {hieu_suat['giao_dich_co_loi']}")
        print(f"🏆 Tỷ lệ thắng: {hieu_suat['ty_le_thang']:.1%}")
        print(f"📍 Vị thế đang mở: {hieu_suat['vi_the_mo']}")
        
        # Đánh giá kết quả
        if hieu_suat['roi_phan_tram'] > 5:
            print("\n🎉 BOT HOẠT ĐỘNG RẤT TỐT!")
            print("   ✅ Chiến lược ML hiệu quả")
            print("   ✅ Quản lý rủi ro tốt")
        elif hieu_suat['roi_phan_tram'] > 0:
            print("\n👍 BOT HOẠT ĐỘNG KHẨN THỂ")
            print("   ✅ Có lãi nhưng cần tối ưu thêm")
            print("   ⚠️ Có thể điều chỉnh tham số")
        else:
            print("\n📉 BOT CẦN CẢI THIỆN")
            print("   ⚠️ Cần xem xét lại chiến lược")
            print("   🔧 Điều chỉnh quản lý rủi ro")
        
        print(f"\n💡 GỢI Ý CẢI THIỆN:")
        print(f"   🔧 Điều chỉnh ngưỡng độ tin cậy ({bot.quan_ly_rui_ro['do_tin_cay_toi_thieu']:.1f})")
        print(f"   📊 Tối ưu kích thước vị thế ({bot.quan_ly_rui_ro['kich_thuoc_vi_the_toi_da']:.1%})")
        print(f"   🎯 Tùy chỉnh stop loss/take profit")
        print(f"   🤖 Thêm các mô hình ML khác")
        
        return {
            'bot': bot,
            'hieu_suat': hieu_suat,
            'lich_su_giao_dich': bot.lich_su_giao_dich
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong mô phỏng bot giao dịch: {e}")
        return None

def vi_du_2_theo_doi_hieu_suat_mo_hinh():
    """Ví dụ 2: Theo dõi hiệu suất và phát hiện drift mô hình"""
    print("\n🔍 VÍ DỤ 2: THEO DÕI HIỆU SUẤT & PHÁT HIỆN DRIFT")
    print("=" * 60)
    
    class BoDeoDoiHieuSuat:
        """Bộ theo dõi hiệu suất mô hình ML"""
        
        def __init__(self):
            self.nhat_ky_du_doan = []
            self.chi_so_mo_hinh = {}
            self.phat_hien_drift = {}
        
        def ghi_nhat_ky_du_doan(self, ten_mo_hinh, dac_trung, du_doan, thuc_te=None, thoi_gian=None):
            """Ghi nhật ký dự đoán để phân tích sau"""
            muc_nhat_ky = {
                'thoi_gian': thoi_gian or datetime.now(),
                'ten_mo_hinh': ten_mo_hinh,
                'dac_trung': dac_trung.to_dict() if hasattr(dac_trung, 'to_dict') else dac_trung,
                'du_doan': du_doan,
                'thuc_te': thuc_te
            }
            self.nhat_ky_du_doan.append(muc_nhat_ky)
        
        def danh_gia_drift_mo_hinh(self, ten_mo_hinh, so_ngay_gan_day=7):
            """Phát hiện drift (suy giảm) hiệu suất mô hình"""
            du_doan_gan_day = [
                p for p in self.nhat_ky_du_doan 
                if p['ten_mo_hinh'] == ten_mo_hinh and 
                p['thoi_gian'] > datetime.now() - timedelta(days=so_ngay_gan_day) and
                p['thuc_te'] is not None
            ]
            
            if len(du_doan_gan_day) < 10:
                return {
                    'trang_thai': 'khong_du_du_lieu', 
                    'thong_bao': 'Cần ít nhất 10 dự đoán gần đây để phân tích'
                }
            
            # Tính hiệu suất gần đây
            loi_gan_day = [abs(p['du_doan'] - p['thuc_te']) for p in du_doan_gan_day]
            mae_gan_day = np.mean(loi_gan_day)
            
            # So sánh với hiệu suất lịch sử (đơn giản hóa)
            du_doan_lich_su = [
                p for p in self.nhat_ky_du_doan 
                if p['ten_mo_hinh'] == ten_mo_hinh and 
                p['thoi_gian'] <= datetime.now() - timedelta(days=so_ngay_gan_day) and
                p['thuc_te'] is not None
            ]
            
            if len(du_doan_lich_su) < 10:
                return {'trang_thai': 'khong_du_du_lieu_lich_su'}
            
            loi_lich_su = [abs(p['du_doan'] - p['thuc_te']) for p in du_doan_lich_su]
            mae_lich_su = np.mean(loi_lich_su)
            
            # Phát hiện drift
            suy_giam_hieu_suat = (mae_gan_day - mae_lich_su) / mae_lich_su
            
            trang_thai_drift = 'on_dinh'
            if suy_giam_hieu_suat > 0.2:  # Kém hơn 20%
                trang_thai_drift = 'drift_nghiem_trong'
            elif suy_giam_hieu_suat > 0.1:  # Kém hơn 10%
                trang_thai_drift = 'drift_nhe'
            
            return {
                'trang_thai': trang_thai_drift,
                'mae_gan_day': mae_gan_day,
                'mae_lich_su': mae_lich_su,
                'thay_doi_hieu_suat_phan_tram': suy_giam_hieu_suat * 100,
                'khuyen_nghi': self._lay_khuyen_nghi_drift(trang_thai_drift)
            }
        
        def _lay_khuyen_nghi_drift(self, trang_thai_drift):
            """Lấy khuyến nghị dựa trên trạng thái drift"""
            khuyen_nghi = {
                'on_dinh': "✅ Hiệu suất mô hình ổn định. Tiếp tục theo dõi.",
                'drift_nhe': "⚠️ Phát hiện drift nhẹ. Cân nhắc huấn luyện lại với dữ liệu mới.",
                'drift_nghiem_trong': "🚨 Drift nghiêm trọng. Cần huấn luyện lại ngay hoặc chuyển sang mô hình dự phòng."
            }
            return khuyen_nghi.get(trang_thai_drift, "❓ Trạng thái drift không xác định")
        
        def tao_bao_cao_hieu_suat(self):
            """Tạo báo cáo hiệu suất toàn diện"""
            if not self.nhat_ky_du_doan:
                return {'thong_bao': 'Chưa có dự đoán nào được ghi nhật ký'}
            
            # Hiệu suất theo mô hình
            hieu_suat_mo_hinh = {}
            for ten_mo_hinh in set(p['ten_mo_hinh'] for p in self.nhat_ky_du_doan):
                du_doan_mo_hinh = [p for p in self.nhat_ky_du_doan if p['ten_mo_hinh'] == ten_mo_hinh]
                
                co_gia_tri_thuc_te = [p for p in du_doan_mo_hinh if p['thuc_te'] is not None]
                if co_gia_tri_thuc_te:
                    loi = [abs(p['du_doan'] - p['thuc_te']) for p in co_gia_tri_thuc_te]
                    hieu_suat_mo_hinh[ten_mo_hinh] = {
                        'tong_du_doan': len(du_doan_mo_hinh),
                        'du_doan_da_xac_minh': len(co_gia_tri_thuc_te),
                        'mae': np.mean(loi),
                        'loi_lon_nhat': max(loi),
                        'do_chinh_xac_trong_5_phan_tram': sum(1 for e in loi if e/co_gia_tri_thuc_te[0]['thuc_te'] < 0.05) / len(loi)
                    }
            
            return {
                'tong_du_doan': len(self.nhat_ky_du_doan),
                'so_mo_hinh_theo_doi': len(hieu_suat_mo_hinh),
                'hieu_suat_mo_hinh': hieu_suat_mo_hinh,
                'thoi_gian_theo_doi': {
                    'bat_dau': min(p['thoi_gian'] for p in self.nhat_ky_du_doan),
                    'ket_thuc': max(p['thoi_gian'] for p in self.nhat_ky_du_doan)
                }
            }
    
    try:
        # Khởi tạo bộ theo dõi hiệu suất
        bo_theo_doi = BoDeoDoiHieuSuat()
        
        # Tải mô hình và dữ liệu
        datasets = load_prepared_datasets('ml_datasets_top3')
        mo_hinh_gia = LinearRegressionModel(target_type='price')
        mo_hinh_gia.train(datasets)
        
        print("📊 Mô phỏng theo dõi dự đoán theo thời gian...")
        
        print(f"\n📚 QUÁ TRÌNH THEO DÕI:")
        print(f"   🔹 Ghi nhật ký mọi dự đoán và kết quả thực tế")
        print(f"   🔹 So sánh hiệu suất hiện tại vs lịch sử")
        print(f"   🔹 Phát hiện drift khi hiệu suất giảm >10%")
        print(f"   🔹 Đưa ra khuyến nghị hành động")
        
        # Mô phỏng dự đoán theo thời gian với drift dần dần
        du_lieu_test = datasets['X_test'].head(100)
        gia_thuc_te = datasets['y_test']['price'].head(100)
        
        for i, (_, dac_trung) in enumerate(du_lieu_test.iterrows()):
            # Mô phỏng data drift bằng cách thêm nhiễu tăng dần
            he_so_drift = 1 + (i / 100) * 0.1  # 10% drift qua 100 dự đoán
            dac_trung_drift = dac_trung * he_so_drift
            
            # Thực hiện dự đoán
            du_doan = mo_hinh_gia.predict(pd.DataFrame([dac_trung_drift]))[0]
            thuc_te = gia_thuc_te.iloc[i]
            
            # Ghi nhật ký dự đoán
            thoi_gian = datetime.now() - timedelta(days=100-i)  # Mô phỏng dữ liệu lịch sử
            bo_theo_doi.ghi_nhat_ky_du_doan(
                ten_mo_hinh='du_doan_gia',
                dac_trung=dac_trung,
                du_doan=du_doan,
                thuc_te=thuc_te,
                thoi_gian=thoi_gian
            )
            
            # Kiểm tra định kỳ (mỗi 20 dự đoán)
            if (i + 1) % 20 == 0:
                print(f"\n   📈 Kiểm tra theo dõi lần {(i+1)//20}:")
                
                # Đánh giá drift
                phan_tich_drift = bo_theo_doi.danh_gia_drift_mo_hinh('du_doan_gia', so_ngay_gan_day=20)
                
                if phan_tich_drift['trang_thai'] != 'khong_du_du_lieu':
                    print(f"      🎯 Trạng thái drift: {phan_tich_drift['trang_thai']}")
                    print(f"      📊 MAE gần đây: ${phan_tich_drift['mae_gan_day']:.2f}")
                    if 'mae_lich_su' in phan_tich_drift:
                        print(f"      📊 MAE lịch sử: ${phan_tich_drift['mae_lich_su']:.2f}")
                        print(f"      📈 Thay đổi hiệu suất: {phan_tich_drift['thay_doi_hieu_suat_phan_tram']:+.1f}%")
                    print(f"      💡 Khuyến nghị: {phan_tich_drift['khuyen_nghi']}")
                else:
                    print(f"      ⏳ {phan_tich_drift['thong_bao']}")
        
        # Báo cáo hiệu suất cuối cùng
        print(f"\n📋 BÁO CÁO HIỆU SUẤT CUỐI CÙNG:")
        print(f"{'='*50}")
        
        bao_cao = bo_theo_doi.tao_bao_cao_hieu_suat()
        
        print(f"📊 Tổng dự đoán: {bao_cao['tong_du_doan']}")
        print(f"🤖 Mô hình theo dõi: {bao_cao['so_mo_hinh_theo_doi']}")
        
        for ten_mo_hinh, hieu_suat in bao_cao['hieu_suat_mo_hinh'].items():
            print(f"\n   📈 {ten_mo_hinh}:")
            print(f"      📋 Tổng dự đoán: {hieu_suat['tong_du_doan']}")
            print(f"      ✅ Đã xác minh: {hieu_suat['du_doan_da_xac_minh']}")
            print(f"      📊 MAE: ${hieu_suat['mae']:.2f}")
            print(f"      ⚡ Lỗi lớn nhất: ${hieu_suat['loi_lon_nhat']:.2f}")
            print(f"      🎯 Độ chính xác (±5%): {hieu_suat['do_chinh_xac_trong_5_phan_tram']:.1%}")
        
        # Kiểm tra drift cuối cùng
        drift_cuoi = bo_theo_doi.danh_gia_drift_mo_hinh('du_doan_gia', so_ngay_gan_day=30)
        print(f"\n🔍 PHÂN TÍCH DRIFT CUỐI CÙNG:")
        print(f"   🎯 Trạng thái: {drift_cuoi['trang_thai']}")
        if drift_cuoi['trang_thai'] != 'khong_du_du_lieu':
            print(f"   📈 Thay đổi hiệu suất: {drift_cuoi['thay_doi_hieu_suat_phan_tram']:+.1f}%")
            print(f"   💡 {drift_cuoi['khuyen_nghi']}")
        
        print(f"\n🎯 GIÁ TRỊ CỦA THEO DÕI HIỆU SUẤT:")
        print(f"   ✅ Phát hiện sớm suy giảm mô hình")
        print(f"   ✅ Đảm bảo chất lượng dự đoán ổn định")
        print(f"   ✅ Tự động hóa quy trình bảo trì mô hình")
        print(f"   ✅ Cung cấp bằng chứng cho quyết định kinh doanh")
        
        return {
            'bo_theo_doi': bo_theo_doi,
            'bao_cao': bao_cao,
            'phan_tich_drift': drift_cuoi
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong theo dõi hiệu suất mô hình: {e}")
        return None

def main():
    """Chạy tất cả ví dụ production tiếng Việt"""
    print("🏭 VÍ DỤ PRODUCTION CRYPTO ML - TIẾNG VIỆT")
    print("=" * 60)
    print("Minh họa các tình huống thực tế trong môi trường production")
    print("cho các ứng dụng ML giao dịch crypto.")
    print("")
    print("🎯 TÌNH HUỐNG PRODUCTION:")
    print("   🤖 Bot giao dịch tự động với ML")
    print("   🔍 Theo dõi và phát hiện drift mô hình")
    print("   📊 Báo cáo hiệu suất real-time")
    print("   ⚠️ Quản lý rủi ro tự động")
    print("   🚀 Triển khai và vận hành thực tế")
    
    cac_vi_du = [
        ("Bot giao dịch tự động", vi_du_1_mo_phong_bot_giao_dich),
        ("Theo dõi hiệu suất mô hình", vi_du_2_theo_doi_hieu_suat_mo_hinh)
    ]
    
    ket_qua = {}
    thanh_cong = 0
    
    for i, (ten_vi_du, ham_vi_du) in enumerate(cac_vi_du, 1):
        try:
            print(f"\n🔄 Đang chạy Ví dụ Production {i}: {ten_vi_du}...")
            result = ham_vi_du()
            ket_qua[ham_vi_du.__name__] = result
            
            if result is not None:
                thanh_cong += 1
                print(f"✅ Ví dụ Production {i} hoàn thành thành công!")
            else:
                print(f"⚠️ Ví dụ Production {i} có vấn đề!")
                
        except Exception as e:
            print(f"❌ Ví dụ Production {i} thất bại: {e}")
            ket_qua[ham_vi_du.__name__] = None
    
    # Tổng kết
    tong_so = len(cac_vi_du)
    print(f"\n{'='*60}")
    print("🎯 TỔNG KẾT VÍ DỤ PRODUCTION")
    print(f"{'='*60}")
    print(f"✅ Thành công: {thanh_cong}/{tong_so}")
    
    if thanh_cong == tong_so:
        print("\n🎉 TẤT CẢ VÍ DỤ PRODUCTION HOÀN THÀNH!")
        print("🚀 Bạn đã hiểu rõ cách triển khai ML trong thực tế!")
        
        print(f"\n📚 KỸ NĂNG PRODUCTION ĐÃ THÀNH THẠO:")
        print(f"   🤖 Bot giao dịch tự động với dự đoán ML")
        print(f"   🔍 Phát hiện và xử lý drift mô hình")
        print(f"   📊 Theo dõi hiệu suất và báo cáo")
        print(f"   ⚠️ Quản lý rủi ro trong trading")
        print(f"   🛡️ Xử lý lỗi và monitoring sức khỏe")
        print(f"   💰 Ứng dụng thực tế trong giao dịch")
        
        print(f"\n🎯 ÁP DỤNG VÀO WEB INTERFACE:")
        print(f"   1. Dashboard theo dõi bot giao dịch real-time")
        print(f"   2. Biểu đồ hiệu suất và P&L")
        print(f"   3. Cảnh báo drift và chất lượng mô hình")
        print(f"   4. Giao diện quản lý rủi ro")
        print(f"   5. Báo cáo giao dịch tự động")
        print(f"   6. API endpoints cho mobile app")
        
        print(f"\n🔗 BƯỚC TIẾP THEO CHO PRODUCTION THỰC TẾ:")
        print(f"   1. Thiết lập infrastructure cloud (AWS/GCP/Azure)")
        print(f"   2. Implement CI/CD pipeline")
        print(f"   3. Thêm logging và monitoring toàn diện")
        print(f"   4. Bảo mật và authentication")
        print(f"   5. Database cho lưu trữ dự đoán và giao dịch")
        print(f"   6. Dashboard và alerting system")
        
    else:
        print(f"\n⚠️ MỘT SỐ VÍ DỤ CẦN KHẮC PHỤC")
        print("🔍 Xem chi tiết lỗi ở trên để điều chỉnh.")
    
    print(f"\n💡 LƯU Ý QUAN TRỌNG:")
    print(f"   🚨 Đây là ví dụ giáo dục, không phải tư vấn tài chính")
    print(f"   📊 Luôn test kỹ lưỡng trước khi dùng tiền thật")
    print(f"   💰 Chỉ đầu tư số tiền có thể chấp nhận mất")
    print(f"   🔄 Cập nhật mô hình thường xuyên")
    print(f"   ⚖️ Tuân thủ quy định pháp luật về giao dịch")
    
    return ket_qua

if __name__ == "__main__":
    main()