#!/usr/bin/env python3
"""
🌐 WEB DASHBOARD CRYPTO ML - TIẾNG VIỆT
======================================

Dashboard web hoàn chỉnh để hiển thị kết quả ML, so sánh mô hình,
và giao diện quản lý bot giao dịch crypto, hoàn toàn tiếng Việt.
"""

import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
import plotly.graph_objs as go
import plotly.express as px
import json
import pandas as pd
import numpy as np
from typing import Any, Dict

# Thêm thư mục gốc dự án vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
from examples.ml.production_examples import BotGiaoDichTuDong

app = Flask(__name__)
app.secret_key = 'crypto-ml-dashboard-2024'

class CryptoMLDashboard:
    """Lớp quản lý dashboard ML cho crypto trading"""
    
    def __init__(self):
        self.bot_giao_dich = None
        self.cac_mo_hinh: Dict[str, Any] = {}
        self.du_lieu_thi_truong = None
        self.khu_vuc_thoi_gian = 'Asia/Ho_Chi_Minh'
        self._khoi_tao_database()
        self._tai_mo_hinh()
    
    def _khoi_tao_database(self):
        """Khởi tạo SQLite database để lưu trữ dữ liệu"""
        try:
            self.conn = sqlite3.connect('crypto_ml_dashboard.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Bảng dự đoán
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS du_doan (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thoi_gian TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ten_mo_hinh TEXT,
                    gia_hien_tai REAL,
                    gia_du_doan REAL,
                    thay_doi_phan_tram REAL,
                    do_tin_cay REAL,
                    xu_huong TEXT,
                    cum_thi_truong INTEGER
                )
            ''')
            
            # Bảng giao dịch bot
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS giao_dich_bot (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thoi_gian TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hanh_dong TEXT,
                    gia REAL,
                    so_luong REAL,
                    gia_tri REAL,
                    lai_lo REAL,
                    do_tin_cay REAL,
                    ly_do TEXT
                )
            ''')
            
            # Bảng hiệu suất mô hình
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hieu_suat_mo_hinh (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thoi_gian TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ten_mo_hinh TEXT,
                    mae REAL,
                    rmse REAL,
                    r2 REAL,
                    accuracy REAL
                )
            ''')
            
            self.conn.commit()
            print("✅ Database đã được khởi tạo")
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo database: {e}")
    
    def _tai_mo_hinh(self):
        """Tải và huấn luyện các mô hình ML"""
        try:
            datasets = load_prepared_datasets('ml_datasets_top3')
            self.du_lieu_thi_truong = datasets
            # Build compatibility datasets for algorithm classes expecting 'train'/'test'
            datasets_compat = {
                'train': datasets.get('train_df', pd.DataFrame()),
                'test': datasets.get('test_df', pd.DataFrame())
            }
            
            # Tạo dữ liệu mock cho training nếu cần
            if 'X_train' in datasets and 'y_train_price' not in datasets:
                # Tạo mock target data từ features (rắn chắc cho cả numpy/pandas)
                X_train = datasets['X_train']
                try:
                    if isinstance(X_train, pd.DataFrame):
                        datasets['y_train_price'] = X_train.iloc[:, 0]
                    elif isinstance(X_train, pd.Series):
                        datasets['y_train_price'] = X_train
                    else:
                        # numpy ndarray
                        import numpy as _np
                        arr = _np.asarray(X_train)
                        if arr.ndim == 1:
                            datasets['y_train_price'] = pd.Series(arr)
                        else:
                            datasets['y_train_price'] = pd.Series(arr[:, 0])
                except Exception as _e:
                    # Fallback: constant series
                    datasets['y_train_price'] = pd.Series([0.0] * (len(X_train) if hasattr(X_train, '__len__') else 100))
            
            # Huấn luyện mô hình dự đoán giá (bỏ qua lỗi training)
            try:
                self.cac_mo_hinh['du_doan_gia'] = LinearRegressionModel(target_type='price')
                self.cac_mo_hinh['du_doan_gia'].train(datasets_compat)
            except Exception as train_error:
                print(f"⚠️ Training model warning: {train_error}")
                # Tạo mock model để demo
                self.cac_mo_hinh['du_doan_gia'] = self._create_mock_model('du_doan_gia')
            
            # Mô hình phân loại xu hướng (bỏ qua lỗi)
            try:
                self.cac_mo_hinh['phan_loai_xu_huong'] = KNNClassifier(n_neighbors=5)
                self.cac_mo_hinh['phan_loai_xu_huong'].train(datasets_compat)
            except Exception as train_error:
                print(f"⚠️ KNN training warning: {train_error}")
                self.cac_mo_hinh['phan_loai_xu_huong'] = self._create_mock_model('phan_loai_xu_huong')
            
            # Mô hình phân tích thị trường (bỏ qua lỗi)
            try:
                self.cac_mo_hinh['phan_tich_thi_truong'] = KMeansClusteringModel(auto_tune=True)
                self.cac_mo_hinh['phan_tich_thi_truong'].train(datasets_compat)
            except Exception as train_error:
                print(f"⚠️ Clustering training warning: {train_error}")
                self.cac_mo_hinh['phan_tich_thi_truong'] = self._create_mock_model('phan_tich_thi_truong')
            
            # Khởi tạo bot giao dịch
            self.bot_giao_dich = BotGiaoDichTuDong(so_du_ban_dau=10000)
            
            print("✅ Tất cả mô hình đã được tải thành công")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tải mô hình: {e}")
            # Tạo mock models để dashboard vẫn hoạt động
            self._create_mock_models()
            return False
    
    def _create_mock_model(self, model_name):
        """Tạo mock model để demo"""
        class MockModel:
            def __init__(self, name):
                self.model_name = name
                self.training_history = {
                    'test_metrics': {
                        'mae': 0.1,
                        'rmse': 0.2,
                        'r2': 0.8
                    },
                    'training_time': 1.5
                }
            
            def predict(self, data):
                return [42000.0]  # Mock prediction
        
        return MockModel(model_name)
    
    def _create_mock_models(self):
        """Tạo tất cả mock models"""
        self.cac_mo_hinh = {
            'du_doan_gia': self._create_mock_model('du_doan_gia'),
            'phan_loai_xu_huong': self._create_mock_model('phan_loai_xu_huong'),
            'phan_tich_thi_truong': self._create_mock_model('phan_tich_thi_truong')
        }
        self.bot_giao_dich = None
        print("🔧 Đã tạo mock models để demo")
    
    def lay_du_doan_moi_nhat(self, so_luong=20):
        """Lấy các dự đoán mới nhất từ database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM du_doan 
                ORDER BY thoi_gian DESC 
                LIMIT ?
            ''', (so_luong,))
            
            columns = [description[0] for description in cursor.description]
            du_doan = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return du_doan
        except Exception as e:
            print(f"❌ Lỗi lấy dự đoán: {e}")
            return []
    
    def luu_du_doan_moi(self, ket_qua_phan_tich):
        """Lưu dự đoán mới vào database"""
        try:
            cursor = self.conn.cursor()
            
            # Tìm xu hướng mạnh nhất
            xu_huong_manh_nhat = max(
                ket_qua_phan_tich['du_doan_xu_huong'].keys(),
                key=lambda x: ket_qua_phan_tich['du_doan_xu_huong'][x]
            )
            
            cursor.execute('''
                INSERT INTO du_doan (
                    ten_mo_hinh, gia_hien_tai, gia_du_doan, 
                    thay_doi_phan_tram, do_tin_cay, xu_huong, cum_thi_truong
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'ensemble_ml',
                ket_qua_phan_tich['gia_hien_tai'],
                ket_qua_phan_tich['gia_du_doan'],
                ket_qua_phan_tich['thay_doi_gia_phan_tram'],
                ket_qua_phan_tich['do_tin_cay'],
                xu_huong_manh_nhat,
                ket_qua_phan_tich['cum_thi_truong']
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lưu dự đoán: {e}")
            return False
    
    def so_sanh_mo_hinh(self):
        """So sánh hiệu suất các mô hình"""
        try:
            ket_qua_so_sanh = {}
            
            for ten_mo_hinh, mo_hinh in self.cac_mo_hinh.items():
                if hasattr(mo_hinh, 'training_history'):
                    lich_su = mo_hinh.training_history
                    
                    metrics = lich_su.get('metrics', {})
                    if metrics:
                        # Chuẩn hóa metrics để template dùng r2/mae/rmse
                        metrics_display = {
                            'r2': metrics.get('test_r2') or metrics.get('r2') or metrics.get('train_r2'),
                            'mae': metrics.get('test_mae') or metrics.get('mae') or metrics.get('train_mae'),
                            'rmse': metrics.get('test_rmse') or metrics.get('rmse') or metrics.get('train_rmse')
                        }
                        ket_qua_so_sanh[ten_mo_hinh] = {
                            'ten_hien_thi': self._lay_ten_hien_thi(ten_mo_hinh),
                            'metrics': metrics_display,
                            'loai': mo_hinh.__class__.__name__,
                            'thoi_gian_train': lich_su.get('training_time', 0)
                        }
            
            return ket_qua_so_sanh
            
        except Exception as e:
            print(f"❌ Lỗi so sánh mô hình: {e}")
            return {}
    
    def _lay_ten_hien_thi(self, ten_mo_hinh):
        """Chuyển đổi tên mô hình sang tiếng Việt"""
        chuyen_doi = {
            'du_doan_gia': 'Dự đoán Giá',
            'phan_loai_xu_huong': 'Phân loại Xu hướng',
            'phan_tich_thi_truong': 'Phân tích Thị trường'
        }
        return chuyen_doi.get(ten_mo_hinh, ten_mo_hinh)
    
    def tao_bieu_do_gia(self, so_diem=50):
        """Tạo biểu đồ dự đoán giá"""
        try:
            du_doan = self.lay_du_doan_moi_nhat(so_diem)
            
            if not du_doan:
                return None
            
            df = pd.DataFrame(du_doan)
            df['thoi_gian'] = pd.to_datetime(df['thoi_gian'])
            df = df.sort_values('thoi_gian')
            
            fig = go.Figure()
            
            # Giá thực tế
            fig.add_trace(go.Scatter(
                x=df['thoi_gian'],
                y=df['gia_hien_tai'],
                mode='lines+markers',
                name='Giá Thực tế',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Giá dự đoán
            fig.add_trace(go.Scatter(
                x=df['thoi_gian'],
                y=df['gia_du_doan'],
                mode='lines+markers',
                name='Giá Dự đoán',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title='Dự đoán Giá Bitcoin (BTC/USD)',
                xaxis_title='Thời gian',
                yaxis_title='Giá (USD)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ giá: {e}")
            return None
    
    def tao_bieu_do_do_tin_cay(self, so_diem=30):
        """Tạo biểu đồ độ tin cậy theo thời gian"""
        try:
            du_doan = self.lay_du_doan_moi_nhat(so_diem)
            
            if not du_doan:
                return None
            
            df = pd.DataFrame(du_doan)
            df['thoi_gian'] = pd.to_datetime(df['thoi_gian'])
            df = df.sort_values('thoi_gian')
            
            fig = go.Figure()
            
            # Độ tin cậy
            mau_sac = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green' for x in df['do_tin_cay']]
            
            fig.add_trace(go.Bar(
                x=df['thoi_gian'],
                y=df['do_tin_cay'],
                name='Độ Tin cậy',
                marker=dict(color=mau_sac),
                text=[f"{x:.2%}" for x in df['do_tin_cay']],
                textposition='auto'
            ))
            
            # Đường ngưỡng
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         annotation_text="Ngưỡng tốt (70%)")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="Ngưỡng chấp nhận (50%)")
            
            fig.update_layout(
                title='Độ Tin cậy Dự đoán Theo Thời gian',
                xaxis_title='Thời gian',
                yaxis_title='Độ Tin cậy',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                template='plotly_white',
                height=350
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ độ tin cậy: {e}")
            return None

# Khởi tạo dashboard global
dashboard = CryptoMLDashboard()

@app.route('/')
def trang_chu():
    """Trang chủ dashboard"""
    try:
        # Lấy dự đoán mới nhất
        du_doan_moi_nhat = dashboard.lay_du_doan_moi_nhat(1)
        du_doan_hien_tai = du_doan_moi_nhat[0] if du_doan_moi_nhat else None
        
        # So sánh mô hình
        so_sanh_mo_hinh = dashboard.so_sanh_mo_hinh()
        
        # Hiệu suất bot
        hieu_suat_bot = None
        if dashboard.bot_giao_dich:
            hieu_suat_bot = dashboard.bot_giao_dich.lay_tom_tat_hieu_suat()
        
        return render_template('dashboard.html',
                             du_doan_hien_tai=du_doan_hien_tai,
                             so_sanh_mo_hinh=so_sanh_mo_hinh,
                             hieu_suat_bot=hieu_suat_bot)
    
    except Exception as e:
        return render_template('error.html', loi=str(e))

@app.route('/api/du-doan-moi')
def api_du_doan_moi():
    """API tạo dự đoán mới"""
    try:
        # Lấy dữ liệu test mới nhất để mô phỏng (hỗ trợ numpy/pandas)
        if dashboard.du_lieu_thi_truong is not None:
            ds = dashboard.du_lieu_thi_truong
            X_test = ds.get('X_test')
            feature_cols = ds.get('feature_cols') or ds.get('features') or []
            du_lieu_test_dict = None
            if X_test is not None:
                try:
                    if isinstance(X_test, pd.DataFrame):
                        du_lieu_test_dict = X_test.iloc[-1].to_dict()
                    else:
                        arr = np.asarray(X_test)
                        row = arr[-1]
                        if isinstance(feature_cols, (list, tuple)) and len(feature_cols) == len(row):
                            du_lieu_test_dict = {k: float(v) for k, v in zip(feature_cols, row)}
                        else:
                            du_lieu_test_dict = {f'f{i}': float(v) for i, v in enumerate(row)}
                except Exception:
                    du_lieu_test_dict = None
            
            # Thực hiện phân tích với bot
            if dashboard.bot_giao_dich and du_lieu_test_dict is not None:
                ket_qua = dashboard.bot_giao_dich.phan_tich_thi_truong(du_lieu_test_dict)
                
                if ket_qua:
                    # Lưu vào database
                    dashboard.luu_du_doan_moi(ket_qua)
                    
                    return jsonify({
                        'trang_thai': 'thanh_cong',
                        'du_doan': {
                            'gia_hien_tai': ket_qua['gia_hien_tai'],
                            'gia_du_doan': ket_qua['gia_du_doan'],
                            'thay_doi_phan_tram': ket_qua['thay_doi_gia_phan_tram'],
                            'do_tin_cay': ket_qua['do_tin_cay'],
                            'thoi_gian': ket_qua['thoi_gian'].isoformat()
                        }
                    })
        
        return jsonify({'trang_thai': 'loi', 'thong_bao': 'Không thể tạo dự đoán'})
        
    except Exception as e:
        return jsonify({'trang_thai': 'loi', 'thong_bao': str(e)})

@app.route('/api/bieu-do-gia')
def api_bieu_do_gia():
    """API lấy dữ liệu biểu đồ giá"""
    try:
        so_diem = request.args.get('so_diem', 50, type=int)
        bieu_do = dashboard.tao_bieu_do_gia(so_diem)
        
        if bieu_do:
            return jsonify({
                'trang_thai': 'thanh_cong',
                'bieu_do': bieu_do
            })
        else:
            return jsonify({
                'trang_thai': 'loi',
                'thong_bao': 'Không thể tạo biểu đồ'
            })
            
    except Exception as e:
        return jsonify({
            'trang_thai': 'loi',
            'thong_bao': str(e)
        })

@app.route('/api/model-info')
def api_model_info():
    """Trả về thông tin mô hình: tên, metrics, metadata (nếu có)."""
    try:
        info = {
            'available': False,
            'models': {},
            'performance': {},
            'metadata': {},
            'feature_cols': []
        }
        # Load from production package if present
        import os, pickle
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prod_path = os.path.join(root, 'data', 'models_production', 'crypto_models_production.pkl')
        if os.path.exists(prod_path):
            with open(prod_path, 'rb') as f:
                pkg = pickle.load(f)
            info['available'] = True
            info['models'] = list((pkg.get('models') or {}).keys())
            info['performance'] = pkg.get('performance') or {}
            info['metadata'] = pkg.get('metadata') or {}
            info['feature_cols'] = pkg.get('feature_cols') or []
        return jsonify({'trang_thai': 'thanh_cong', 'thong_tin': info})
    except Exception as e:
        return jsonify({'trang_thai': 'loi', 'thong_bao': str(e)})

@app.route('/model-info')
def model_info_page():
    """Trang đơn giản hiển thị thông tin mô hình (không sửa template lớn)."""
    try:
        from markupsafe import Markup
        resp = api_model_info().get_json(silent=True)
        info = resp.get('thong_tin', {}) if isinstance(resp, dict) else {}
        html = [
            '<html><head><meta charset="utf-8"><title>Thông tin Mô hình</title>',
            '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet"></head><body class="p-4">',
            '<h3>Thông tin Mô hình</h3>'
        ]
        if info.get('available'):
            html.append('<div class="alert alert-success">Production models available</div>')
            html.append(f"<p><b>Models:</b> {', '.join(info.get('models', []))}</p>")
            perf = info.get('performance', {})
            if perf:
                html.append('<h5>Performance</h5><ul>')
                for k,v in perf.items():
                    html.append(f"<li>{k}: {v}</li>")
                html.append('</ul>')
            meta = info.get('metadata', {})
            if meta:
                html.append('<h5>Metadata</h5><ul>')
                for k,v in meta.items():
                    html.append(f"<li>{k}: {v}</li>")
                html.append('</ul>')
            html.append(f"<p><b>Features:</b> {len(info.get('feature_cols', []))} columns</p>")
        else:
            html.append('<div class="alert alert-warning">No production models found. Using demo/stub.</div>')
        html.append('<p><a class="btn btn-primary" href="/">Về trang Dashboard</a></p>')
        html.append('</body></html>')
        return Markup('\n'.join(html))
    except Exception as e:
        return f"Lỗi hiển thị: {e}", 500

@app.route('/api/movers')
def api_movers():
    """Top gainers/losers 24h (demo)."""
    gainers = [("BTC", +2.5), ("ETH", +1.8), ("BNB", +1.2)]
    losers = [("SOL", -3.1), ("ADA", -2.2), ("XRP", -1.7)]
    return jsonify({
        'trang_thai': 'thanh_cong',
        'gainers': [{'symbol': s, 'change_pct': p} for s,p in gainers],
        'losers': [{'symbol': s, 'change_pct': p} for s,p in losers]
    })

@app.route('/api/price')
def api_price():
    """Current price (demo) for a symbol."""
    symbol = request.args.get('symbol', 'BTC').upper()
    price = 42100.0
    now = datetime.now().isoformat()
    return jsonify({'symbol': symbol, 'price': price, 'time': now})

@app.route('/api/chart')
def api_chart():
    symbol = request.args.get('symbol', 'BTC').upper()
    link = f"https://www.tradingview.com/chart/?symbol={symbol}USD"
    return jsonify({'symbol': symbol, 'link': link})

@app.route('/api/bieu-do-tin-cay')
def api_bieu_do_tin_cay():
    """API lấy biểu đồ độ tin cậy"""
    try:
        so_diem = request.args.get('so_diem', 30, type=int)
        bieu_do = dashboard.tao_bieu_do_do_tin_cay(so_diem)
        
        if bieu_do:
            return jsonify({
                'trang_thai': 'thanh_cong',
                'bieu_do': bieu_do
            })
        else:
            return jsonify({
                'trang_thai': 'loi',
                'thong_bao': 'Không thể tạo biểu đồ độ tin cậy'
            })
            
    except Exception as e:
        return jsonify({
            'trang_thai': 'loi',
            'thong_bao': str(e)
        })

@app.route('/api/so-sanh-mo-hinh')
def api_so_sanh_mo_hinh():
    """API so sánh hiệu suất mô hình"""
    try:
        so_sanh = dashboard.so_sanh_mo_hinh()
        return jsonify({
            'trang_thai': 'thanh_cong',
            'so_sanh': so_sanh
        })
        
    except Exception as e:
        return jsonify({
            'trang_thai': 'loi',
            'thong_bao': str(e)
        })

@app.route('/bot-giao-dich')
def trang_bot():
    """Trang quản lý bot giao dịch"""
    try:
        hieu_suat = dashboard.bot_giao_dich.lay_tom_tat_hieu_suat() if dashboard.bot_giao_dich else None
        lich_su = dashboard.bot_giao_dich.lich_su_giao_dich[-10:] if dashboard.bot_giao_dich else []
        
        return render_template('bot_trading.html',
                             hieu_suat=hieu_suat,
                             lich_su_giao_dich=lich_su)
    
    except Exception as e:
        return render_template('error.html', loi=str(e))

@app.route('/phan-tich')
def trang_phan_tich():
    """Trang phân tích chi tiết"""
    try:
        so_sanh = dashboard.so_sanh_mo_hinh()
        du_doan_gan_day = dashboard.lay_du_doan_moi_nhat(100)
        
        return render_template('analytics.html',
                             so_sanh_mo_hinh=so_sanh,
                             du_doan_gan_day=du_doan_gan_day)
    
    except Exception as e:
        return render_template('error.html', loi=str(e))

@app.route('/favicon.ico')
def favicon():
    """Serve favicon để tránh lỗi 404"""
    # Tạo response trống cho favicon
    from flask import make_response
    response = make_response('')
    response.headers['Content-Type'] = 'image/x-icon'
    return response

def tao_template_html():
    """Tạo file template HTML cho dashboard"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Template base
    base_template = '''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Crypto ML Dashboard{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .sidebar {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-crypto {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .card-ml {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        .card-trading {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }
        .metric-positive { color: #28a745; }
        .metric-negative { color: #dc3545; }
        .metric-neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav class="col-md-3 col-lg-2 d-md-block sidebar collapse">
                <div class="position-sticky pt-3">
                    <h4 class="text-white text-center mb-4">
                        <i class="fas fa-chart-line"></i> Crypto ML
                    </h4>
                    <ul class="nav flex-column">
                        <li class="nav-item">
                            <a class="nav-link text-white" href="{{ url_for('trang_chu') }}">
                                <i class="fas fa-home"></i> Trang Chủ
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="{{ url_for('trang_bot') }}">
                                <i class="fas fa-robot"></i> Bot Giao Dịch
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="{{ url_for('trang_phan_tich') }}">
                                <i class="fas fa-chart-bar"></i> Phân Tích
                            </a>
                        </li>
                    </ul>
                </div>
            </nav>

            <!-- Main content -->
            <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
                {% block content %}{% endblock %}
            </main>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Template dashboard chính
    dashboard_template = '''{% extends "base.html" %}

{% block title %}Dashboard - Crypto ML{% endblock %}

{% block content %}
<div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
    <h1 class="h2">🚀 Crypto ML Dashboard</h1>
    <div class="btn-toolbar mb-2 mb-md-0">
        <button type="button" class="btn btn-primary" onclick="tao_du_doan_moi()">
            <i class="fas fa-sync"></i> Dự đoán Mới
        </button>
    </div>
</div>

<!-- Cards tổng quan -->
<div class="row mb-4">
    <div class="col-md-4">
        <div class="card card-crypto">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5 class="card-title">Dự đoán Mới nhất</h5>
                        {% if du_doan_hien_tai %}
                        <h3>${{ "%.2f"|format(du_doan_hien_tai.gia_du_doan) }}</h3>
                        <small>
                            {% if du_doan_hien_tai.thay_doi_phan_tram > 0 %}
                            <span class="badge bg-success">+{{ "%.2f"|format(du_doan_hien_tai.thay_doi_phan_tram) }}%</span>
                            {% else %}
                            <span class="badge bg-danger">{{ "%.2f"|format(du_doan_hien_tai.thay_doi_phan_tram) }}%</span>
                            {% endif %}
                        </small>
                        {% else %}
                        <h3>--</h3>
                        <small>Chưa có dữ liệu</small>
                        {% endif %}
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-chart-line fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card card-ml">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5 class="card-title">Độ Tin cậy</h5>
                        {% if du_doan_hien_tai %}
                        <h3>{{ "%.1f"|format(du_doan_hien_tai.do_tin_cay * 100) }}%</h3>
                        <small>{{ du_doan_hien_tai.xu_huong }}</small>
                        {% else %}
                        <h3>--</h3>
                        <small>Chưa có dữ liệu</small>
                        {% endif %}
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-brain fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card card-trading">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5 class="card-title">Bot P&L</h5>
                        {% if hieu_suat_bot %}
                        <h3>
                            {% if hieu_suat_bot.roi_phan_tram > 0 %}
                            <span class="text-success">+{{ "%.2f"|format(hieu_suat_bot.roi_phan_tram) }}%</span>
                            {% else %}
                            <span class="text-danger">{{ "%.2f"|format(hieu_suat_bot.roi_phan_tram) }}%</span>
                            {% endif %}
                        </h3>
                        <small>${{ "%.2f"|format(hieu_suat_bot.so_du_hien_tai) }}</small>
                        {% else %}
                        <h3>--</h3>
                        <small>Bot chưa hoạt động</small>
                        {% endif %}
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-robot fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Biểu đồ -->
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-line"></i> Dự đoán Giá Bitcoin</h5>
            </div>
            <div class="card-body">
                <div id="bieu_do_gia" style="height: 400px;"></div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-thermometer-half"></i> Độ Tin cậy</h5>
            </div>
            <div class="card-body">
                <div id="bieu_do_tin_cay" style="height: 350px;"></div>
            </div>
        </div>
    </div>
</div>

<!-- So sánh mô hình -->
{% if so_sanh_mo_hinh %}
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-balance-scale"></i> So sánh Hiệu suất Mô hình</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Mô hình</th>
                                <th>Loại</th>
                                <th>R² Score</th>
                                <th>MAE</th>
                                <th>RMSE</th>
                                <th>Thời gian Train (s)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for ten, info in so_sanh_mo_hinh.items() %}
                            <tr>
                                <td><strong>{{ info.ten_hien_thi }}</strong></td>
                                <td><span class="badge bg-info">{{ info.loai }}</span></td>
                                <td>
                                    {% if info.metrics.r2 %}
                                    <span class="{% if info.metrics.r2 > 0.8 %}metric-positive{% elif info.metrics.r2 > 0.6 %}metric-neutral{% else %}metric-negative{% endif %}">
                                        {{ "%.3f"|format(info.metrics.r2) }}
                                    </span>
                                    {% else %}--{% endif %}
                                </td>
                                <td>
                                    {% if info.metrics.mae %}
                                    {{ "%.2f"|format(info.metrics.mae) }}
                                    {% else %}--{% endif %}
                                </td>
                                <td>
                                    {% if info.metrics.rmse %}
                                    {{ "%.2f"|format(info.metrics.rmse) }}
                                    {% else %}--{% endif %}
                                </td>
                                <td>{{ "%.2f"|format(info.thoi_gian_train) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endif %}

<!-- Thông tin mô hình và thao tác demo -->
<div class="row mt-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0"><i class="fas fa-info-circle"></i> Thông tin Mô hình</h5>
                <div>
                    <a href="/model-info" class="btn btn-sm btn-outline-primary">Mở trang</a>
                    <button class="btn btn-sm btn-primary" onclick="tai_thong_tin_mo_hinh()">Tải thông tin</button>
                </div>
            </div>
            <div class="card-body">
                <div id="model_info_content" class="small text-muted">Nhấn "Tải thông tin" để xem mô hình, độ chính xác và metadata.</div>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header"><h5 class="mb-0"><i class="fas fa-bolt"></i> Thao tác Demo Nhanh</h5></div>
            <div class="card-body">
                <div class="row g-2">
                    <div class="col-4">
                        <button class="btn btn-outline-secondary w-100" onclick="go_movers()">Movers 24h</button>
                    </div>
                    <div class="col-4">
                        <button class="btn btn-outline-secondary w-100" onclick="go_price()">Giá BTC</button>
                    </div>
                    <div class="col-4">
                        <button class="btn btn-outline-secondary w-100" onclick="go_chart()">Chart BTC</button>
                    </div>
                </div>
                <div id="demo_actions_output" class="mt-3 small text-muted"></div>
            </div>
        </div>
    </div>
</div>

{% endblock %}

{% block scripts %}
<script>
function format_percent(x) {
    try { return (x*100).toFixed(1) + '%'; } catch { return String(x); }
}

function tao_du_doan_moi() {
    fetch('/api/du-doan-moi')
        .then(response => response.json())
        .then(data => {
            if (data.trang_thai === 'thanh_cong') {
                location.reload();
            } else {
                alert('Lỗi tạo dự đoán: ' + data.thong_bao);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Lỗi kết nối API');
        });
}

// Model info loader
function tai_thong_tin_mo_hinh() {
    const el = document.getElementById('model_info_content');
    el.textContent = 'Đang tải...';
    fetch('/api/model-info')
        .then(r => r.json())
        .then(d => {
            if (d.trang_thai !== 'thanh_cong') { el.textContent = 'Không thể tải thông tin mô hình'; return; }
            const info = d.thong_tin || {};
            if (!info.available) { el.innerHTML = '<div class="alert alert-warning">Không tìm thấy mô hình production. Đang dùng chế độ demo.</div>'; return; }
            let html = [];
            html.push('<div class="alert alert-success py-2">Đã phát hiện gói mô hình production.</div>');
            if (info.models && info.models.length) {
                html.push('<p><b>Danh sách mô hình:</b> ' + info.models.join(', ') + '</p>');
            }
            const perf = info.performance || {};
            if (Object.keys(perf).length) {
                html.push('<h6>Hiệu suất</h6><ul class="mb-2">');
                for (const k in perf) { html.push(`<li>${k}: ${perf[k]}</li>`); }
                html.push('</ul>');
            }
            const meta = info.metadata || {};
            if (Object.keys(meta).length) {
                html.push('<h6>Metadata</h6><ul class="mb-2">');
                for (const k in meta) { html.push(`<li>${k}: ${meta[k]}</li>`); }
                html.push('</ul>');
            }
            html.push(`<p><b>Số cột features:</b> ${ (info.feature_cols||[]).length }</p>`);
            el.innerHTML = html.join('\n');
        })
        .catch(() => { el.textContent = 'Lỗi khi tải thông tin mô hình'; });
}

// Demo actions
function go_movers() {
    const out = document.getElementById('demo_actions_output');
    out.textContent = 'Đang lấy danh sách movers...';
    fetch('/api/movers').then(r=>r.json()).then(d=>{
        if (d.trang_thai !== 'thanh_cong') { out.textContent = 'Không lấy được movers'; return; }
        const g = d.gainers || [], l = d.losers || [];
        out.innerHTML = '<div><b>Top tăng:</b> ' + g.map(x=>`${x.symbol} (${x.change_pct}%)`).join(', ') + '</div>' +
                        '<div><b>Top giảm:</b> ' + l.map(x=>`${x.symbol} (${x.change_pct}%)`).join(', ') + '</div>';
    }).catch(()=> out.textContent = 'Lỗi lấy movers');
}
function go_price() {
    const out = document.getElementById('demo_actions_output');
    out.textContent = 'Đang lấy giá...';
    fetch('/api/price?symbol=BTC').then(r=>r.json()).then(d=>{
        out.innerHTML = `<div>BTC hiện tại: <b>$${d.price}</b> (cập nhật: ${d.time})</div>`;
    }).catch(()=> out.textContent = 'Lỗi lấy giá');
}
function go_chart() {
    const out = document.getElementById('demo_actions_output');
    out.textContent = 'Đang lấy link chart...';
    fetch('/api/chart?symbol=BTC').then(r=>r.json()).then(d=>{
        out.innerHTML = `<div>Chart TradingView: <a href="${d.link}" target="_blank">Mở chart ${d.symbol}</a></div>`;
    }).catch(()=> out.textContent = 'Lỗi lấy chart');
}

// Tải biểu đồ giá
fetch('/api/bieu-do-gia')
    .then(response => response.json())
    .then(data => {
        if (data.trang_thai === 'thanh_cong') {
            const bieu_do = JSON.parse(data.bieu_do);
            Plotly.newPlot('bieu_do_gia', bieu_do.data, bieu_do.layout);
        }
    });

// Tải biểu đồ độ tin cậy
fetch('/api/bieu-do-tin-cay')
    .then(response => response.json())
    .then(data => {
        if (data.trang_thai === 'thanh_cong') {
            const bieu_do = JSON.parse(data.bieu_do);
            Plotly.newPlot('bieu_do_tin_cay', bieu_do.data, bieu_do.layout);
        }
    });

// Tự động refresh mỗi 30 giây
setInterval(() => {
    fetch('/api/du-doan-moi');
}, 30000);
</script>
{% endblock %}'''
    
    # Template lỗi
    error_template = '''{% extends "base.html" %}

{% block title %}Lỗi - Crypto ML{% endblock %}

{% block content %}
<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card border-danger">
                <div class="card-header bg-danger text-white">
                    <h5><i class="fas fa-exclamation-triangle"></i> Đã xảy ra lỗi</h5>
                </div>
                <div class="card-body">
                    <p>{{ loi }}</p>
                    <a href="{{ url_for('trang_chu') }}" class="btn btn-primary">
                        <i class="fas fa-home"></i> Về trang chủ
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Lưu các template
    with open(os.path.join(templates_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(dashboard_template)
    
    with open(os.path.join(templates_dir, 'error.html'), 'w', encoding='utf-8') as f:
        f.write(error_template)
    
    print("✅ Đã tạo các template HTML")

def main():
    """Chạy web dashboard"""
    print("🌐 CRYPTO ML WEB DASHBOARD - TIẾNG VIỆT")
    print("=" * 60)
    
    # Tạo templates nếu chưa có
    tao_template_html()
    
    print("🚀 Khởi động web dashboard...")
    host = 'localhost'
    port_env = os.getenv('WEB_PORT', '5000')
    try:
        port_show = int(port_env)
    except Exception:
        port_show = 5000
    print(f"📊 Dashboard sẽ chạy tại: http://{host}:{port_show}")
    print("")
    print("🎯 CÁC TÍNH NĂNG DASHBOARD:")
    print("   📈 Dự đoán giá Bitcoin real-time")
    print("   🤖 Theo dõi bot giao dịch tự động")
    print("   📊 So sánh hiệu suất mô hình ML")
    print("   📉 Biểu đồ độ tin cậy và xu hướng")
    print("   💰 Báo cáo P&L và hiệu suất")
    print("   🔄 Cập nhật tự động mỗi 30 giây")
    print("")
    print("💡 HƯỚNG DẪN SỬ DỤNG:")
    print("   1. Mở trình duyệt và truy cập http://localhost:5000")
    print("   2. Xem dự đoán mới nhất và độ tin cậy")
    print("   3. Nhấn 'Dự đoán Mới' để cập nhật")
    print("   4. Xem tab 'Bot Giao dịch' để theo dõi trading")
    print("   5. Tab 'Phân tích' để so sánh mô hình")
    print("")
    print("🔧 TÙYỈNH VÀ MỞ RỘNG:")
    print("   • Thêm các mô hình ML mới")
    print("   • Tích hợp API exchange thực tế")
    print("   • Cảnh báo qua email/SMS")
    print("   • Lưu trữ dữ liệu lâu dài")
    print("   • Mobile responsive design")
    
    try:
        # Chạy Flask app
        port = int(os.getenv('WEB_PORT', '5000'))
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"❌ Lỗi khởi động web server: {e}")
        print("💡 Đảm bảo port 5000 không bị sử dụng")

if __name__ == "__main__":
    import os
    # Allow overriding port via WEB_PORT for Docker
    port_env = os.getenv('WEB_PORT')
    if port_env:
        try:
            os.environ['FLASK_RUN_PORT'] = str(int(port_env))
        except Exception:
            pass
    main()