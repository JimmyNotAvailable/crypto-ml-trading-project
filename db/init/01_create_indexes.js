// 01_create_indexes.js
// Tạo index và TTL cho các collection MongoDB

db = db.getSiblingDB('crypto');

// Index cho collection OHLCV (giá lịch sử)
db.ohlcv.createIndex({ symbol: 1, tf: 1, ts: 1 }); // Truy vấn theo symbol, timeframe, timestamp

// Index cho collection alerts (cảnh báo biến động), có TTL tự động xóa sau 7 ngày
db.alerts.createIndex({ createdAt: 1 }, { expireAfterSeconds: 604800 }); // 7 ngày

// Index cho collection models (lưu checkpoint mô hình)
db.models.createIndex({ name: 1, version: 1 }); // Truy vấn theo tên và version mô hình

// Index cho collection metrics (chỉ số, indicator)
db.metrics.createIndex({ symbol: 1, indicator: 1, ts: 1 }); // Truy vấn theo symbol, indicator, timestamp
