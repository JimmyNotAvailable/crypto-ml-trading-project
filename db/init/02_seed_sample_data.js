// 02_seed_sample_data.js
// Seed dữ liệu mẫu cho các collection

db = db.getSiblingDB('crypto');

db.ohlcv.insertMany([
  { symbol: "1INCHUSDT", tf: "1h", ts: 1694822400, open: 0.25, high: 0.27, low: 0.24, close: 0.26, volume: 10000 },
  { symbol: "AAVEUSDT", tf: "1h", ts: 1694822400, open: 60, high: 62, low: 59, close: 61, volume: 500 },
  { symbol: "ADAUSDT", tf: "1h", ts: 1694822400, open: 0.24, high: 0.25, low: 0.23, close: 0.24, volume: 20000 },
  { symbol: "ALGOUSDT", tf: "1h", ts: 1694822400, open: 0.10, high: 0.11, low: 0.09, close: 0.10, volume: 15000 },
  { symbol: "ATOMUSDT", tf: "1h", ts: 1694822400, open: 7, high: 7.2, low: 6.8, close: 7.1, volume: 3000 },
  { symbol: "AVAXUSDT", tf: "1h", ts: 1694822400, open: 10, high: 10.5, low: 9.8, close: 10.2, volume: 2500 },
  { symbol: "BALUSDT", tf: "1h", ts: 1694822400, open: 3, high: 3.2, low: 2.9, close: 3.1, volume: 800 },
  { symbol: "BCHUSDT", tf: "1h", ts: 1694822400, open: 200, high: 205, low: 198, close: 202, volume: 1200 },
  { symbol: "BNBUSDT", tf: "1h", ts: 1694822400, open: 210, high: 215, low: 208, close: 212, volume: 4000 },
  { symbol: "BTCUSDT", tf: "1h", ts: 1694822400, open: 26000, high: 26200, low: 25900, close: 26100, volume: 12.5 },
  { symbol: "COMPUSDT", tf: "1h", ts: 1694822400, open: 40, high: 41, low: 39, close: 40.5, volume: 600 },
  { symbol: "CRVUSDT", tf: "1h", ts: 1694822400, open: 0.5, high: 0.52, low: 0.49, close: 0.51, volume: 7000 },
  { symbol: "DENTUSDT", tf: "1h", ts: 1694822400, open: 0.001, high: 0.0012, low: 0.0009, close: 0.0011, volume: 100000 },
  { symbol: "DOGEUSDT", tf: "1h", ts: 1694822400, open: 0.06, high: 0.065, low: 0.059, close: 0.062, volume: 50000 },
  { symbol: "DOTUSDT", tf: "1h", ts: 1694822400, open: 4, high: 4.2, low: 3.9, close: 4.1, volume: 9000 },
  { symbol: "DYDXUSDT", tf: "1h", ts: 1694822400, open: 2, high: 2.1, low: 1.9, close: 2.05, volume: 3000 },
  { symbol: "ETCUSDT", tf: "1h", ts: 1694822400, open: 15, high: 15.5, low: 14.8, close: 15.2, volume: 4000 },
  { symbol: "ETHUSDT", tf: "1h", ts: 1694822400, open: 1600, high: 1620, low: 1590, close: 1610, volume: 25.3 },
  { symbol: "FILUSDT", tf: "1h", ts: 1694822400, open: 3, high: 3.1, low: 2.9, close: 3.05, volume: 3500 },
  { symbol: "HBARUSDT", tf: "1h", ts: 1694822400, open: 0.05, high: 0.052, low: 0.049, close: 0.051, volume: 12000 },
  { symbol: "ICPUSDT", tf: "1h", ts: 1694822400, open: 4, high: 4.1, low: 3.9, close: 4.05, volume: 2000 },
  { symbol: "LINKUSDT", tf: "1h", ts: 1694822400, open: 6, high: 6.2, low: 5.8, close: 6.1, volume: 7000 },
  { symbol: "LTCUSDT", tf: "1h", ts: 1694822400, open: 70, high: 72, low: 69, close: 71, volume: 5000 },
  { symbol: "MATICUSDT", tf: "1h", ts: 1694822400, open: 0.5, high: 0.52, low: 0.49, close: 0.51, volume: 15000 },
  { symbol: "MKRUSDT", tf: "1h", ts: 1694822400, open: 1200, high: 1220, low: 1190, close: 1210, volume: 300 },
  { symbol: "RVNUSDT", tf: "1h", ts: 1694822400, open: 0.02, high: 0.021, low: 0.019, close: 0.0205, volume: 8000 },
  { symbol: "SHIBUSDT", tf: "1h", ts: 1694822400, open: 0.000008, high: 0.000009, low: 0.000007, close: 0.0000085, volume: 1000000 },
  { symbol: "SOLUSDT", tf: "1h", ts: 1694822400, open: 20, high: 21, low: 19.8, close: 20.5, volume: 6000 },
  { symbol: "SUSHIUSDT", tf: "1h", ts: 1694822400, open: 1, high: 1.1, low: 0.9, close: 1.05, volume: 2000 },
  { symbol: "TRXUSDT", tf: "1h", ts: 1694822400, open: 0.07, high: 0.072, low: 0.069, close: 0.071, volume: 18000 },
  { symbol: "UNIUSDT", tf: "1h", ts: 1694822400, open: 4, high: 4.2, low: 3.9, close: 4.1, volume: 4000 },
  { symbol: "VETUSDT", tf: "1h", ts: 1694822400, open: 0.02, high: 0.021, low: 0.019, close: 0.0205, volume: 9000 },
  { symbol: "XLMUSDT", tf: "1h", ts: 1694822400, open: 0.1, high: 0.11, low: 0.09, close: 0.105, volume: 11000 },
  { symbol: "XMRUSDT", tf: "1h", ts: 1694822400, open: 150, high: 152, low: 149, close: 151, volume: 700 }
]);
