# 🐳 Docker Unified Container Guide

## 🚀 Quick Start - Unified Container (Khuyến nghị)

Chạy **cả Discord bot và web server** trong **1 container duy nhất**:

```bash
# 1. Build và chạy unified container
docker-compose up unified

# Hoặc với rebuild
docker-compose up --build unified

# Background mode
docker-compose up -d unified
```

**✅ Kết quả:**
- 🤖 Discord bot sẽ connect và sẵn sàng nhận lệnh
- 🌐 Web server chạy tại http://localhost:8000
- 📊 MongoDB database tự động khởi động
- 🔄 Auto-restart nếu có lỗi

## 📋 Available Commands

### 🎯 Production Usage
```bash
# Chạy unified container (production)
docker-compose up -d unified

# Xem logs realtime
docker-compose logs -f unified

# Stop service
docker-compose down
```

### 🔧 Development & Testing
```bash
# Chạy chỉ web server (legacy)
docker-compose --profile legacy up web

# Chạy chỉ bot (legacy)  
docker-compose --profile legacy up bot

# Chạy demo với dashboard
docker-compose --profile legacy up demo
```

## 🛠️ Configuration

### Environment Variables
Tạo file `.env` hoặc sửa `.env.example`:

```bash
# Discord Bot
BOT_TOKEN=your_discord_bot_token_here

# Web Server
WEB_PORT=8000

# Database
MONGO_URL=mongodb://mongo:27017/crypto_ml
```

### Token File
Hoặc tạo file `token.txt` với Discord bot token:
```bash
echo "your_discord_bot_token" > token.txt
```

## 📊 Service Architecture

### Unified Container
```
┌─────────────────────────────────┐
│        crypto_unified           │
├─────────────────────────────────┤
│  📱 unified_launcher.py         │
│  ├── 🤖 Discord Bot Process     │
│  └── 🌐 Web Server Process      │
│                                 │
│  🔄 Process Monitoring          │
│  📋 Centralized Logging         │
│  ⚡ Auto-restart on Failure     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│         MongoDB                 │
│     (crypto_mongo)              │
└─────────────────────────────────┘
```

## 🎛️ Advanced Usage

### Custom Port
```bash
WEB_PORT=3000 docker-compose up unified
```

### Debug Mode
```bash
# Xem logs chi tiết
docker-compose up unified --verbose

# Access container shell
docker exec -it crypto_unified bash
```

### Production Deployment
```bash
# Build optimized image
docker build -t crypto-unified:prod .

# Run with resource limits
docker run -d \
  --name crypto-prod \
  --restart unless-stopped \
  --memory="512m" \
  --cpus="1.0" \
  -p 8000:8000 \
  -v $(pwd)/token.txt:/app/token.txt:ro \
  crypto-unified:prod
```

## 🔍 Troubleshooting

### Container Logs
```bash
# Unified service logs
docker-compose logs unified

# Bot-specific logs
docker-compose logs unified | grep "\[BOT\]"

# Web-specific logs  
docker-compose logs unified | grep "\[WEB\]"
```

### Health Check
```bash
# Check if services are running
curl http://localhost:8000/health

# Check container status
docker-compose ps
```

### Common Issues

**Bot không connect:**
- Kiểm tra `token.txt` hoặc `BOT_TOKEN` environment variable
- Xem logs: `docker-compose logs unified`

**Web không truy cập được:**
- Kiểm tra port mapping: `-p 8000:8000`
- Firewall có block port 8000 không

**Container crash:**
- Kiểm tra logs để xem lỗi cụ thể
- Restart: `docker-compose restart unified`

## 📈 Benefits of Unified Container

✅ **Đơn giản hóa deployment** - Chỉ cần 1 lệnh duy nhất  
✅ **Ít resource overhead** - Không cần nhiều container  
✅ **Centralized logging** - Tất cả logs ở một chỗ  
✅ **Process monitoring** - Auto-restart khi crash  
✅ **Easier scaling** - Scale cả bot lẫn web cùng lúc  
✅ **Development friendly** - Test cả 2 service cùng lúc  

## 🆚 So sánh với Legacy Setup

| Feature | Unified Container | Legacy (3 containers) |
|---------|------------------|----------------------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐ Complex |
| **Resource Usage** | ⭐⭐⭐ Low | ⭐ High |
| **Startup Time** | ⭐⭐⭐ Fast | ⭐⭐ Slower |
| **Monitoring** | ⭐⭐⭐ Centralized | ⭐ Scattered |
| **Development** | ⭐⭐⭐ Easy | ⭐⭐ Moderate |