# 🔄 QUY TRÌNH ĐỒNG BỘ GIT

## Khi BẮT ĐẦU làm việc (trên bất kỳ máy nào):

```bash
# 1. Kéo code mới nhất từ GitHub
git pull origin main

# 2. Kiểm tra trạng thái
git status

# 3. Kích hoạt môi trường Python
crypto-venv\Scripts\activate  # Windows
# source crypto-venv/bin/activate  # Linux/Mac
```

## Khi HOÀN THÀNH công việc:

```bash
# 1. Kiểm tra thay đổi
git status
git diff

# 2. Thêm files đã thay đổi
git add .
# hoặc thêm từng file cụ thể:
# git add examples/ml/new_feature.py

# 3. Commit với message mô tả
git commit -m "feat: thêm tính năng ABC"
# hoặc:
# git commit -m "fix: sửa lỗi XYZ"
# git commit -m "docs: cập nhật README"

# 4. Đẩy lên GitHub
git push origin main
```

## Các trường hợp đặc biệt:

### Nếu có CONFLICT khi pull:
```bash
git pull origin main
# Nếu có conflict, sửa file conflict
# Sau đó:
git add .
git commit -m "resolve: merge conflicts"
git push origin main
```

### Nếu quên pull trước khi push:
```bash
git pull --rebase origin main
git push origin main
```

### Kiểm tra lịch sử commit:
```bash
git log --oneline -10  # 10 commit gần nhất
git log --graph --oneline  # Xem dạng cây
```

## ⚠️ LƯU Ý QUAN TRỌNG:

1. **LUÔN `git pull` trước khi bắt đầu làm việc**
2. **LUÔN `git push` sau khi hoàn thành công việc**
3. **KHÔNG commit file .pkl (đã exclude trong .gitignore)**
4. **Models sẽ cần tạo lại trên máy mới**
5. **Backup dữ liệu quan trọng thường xuyên**

## 📁 Cấu trúc thư mục quan trọng:

```
crypto-project/
├── examples/ml/           # Code ML chính
├── data/                  # Dữ liệu (không sync .csv lớn)
├── models/trained/        # Models (không sync .pkl)
├── crypto-venv/          # Môi trường Python (không sync)
├── requirements.txt       # Dependencies (SYNC)
├── README.md             # Documentation (SYNC)
└── .gitignore           # Git config (SYNC)
```