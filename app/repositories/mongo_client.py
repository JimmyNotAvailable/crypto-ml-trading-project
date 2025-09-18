# mongo_client.py
# Kết nối MongoDB, khởi tạo client, get_db từ .env

from pymongo import MongoClient

def test_connection():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["crypto"]
        print("Kết nối MongoDB thành công!")
        print("Database:", db.name)
        print("Collections:", db.list_collection_names())
    except Exception as e:
        print("Kết nối MongoDB thất bại:", e)

if __name__ == "__main__":
    test_connection()
