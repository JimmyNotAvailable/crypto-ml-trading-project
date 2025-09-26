# app.py
from flask import Flask
import os
app = Flask(__name__)

@app.route('/')
def home():
    return "Crypto Web App"

if __name__ == '__main__':
    port = int(os.getenv('WEB_PORT', '8000'))
    app.run(host='0.0.0.0', port=port)
