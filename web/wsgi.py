"""WSGI entrypoint for the Flask web app."""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web.app import app as application  # WSGI servers expect 'application'

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000)
