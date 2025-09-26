#!/usr/bin/env python3
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python path:", sys.path[:3])

try:
    import plotly
    print("✅ Plotly import successful, version:", plotly.__version__)
except ImportError as e:
    print("❌ Plotly import failed:", e)

try:
    import plotly.graph_objs as go
    print("✅ Plotly graph_objs import successful")
except ImportError as e:
    print("❌ Plotly graph_objs import failed:", e)

try:
    import pandas as pd
    print("✅ Pandas import successful")
except ImportError as e:
    print("❌ Pandas import failed:", e)