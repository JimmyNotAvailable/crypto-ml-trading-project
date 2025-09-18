#!/usr/bin/env python3

import pickle
import sys
import os

# Add correct path to app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
sys.path.append(app_dir)

# Import with proper path
from app.ml.evaluate import load_trained_models, evaluate_regression_models

# Load models
print("Loading models...")
models = load_trained_models()

# Load datasets
print("Loading datasets...")
with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
    datasets = pickle.load(f)

print("Running regression evaluation...")
regression_results = evaluate_regression_models(models, datasets)

print("Results:")
for model, results in regression_results.items():
    print(f"\n{model}:")
    print(f"  {results}")