#!/usr/bin/env python3

import numpy as np

# Test feature reordering logic
current = ['open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'volatility', 'returns', 'hour']
trained = ['open', 'high', 'low', 'close', 'volume', 'volatility', 'returns', 'ma_10', 'ma_50', 'hour']

print("Current:", current)
print("Trained:", trained)

# Create feature index mapping
feature_indices = [current.index(feat) for feat in trained]
print("Feature indices mapping:", feature_indices)

# Expected order after reordering should be trained order:  
expected = trained
print("Expected after reordering:", expected)

# Test with sample data
X_test = np.random.rand(3, 10)  # 3 samples, 10 features
print("\nOriginal X_test:")
for i, feat in enumerate(current):
    print(f"  {i}: {feat} = {X_test[0, i]:.4f}")

X_test_reordered = X_test[:, feature_indices]
print("\nReordered X_test:")
for i, feat in enumerate(trained):
    print(f"  {i}: {feat} = {X_test_reordered[0, i]:.4f}")

# Verify the reordering worked
print("\nVerification:")
for i, feat in enumerate(trained):
    original_idx = current.index(feat)
    reordered_val = X_test_reordered[0, i]
    original_val = X_test[0, original_idx]
    match = np.isclose(reordered_val, original_val)
    print(f"  {feat}: {reordered_val:.4f} == {original_val:.4f} -> {match}")