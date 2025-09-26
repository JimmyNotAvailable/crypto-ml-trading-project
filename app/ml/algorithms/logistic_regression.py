"""
🧮 LOGISTIC REGRESSION MODEL
===========================

Simple Logistic Regression wrapper for trend classification.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression for trend classification (0/1)."""

    def __init__(self):
        super().__init__(model_name="logistic_regression_trend", model_type="classification")
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metadata['algorithm'] = 'LogisticRegression'
        self.metadata['target_type'] = 'trend'

    # -----------------------
    # Helpers
    # -----------------------
    def _check_trained(self):
        if not getattr(self, "is_trained", False) or self.feature_columns is None or getattr(self, "model", None) is None:
            raise ValueError("Model chưa được train. Hãy gọi .train(...) trước khi predict/evaluate.")

    def _ensure_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            return X.to_frame().T if X.name is not None else pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        return X

    # -----------------------
    # Train / Prepare
    # -----------------------
    def _prepare(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        train = datasets['train'].copy()
        exclude = ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']
        features = [c for c in train.columns if c not in exclude]
        if not features:
            raise ValueError("No feature columns found")
        self.feature_columns = features
        X = train[features].dropna()
        # 'target_trend' có thể không tồn tại trong một số dataset => suy ra từ 'target_price_change'
        if 'target_trend' in train.columns:
            y = train.loc[X.index, 'target_trend'].astype(int)
        elif 'target_price_change' in train.columns:
            # Quy ước: tăng (>=0) -> 1, giảm (<0) -> 0
            y = (train.loc[X.index, 'target_price_change'] >= 0).astype(int)
        else:
            raise ValueError("Không tìm thấy cột 'target_trend' hoặc 'target_price_change' để huấn luyện.")
        return X, y

    def train(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        X, y = self._prepare(datasets)
        Xs = self.scaler.fit_transform(X)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(Xs, y)
        yhat = self.model.predict(Xs)
        metrics = {'train_accuracy': float(accuracy_score(y, yhat))}
        # Mark trained before running test evaluation to satisfy checks
        self.is_trained = True
        if 'test' in datasets:
            tm = self._eval_test(datasets['test'])
            metrics.update(tm)
        n_features = len(self.feature_columns) if self.feature_columns is not None else 0
        self.training_history = {'metrics': metrics, 'n_samples': len(X), 'n_features': n_features}
        return metrics

    # -----------------------
    # Evaluation on test split
    # -----------------------
    def _eval_test(self, test_df: pd.DataFrame) -> Dict[str, float]:
        self._check_trained()
        features = list(self.feature_columns or [])
        if not features:
            raise ValueError("Không có danh sách cột đặc trưng để đánh giá.")
        missing = [c for c in features if c not in test_df.columns]
        if missing:
            raise ValueError(f"Test set thiếu các cột đặc trưng: {missing}")

        X = test_df.loc[:, features].copy()
        valid_mask = X.notna().all(axis='columns')
        valid_idx = X.index[valid_mask]
        X = X.loc[valid_idx]
        if X.empty:
            raise ValueError("Test set không còn bản ghi hợp lệ sau khi loại NaN.")
        if 'target_trend' in test_df.columns:
            y = test_df.loc[valid_idx, 'target_trend'].astype(int)
        elif 'target_price_change' in test_df.columns:
            y = (test_df.loc[valid_idx, 'target_price_change'] >= 0).astype(int)
        else:
            raise ValueError("Không tìm thấy 'target_trend' hoặc 'target_price_change' trên test set.")

        Xs = self.scaler.transform(X)
        yhat = self.model.predict(Xs)
        return {'test_accuracy': float(accuracy_score(y, yhat))}

    # -----------------------
    # Predict
    # -----------------------
    def _prepare_X_for_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hoá và lọc X để dự đoán: trả về DataFrame đã sắp cột và loại NaN."""
        # Chuẩn hoá input về DataFrame
        X = self._ensure_dataframe(X)
        # Kiểm tra thiếu cột
        features = list(self.feature_columns or [])
        if not features:
            raise ValueError("Model chưa có danh sách cột đặc trưng.")
        missing_cols = [c for c in features if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột đặc trưng: {missing_cols}")
        # Giữ đúng thứ tự cột như lúc train và loại NaN theo hàng
        X = X.loc[:, features].copy()
        valid_mask = X.notna().all(axis='columns')
        valid_idx = X.index[valid_mask]
        X = X.loc[valid_idx]
        if X.empty:
            raise ValueError("Không có bản ghi hợp lệ để dự đoán (toàn NaN ở các cột đặc trưng).")
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về mảng 1 chiều các nhãn 0/1 để tương thích với loader hiện có."""
        self._check_trained()
        X_clean = self._prepare_X_for_predict(X)
        Xs = self.scaler.transform(X_clean)
        yhat = self.model.predict(Xs)
        # Đảm bảo là ndarray 1D kiểu int
        return np.asarray(yhat, dtype=int)

    # -----------------------
    # Evaluate on arbitrary X, y
    # -----------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        self._check_trained()
        # Chuẩn bị X giống hệt lúc predict để đảm bảo cùng các hàng hợp lệ
        X_clean = self._prepare_X_for_predict(X)
        # Căn chỉnh y theo index của X_clean, fallback nếu không có các index tương ứng
        try:
            y_aligned = y.loc[X_clean.index].astype(int)
        except Exception:
            y_aligned = y.iloc[:len(X_clean)].astype(int)
        y_pred = self.predict(X_clean)
        return {'accuracy': float(accuracy_score(y_aligned, y_pred))}
