# metrics.py
# Advanced Model Evaluation Metrics for Crypto Trading ML

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class CryptoMetrics:
    """
    Comprehensive metrics evaluation for crypto trading ML models
    Combines traditional ML metrics with finance-specific indicators
    """
    
    def __init__(self):
        self.results = {}
    
    # ================== REGRESSION METRICS ==================
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str = "model") -> Dict[str, float]:
        """
        Comprehensive regression evaluation with crypto-specific metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Percentage-based metrics
        metrics['mape'] = self._calculate_mape(y_true, y_pred)
        metrics['smape'] = self._calculate_smape(y_true, y_pred)
        
        # Directional accuracy (for price prediction)
        if len(y_true) > 1:
            metrics['directional_accuracy'] = self._directional_accuracy(y_true, y_pred)
        
        # Price prediction specific metrics
        metrics['price_correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['median_error'] = np.median(np.abs(y_true - y_pred))
        
        self.results[f"{model_name}_regression"] = metrics
        return metrics
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "model", target_name: str = "target") -> Dict[str, Any]:
        """
        Comprehensive classification evaluation
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            model_name: Name of the model
            target_name: Name of the target (trend, volatility, etc.)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Trading-specific metrics
        if target_name == "trend":
            metrics['trend_precision'] = self._trend_precision(y_true, y_pred)
            metrics['bull_bear_ratio'] = self._bull_bear_ratio(y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        self.results[f"{model_name}_{target_name}_classification"] = metrics
        return metrics
    
    # ================== FINANCIAL METRICS ==================
    
    def evaluate_trading_performance(self, actual_prices: np.ndarray, 
                                   predicted_prices: np.ndarray,
                                   initial_investment: float = 10000) -> Dict[str, float]:
        """
        Evaluate trading performance based on price predictions
        
        Args:
            actual_prices: Actual price series
            predicted_prices: Predicted price series
            initial_investment: Starting investment amount
            
        Returns:
            Trading performance metrics
        """
        metrics = {}
        
        # Calculate returns
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
        
        # Trading signals based on predicted direction
        signals = np.sign(predicted_returns)
        trading_returns = actual_returns * signals
        
        # Performance metrics
        metrics['total_return'] = np.sum(trading_returns)
        metrics['annualized_return'] = metrics['total_return'] * 365 / len(trading_returns)
        metrics['sharpe_ratio'] = self._sharpe_ratio(trading_returns)
        metrics['max_drawdown'] = self._max_drawdown(np.cumprod(1 + trading_returns))
        metrics['win_rate'] = np.sum(trading_returns > 0) / len(trading_returns)
        metrics['profit_factor'] = self._profit_factor(trading_returns)
        
        # Investment simulation
        portfolio_value = initial_investment * np.cumprod(1 + trading_returns)
        metrics['final_portfolio_value'] = portfolio_value[-1] if len(portfolio_value) > 0 else initial_investment
        metrics['roi'] = (metrics['final_portfolio_value'] - initial_investment) / initial_investment
        
        self.results['trading_performance'] = metrics
        return metrics
    
    # ================== MODEL COMPARISON ==================
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models across different metrics
        
        Args:
            model_results: Dictionary of model names and their results
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            
            # Extract key metrics
            if 'mae' in results:
                row['MAE'] = results['mae']
            if 'r2' in results:
                row['R²'] = results['r2']
            if 'mape' in results:
                row['MAPE'] = results['mape']
            if 'directional_accuracy' in results:
                row['Directional_Accuracy'] = results['directional_accuracy']
            if 'accuracy' in results:
                row['Classification_Accuracy'] = results['accuracy']
                
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("📊 COMPREHENSIVE MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        
        for model_key, metrics in self.results.items():
            report.append(f"\n🎯 {model_key.upper()}")
            report.append("-" * 50)
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'accuracy' in metric_name or 'precision' in metric_name or 'recall' in metric_name or 'f1' in metric_name:
                        report.append(f"  {metric_name}: {value:.4f}")
                    elif 'mae' in metric_name or 'mse' in metric_name or 'rmse' in metric_name:
                        report.append(f"  {metric_name}: {value:.2f}")
                    elif 'r2' in metric_name or 'correlation' in metric_name:
                        report.append(f"  {metric_name}: {value:.4f}")
                    else:
                        report.append(f"  {metric_name}: {value:.4f}")
        
        return "\n".join(report)
    
    # ================== HELPER METHODS ==================
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy of predicting direction of change"""
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        return np.mean(true_direction == pred_direction)
    
    def _trend_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision specifically for upward trend prediction"""
        if np.sum(y_pred == 1) == 0:
            return 0.0
        return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    
    def _bull_bear_ratio(self, y_pred: np.ndarray) -> float:
        """Ratio of bullish to bearish predictions"""
        bull_count = np.sum(y_pred == 1)
        bear_count = np.sum(y_pred == 0)
        return bull_count / bear_count if bear_count > 0 else float('inf')
    
    def _sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 365
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if np.std(excess_returns) != 0 else 0
    
    def _max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    def _profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (total profit / total loss)"""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_profit = np.sum(profits)
        total_loss = np.sum(np.abs(losses))
        
        return total_profit / total_loss if total_loss != 0 else float('inf')

# ================== CONVENIENCE FUNCTIONS ==================

def quick_regression_eval(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> Dict[str, float]:
    """Quick regression evaluation"""
    evaluator = CryptoMetrics()
    return evaluator.evaluate_regression(y_true, y_pred, model_name)

def quick_classification_eval(y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "model", target_name: str = "target") -> Dict[str, Any]:
    """Quick classification evaluation"""
    evaluator = CryptoMetrics()
    return evaluator.evaluate_classification(y_true, y_pred, model_name, target_name)

def quick_trading_eval(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict[str, float]:
    """Quick trading performance evaluation"""
    evaluator = CryptoMetrics()
    return evaluator.evaluate_trading_performance(actual_prices, predicted_prices)
