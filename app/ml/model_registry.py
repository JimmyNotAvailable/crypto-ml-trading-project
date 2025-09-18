# model_registry.py
# Comprehensive Model Registry và Versioning System

import os
import json
import joblib
import pandas as pd
import numpy as np
import sys
import sklearn
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import hashlib
import shutil
from dataclasses import dataclass, asdict
from app.ml.core import ModelPersistence

def get_project_root():
    """Get project root directory"""
    from pathlib import Path
    current_file = Path(__file__).resolve()
    # Go up from app/ml/model_registry.py to project root
    return current_file.parent.parent.parent

@dataclass
class ModelMetadata:
    """Comprehensive metadata for model tracking"""
    model_id: str
    model_name: str
    model_type: str
    version: str
    timestamp: str
    
    # Training data info
    dataset_version: str
    dataset_hash: str
    feature_cols: List[str]
    target_type: str
    training_samples: int
    validation_samples: int
    test_samples: int
    
    # Model performance
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    
    # Environment info
    python_version: str
    sklearn_version: str
    dependencies: Dict[str, str]
    
    # Model lifecycle
    status: str  # training, validation, production, archived
    created_by: str
    deployment_date: Optional[str] = None
    retirement_date: Optional[str] = None
    
    # Business metrics
    business_impact: Optional[Dict[str, float]] = None
    a_b_test_results: Optional[Dict[str, Any]] = None

class ModelRegistry:
    """Enterprise-grade model registry system"""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.project_root = get_project_root()
        self.registry_path = Path(registry_path) if registry_path else self.project_root / "models"
        self.metadata_db_path = self.registry_path / "model_registry.json"
        
        # Create directory structure
        self._setup_directory_structure()
        
        # Load or create registry database
        self.registry_db = self._load_registry_db()
    
    def _setup_directory_structure(self):
        """Setup model registry directory structure"""
        directories = [
            self.registry_path,
            self.registry_path / "production",
            self.registry_path / "staging", 
            self.registry_path / "archived",
            self.registry_path / "experiments",
            self.registry_path / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_registry_db(self) -> Dict[str, Any]:
        """Load model registry database"""
        if self.metadata_db_path.exists():
            with open(self.metadata_db_path, 'r') as f:
                return json.load(f)
        return {
            "models": {},
            "deployments": {},
            "experiments": {},
            "metadata": {
                "created": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0"
            }
        }
    
    def _save_registry_db(self):
        """Save model registry database"""
        with open(self.metadata_db_path, 'w') as f:
            json.dump(self.registry_db, f, indent=2, default=str)
    
    def _generate_model_id(self, model_name: str, model_type: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{model_type}_{timestamp}"
    
    def _calculate_dataset_hash(self, X, y) -> str:
        """Calculate hash of training dataset for reproducibility"""
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
            
        combined_data = np.concatenate([X_array.flatten(), y_array.flatten()])
        return hashlib.md5(combined_data.tobytes()).hexdigest()[:16]
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get current environment information"""
        import sys
        import sklearn
        import pandas
        import numpy
        
        return {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "pandas_version": pandas.__version__,
            "numpy_version": numpy.__version__
        }
    
    def register_model(self, 
                      model: Any,
                      model_name: str,
                      model_type: str,
                      target_type: str,
                      feature_cols: List[str],
                      train_data: Tuple[np.ndarray, np.ndarray],
                      val_data: Tuple[np.ndarray, np.ndarray],
                      test_data: Tuple[np.ndarray, np.ndarray],
                      train_metrics: Dict[str, float],
                      validation_metrics: Dict[str, float],
                      test_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      dataset_version: str = "v1.0",
                      created_by: str = "ml_pipeline",
                      status: str = "training") -> str:
        """
        Register a new model with comprehensive metadata
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        
        # Generate model ID and version
        model_id = self._generate_model_id(model_name, model_type)
        version = f"v{len([m for m in self.registry_db['models'].values() if m['model_name'] == model_name]) + 1}.0"
        
        # Calculate dataset hash
        X_train, y_train = train_data
        dataset_hash = self._calculate_dataset_hash(X_train, y_train)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            feature_cols=feature_cols,
            target_type=target_type,
            training_samples=len(X_train),
            validation_samples=len(val_data[0]),
            test_samples=len(test_data[0]),
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            hyperparameters=hyperparameters,
            python_version=sys.version,
            sklearn_version=sklearn.__version__,
            dependencies=self._get_environment_info(),
            status=status,
            created_by=created_by
        )
        
        # Save model files
        model_dir = self.registry_path / "experiments" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Update registry database
        self.registry_db["models"][model_id] = asdict(metadata)
        self._save_registry_db()
        
        print(f"✅ Model registered: {model_id}")
        print(f"   📁 Path: {model_dir}")
        print(f"   🎯 Performance: R²={test_metrics.get('r2', 'N/A'):.4f}")
        
        return model_id
    
    def promote_to_staging(self, model_id: str) -> bool:
        """Promote model from experiments to staging"""
        if model_id not in self.registry_db["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Copy model to staging
        exp_dir = self.registry_path / "experiments" / model_id
        staging_dir = self.registry_path / "staging" / model_id
        
        if exp_dir.exists():
            shutil.copytree(exp_dir, staging_dir, dirs_exist_ok=True)
            
            # Update status
            self.registry_db["models"][model_id]["status"] = "staging"
            self._save_registry_db()
            
            print(f"✅ Model {model_id} promoted to staging")
            return True
        
        return False
    
    def promote_to_production(self, model_id: str) -> bool:
        """Promote model from staging to production"""
        if model_id not in self.registry_db["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_metadata = self.registry_db["models"][model_id]
        
        # Validate model before production deployment
        if not self._validate_for_production(model_metadata):
            print(f"❌ Model {model_id} failed production validation")
            return False
        
        # Copy model to production
        staging_dir = self.registry_path / "staging" / model_id
        production_dir = self.registry_path / "production" / model_id
        
        if staging_dir.exists():
            # Backup current production model if exists
            self._backup_current_production(model_metadata["model_name"])
            
            # Deploy new model
            shutil.copytree(staging_dir, production_dir, dirs_exist_ok=True)
            
            # Update status and deployment date
            self.registry_db["models"][model_id]["status"] = "production"
            self.registry_db["models"][model_id]["deployment_date"] = datetime.now(timezone.utc).isoformat()
            
            # Update deployments tracking
            if model_metadata["model_name"] not in self.registry_db["deployments"]:
                self.registry_db["deployments"][model_metadata["model_name"]] = []
            
            self.registry_db["deployments"][model_metadata["model_name"]].append({
                "model_id": model_id,
                "deployment_date": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            })
            
            self._save_registry_db()
            
            print(f"🚀 Model {model_id} deployed to production")
            return True
        
        return False
    
    def _validate_for_production(self, metadata: Dict[str, Any]) -> bool:
        """Validate model meets production requirements"""
        # Check minimum performance thresholds
        test_metrics = metadata.get("test_metrics", {})
        
        # Regression models
        if "r2" in test_metrics:
            if test_metrics["r2"] < 0.8:  # Minimum R² threshold
                print(f"❌ R² {test_metrics['r2']:.4f} below minimum 0.8")
                return False
        
        # Classification models
        if "accuracy" in test_metrics:
            if test_metrics["accuracy"] < 0.75:  # Minimum accuracy threshold
                print(f"❌ Accuracy {test_metrics['accuracy']:.4f} below minimum 0.75")
                return False
        
        # Check data freshness
        timestamp = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - timestamp).days
        
        if age_days > 30:  # Model older than 30 days
            print(f"⚠️ Model is {age_days} days old, consider retraining")
        
        print(f"✅ Model passed production validation")
        return True
    
    def _backup_current_production(self, model_name: str):
        """Backup current production model before deploying new one"""
        production_models = list((self.registry_path / "production").glob(f"{model_name}_*"))
        
        if production_models:
            for prod_model in production_models:
                backup_path = self.registry_path / "backups" / f"{prod_model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(prod_model, backup_path)
                print(f"📦 Backed up previous model to {backup_path}")
    
    def get_production_model(self, model_name: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Get current production model"""
        # Find active deployment
        deployments = self.registry_db["deployments"].get(model_name, [])
        active_deployments = [d for d in deployments if d["status"] == "active"]
        
        if not active_deployments:
            return None
        
        # Get most recent deployment
        latest_deployment = max(active_deployments, key=lambda x: x["deployment_date"])
        model_id = latest_deployment["model_id"]
        
        # Load model
        production_dir = self.registry_path / "production" / model_id
        model_path = production_dir / "model.joblib"
        metadata_path = production_dir / "metadata.json"
        
        if model_path.exists() and metadata_path.exists():
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        
        return None
    
    def list_models(self, status: Optional[str] = None, model_name: Optional[str] = None) -> pd.DataFrame:
        """List models with optional filtering"""
        models_data = []
        
        for model_id, metadata in self.registry_db["models"].items():
            if status and metadata["status"] != status:
                continue
            if model_name and metadata["model_name"] != model_name:
                continue
            
            models_data.append({
                "model_id": model_id,
                "model_name": metadata["model_name"],
                "model_type": metadata["model_type"],
                "version": metadata["version"],
                "status": metadata["status"],
                "timestamp": metadata["timestamp"],
                "test_r2": metadata["test_metrics"].get("r2"),
                "test_mae": metadata["test_metrics"].get("mae"),
                "test_accuracy": metadata["test_metrics"].get("accuracy")
            })
        
        return pd.DataFrame(models_data)
    
    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get complete lineage for a model family"""
        models = [m for m in self.registry_db["models"].values() if m["model_name"] == model_name]
        deployments = self.registry_db["deployments"].get(model_name, [])
        
        return {
            "model_name": model_name,
            "total_versions": len(models),
            "models": sorted(models, key=lambda x: x["timestamp"]),
            "deployments": sorted(deployments, key=lambda x: x["deployment_date"]),
            "current_production": self._get_current_production_version(model_name)
        }
    
    def _get_current_production_version(self, model_name: str) -> Optional[str]:
        """Get current production version for model"""
        deployments = self.registry_db["deployments"].get(model_name, [])
        active_deployments = [d for d in deployments if d["status"] == "active"]
        
        if active_deployments:
            latest = max(active_deployments, key=lambda x: x["deployment_date"])
            return latest["model_id"]
        
        return None
    
    def retire_model(self, model_id: str, reason: str = ""):
        """Retire a model"""
        if model_id in self.registry_db["models"]:
            self.registry_db["models"][model_id]["status"] = "archived"
            self.registry_db["models"][model_id]["retirement_date"] = datetime.now(timezone.utc).isoformat()
            self.registry_db["models"][model_id]["retirement_reason"] = reason
            
            # Move to archived
            production_dir = self.registry_path / "production" / model_id
            archived_dir = self.registry_path / "archived" / model_id
            
            if production_dir.exists():
                shutil.move(production_dir, archived_dir)
            
            # Update deployments
            model_name = self.registry_db["models"][model_id]["model_name"]
            deployments = self.registry_db["deployments"].get(model_name, [])
            for deployment in deployments:
                if deployment["model_id"] == model_id:
                    deployment["status"] = "retired"
            
            self._save_registry_db()
            print(f"📦 Model {model_id} retired")

# Global registry instance
model_registry = ModelRegistry()

def register_model(*args, **kwargs):
    """Convenience function for model registration"""
    return model_registry.register_model(*args, **kwargs)

def get_production_model(model_name: str):
    """Convenience function to get production model"""
    return model_registry.get_production_model(model_name)

def list_models(**kwargs):
    """Convenience function to list models"""
    return model_registry.list_models(**kwargs)