# trainer.py
# Enterprise Model Training và Automation Service

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import asyncio
from dataclasses import dataclass, asdict
import threading
import time

from ..ml.core import ModelPersistence

def get_project_root():
    """Get project root directory"""
    from pathlib import Path
    current_file = Path(__file__).resolve()
    # Go up from app/services/trainer.py to project root
    return current_file.parent.parent.parent
from ..ml.model_registry import model_registry
from ..ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
from ..ml.data_prep import load_prepared_datasets
from .store import data_store

@dataclass
class TrainingJob:
    """Training job configuration"""
    job_id: str
    model_type: str
    model_name: str
    dataset_name: str
    hyperparameters: Dict[str, Any]
    target_type: str
    created_by: str
    created_at: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model_id: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class EnterpriseTrainer:
    """Enterprise-grade automated training service"""
    
    def __init__(self, trainer_path: Optional[str] = None):
        self.project_root = get_project_root() if get_project_root else Path.cwd()
        self.trainer_path = Path(trainer_path) if trainer_path else self.project_root / "training"
        self.jobs_db_path = self.trainer_path / "training_jobs.json"
        
        # Setup directories
        self._setup_trainer_structure()
        
        # Load/create jobs database
        self.jobs_db = self._load_jobs_db()
        
        # Training job queue
        self.job_queue = []
        self.is_training = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Available model types
        self.model_trainers = {
            'linear_regression': self._train_linear_regression,
            'knn': self._train_knn,
            'kmeans': self._train_kmeans
        }
    
    def _setup_trainer_structure(self):
        """Setup trainer directory structure"""
        directories = [
            self.trainer_path,
            self.trainer_path / "configs",
            self.trainer_path / "logs",
            self.trainer_path / "experiments"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_jobs_db(self) -> Dict[str, Dict[str, Any]]:
        """Load training jobs database"""
        if self.jobs_db_path.exists():
            with open(self.jobs_db_path, 'r') as f:
                return json.load(f)
        return {"jobs": {}}
    
    def _save_jobs_db(self):
        """Save training jobs database"""
        with open(self.jobs_db_path, 'w') as f:
            json.dump(self.jobs_db, f, indent=2, default=str)
    
    def submit_training_job(self, model_type: str, model_name: str, 
                           dataset_name: str = 'ml_datasets_top3',
                           hyperparameters: Optional[Dict[str, Any]] = None,
                           target_type: str = 'price',
                           created_by: str = "AutoTrainer") -> str:
        """Submit a new training job"""
        
        if model_type not in self.model_trainers:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Generate job ID
        job_id = f"{model_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create job
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            model_name=model_name,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters or {},
            target_type=target_type,
            created_by=created_by,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Store job
        self.jobs_db["jobs"][job_id] = asdict(job)
        self._save_jobs_db()
        
        # Add to queue
        self.job_queue.append(job_id)
        
        self.logger.info(f"📝 Submitted training job: {job_id}")
        return job_id
    
    def _train_linear_regression(self, job: TrainingJob) -> Dict[str, Any]:
        """Train linear regression model"""
        try:
            # Load datasets
            datasets = load_prepared_datasets(job.dataset_name)
            
            # Train model using new algorithm class
            model = LinearRegressionModel(target_type=job.target_type)
            
            # Train the model (datasets contains all needed data)
            metrics = model.train(datasets)
            
            # Register model
            model_id = model_registry.register_model(
                model=model.model,  # sklearn model object
                model_name=job.model_name,
                model_type="LinearRegression",
                train_data=(datasets['X_train'], datasets['y_train'][job.target_type]),
                val_data=(datasets['X_val'], datasets['y_val'][job.target_type]),
                test_data=(datasets['X_test'], datasets['y_test'][job.target_type]),
                train_metrics=model.training_history['train_metrics'],
                validation_metrics=model.training_history.get('val_metrics', {}),
                test_metrics=model.training_history.get('test_metrics', {}),
                feature_cols=datasets['feature_cols'],
                target_type=job.target_type,
                hyperparameters=job.hyperparameters,
                dataset_version=job.dataset_name,
                created_by=job.created_by
            )
            
            return {
                'model_id': model_id,
                'metrics': model.training_history,
                'feature_importance': None  # Linear regression doesn't have feature importance in this implementation
            }
            
        except Exception as e:
            raise Exception(f"Linear regression training failed: {str(e)}")
    
    def _train_knn(self, job: TrainingJob) -> Dict[str, Any]:
        """Train KNN model"""
        try:
            # Load datasets
            datasets = load_prepared_datasets(job.dataset_name)
            
            # Determine if it's classification or regression based on target_type
            if job.target_type in ['trend', 'direction']:
                # Classification
                model = KNNClassifier()
                
                # Prepare classification targets (you may need to adapt this based on your data)
                X_train = datasets['X_train']
                y_train = datasets['y_train'][job.target_type] if job.target_type in datasets['y_train'] else datasets['y_train']['price']
                X_val = datasets['X_val']
                y_val = datasets['y_val'][job.target_type] if job.target_type in datasets['y_val'] else datasets['y_val']['price']
                X_test = datasets['X_test']
                y_test = datasets['y_test'][job.target_type] if job.target_type in datasets['y_test'] else datasets['y_test']['price']
                
                # For classification, convert to categorical if needed
                # This is a simplified version - you may need to adapt based on your data structure
                model.train(datasets)
                
            else:
                # Regression
                model = KNNRegressor()
                
                X_train = datasets['X_train']
                y_train = datasets['y_train'][job.target_type]
                X_val = datasets['X_val']
                y_val = datasets['y_val'][job.target_type]
                X_test = datasets['X_test']
                y_test = datasets['y_test'][job.target_type]
                
                model.train(datasets)
            
            return {
                'model_id': f"knn_{job.job_id}",
                'metrics': model.training_history,
                'model_type': 'KNNClassifier' if job.target_type in ['trend', 'direction'] else 'KNNRegressor'
            }
            
        except Exception as e:
            raise Exception(f"KNN training failed: {str(e)}")
    
    def _train_kmeans(self, job: TrainingJob) -> Dict[str, Any]:
        """Train K-Means model"""
        try:
            # Load datasets
            datasets = load_prepared_datasets(job.dataset_name)
            
            # Initialize KMeans clustering model
            model = KMeansClusteringModel()
            
            # Prepare features for clustering (use training data)
            X_train = datasets['X_train']
            
            # Train the clustering model
            model.train(X_train)
            
            # Get cluster assignments for evaluation
            cluster_labels = model.predict(X_train)
            
            return {
                'model_id': f"kmeans_{job.job_id}",
                'metrics': model.training_history,
                'n_clusters': model.n_clusters,
                'feature_cols': datasets['feature_cols']
            }
            
        except Exception as e:
            raise Exception(f"K-Means training failed: {str(e)}")
    
    def run_job(self, job_id: str) -> bool:
        """Run a specific training job"""
        if job_id not in self.jobs_db["jobs"]:
            self.logger.error(f"❌ Job {job_id} not found")
            return False
        
        job_data = self.jobs_db["jobs"][job_id]
        job = TrainingJob(**job_data)
        
        # Update job status
        job.status = "running"
        job.start_time = datetime.now(timezone.utc).isoformat()
        self.jobs_db["jobs"][job_id] = asdict(job)
        self._save_jobs_db()
        
        self.logger.info(f"🚀 Starting training job: {job_id}")
        
        try:
            # Get trainer function
            trainer_func = self.model_trainers[job.model_type]
            
            # Run training
            results = trainer_func(job)
            
            # Update job with results
            job.status = "completed"
            job.end_time = datetime.now(timezone.utc).isoformat()
            job.model_id = results.get('model_id')
            job.metrics = results.get('metrics')
            
            self.logger.info(f"✅ Job completed: {job_id} -> {job.model_id}")
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now(timezone.utc).isoformat()
            job.error_message = str(e)
            
            self.logger.error(f"❌ Job failed: {job_id} - {str(e)}")
        
        # Save final status
        self.jobs_db["jobs"][job_id] = asdict(job)
        self._save_jobs_db()
        
        return job.status == "completed"
    
    def process_queue(self, max_concurrent: int = 1):
        """Process training job queue"""
        self.is_training = True
        
        while self.job_queue and self.is_training:
            # Get next job
            job_id = self.job_queue.pop(0)
            
            self.logger.info(f"🔄 Processing job: {job_id}")
            self.run_job(job_id)
            
            # Small delay between jobs
            time.sleep(1)
        
        self.is_training = False
        self.logger.info("🎯 Queue processing completed")
    
    def start_auto_trainer(self, check_interval: int = 30):
        """Start automatic trainer in background"""
        def auto_trainer():
            while True:
                if self.job_queue and not self.is_training:
                    self.logger.info("🤖 Auto-trainer starting queue processing")
                    self.process_queue()
                
                time.sleep(check_interval)
        
        trainer_thread = threading.Thread(target=auto_trainer, daemon=True)
        trainer_thread.start()
        self.logger.info("🚀 Auto-trainer started")
    
    def list_jobs(self, status: Optional[str] = None) -> pd.DataFrame:
        """List training jobs"""
        jobs_data = []
        
        for job_id, job_data in self.jobs_db["jobs"].items():
            if status and job_data["status"] != status:
                continue
            
            jobs_data.append({
                "job_id": job_id,
                "model_name": job_data["model_name"],
                "model_type": job_data["model_type"],
                "target_type": job_data["target_type"],
                "status": job_data["status"],
                "created_at": job_data["created_at"],
                "created_by": job_data["created_by"],
                "model_id": job_data.get("model_id"),
                "duration": self._calculate_duration(job_data)
            })
        
        return pd.DataFrame(jobs_data)
    
    def _calculate_duration(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Calculate job duration"""
        if not job_data.get("start_time") or not job_data.get("end_time"):
            return None
        
        start = datetime.fromisoformat(job_data["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(job_data["end_time"].replace("Z", "+00:00"))
        
        duration = end - start
        return str(duration).split(".")[0]  # Remove microseconds
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details"""
        return self.jobs_db["jobs"].get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)
            
            # Update status
            if job_id in self.jobs_db["jobs"]:
                self.jobs_db["jobs"][job_id]["status"] = "cancelled"
                self._save_jobs_db()
            
            self.logger.info(f"❌ Cancelled job: {job_id}")
            return True
        
        return False

# Global trainer instance
enterprise_trainer = EnterpriseTrainer()

def submit_training_job(*args, **kwargs):
    """Convenience function for submitting training jobs"""
    return enterprise_trainer.submit_training_job(*args, **kwargs)

def list_training_jobs(*args, **kwargs):
    """Convenience function for listing training jobs"""
    return enterprise_trainer.list_jobs(*args, **kwargs)

def start_auto_trainer(*args, **kwargs):
    """Convenience function for starting auto trainer"""
    return enterprise_trainer.start_auto_trainer(*args, **kwargs)
