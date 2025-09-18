# store.py
# Enterprise Data Store và Storage Management Service

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import sqlite3
from dataclasses import dataclass, asdict
import logging

from ..ml.core import ModelPersistence

def get_project_root():
    """Get project root directory"""
    from pathlib import Path
    current_file = Path(__file__).resolve()
    # Go up from app/services/store.py to project root
    return current_file.parent.parent.parent
from ..ml.model_registry import model_registry

@dataclass
class DatasetVersion:
    """Data version tracking"""
    version_id: str
    dataset_name: str
    timestamp: str
    file_path: str
    size_mb: float
    row_count: int
    column_count: int
    data_hash: str
    created_by: str
    description: str = ""

class EnterpriseDataStore:
    """Enterprise-grade data storage and versioning service"""
    
    def __init__(self, store_path: Optional[str] = None):
        self.project_root = get_project_root() if get_project_root else Path.cwd()
        self.store_path = Path(store_path) if store_path else self.project_root / "data" / "store"
        self.metadata_db_path = self.store_path / "data_registry.db"
        
        # Setup directories
        self._setup_store_structure()
        
        # Initialize SQLite database
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_store_structure(self):
        """Setup enterprise data store directory structure"""
        directories = [
            self.store_path,
            self.store_path / "raw",
            self.store_path / "processed", 
            self.store_path / "features",
            self.store_path / "datasets",
            self.store_path / "artifacts",
            self.store_path / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for metadata tracking"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_mb REAL NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    data_hash TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_dataset TEXT NOT NULL,
                    target_dataset TEXT NOT NULL,
                    transformation TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            """)
    
    def store_dataset(self, df: pd.DataFrame, dataset_name: str, 
                     created_by: str, description: str = "") -> str:
        """Store dataset with versioning and metadata"""
        
        # Generate version ID
        timestamp = datetime.now(timezone.utc)
        version_id = f"{dataset_name}_v{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate metadata
        size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        data_hash = pd.util.hash_pandas_object(df).sum()
        
        # Save dataset
        file_path = self.store_path / "datasets" / f"{version_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        
        # Create version record
        version = DatasetVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            timestamp=timestamp.isoformat(),
            file_path=str(file_path),
            size_mb=size_mb,
            row_count=len(df),
            column_count=len(df.columns),
            data_hash=str(data_hash),
            created_by=created_by,
            description=description
        )
        
        # Store in database
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id, version.dataset_name, version.timestamp,
                version.file_path, version.size_mb, version.row_count,
                version.column_count, version.data_hash, version.created_by,
                version.description
            ))
        
        self.logger.info(f"✅ Stored dataset {version_id}: {len(df):,} rows, {size_mb:.1f}MB")
        return version_id
    
    def load_dataset(self, version_id: str) -> pd.DataFrame:
        """Load dataset by version ID"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM dataset_versions WHERE version_id = ?",
                (version_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise FileNotFoundError(f"Dataset version {version_id} not found")
            
            file_path = result[0]
            
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        
        self.logger.info(f"📊 Loaded dataset {version_id}: {len(df):,} rows")
        return df
    
    def list_datasets(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """List all dataset versions"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            if dataset_name:
                query = "SELECT * FROM dataset_versions WHERE dataset_name = ? ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn, params=(dataset_name,))
            else:
                query = "SELECT * FROM dataset_versions ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_latest_version(self, dataset_name: str) -> Optional[str]:
        """Get latest version ID for dataset"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.execute("""
                SELECT version_id FROM dataset_versions 
                WHERE dataset_name = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (dataset_name,))
            result = cursor.fetchone()
            
        return result[0] if result else None
    
    def create_backup(self, version_id: str) -> str:
        """Create backup of dataset version"""
        # Load original
        df = self.load_dataset(version_id)
        
        # Create backup
        backup_id = f"{version_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.store_path / "backups" / f"{backup_id}.pkl"
        
        with open(backup_path, 'wb') as f:
            pickle.dump(df, f)
        
        self.logger.info(f"💾 Created backup: {backup_id}")
        return backup_id
    
    def track_lineage(self, source_dataset: str, target_dataset: str, 
                     transformation: str, created_by: str):
        """Track data lineage and transformations"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                INSERT INTO data_lineage (source_dataset, target_dataset, transformation, timestamp, created_by)
                VALUES (?, ?, ?, ?, ?)
            """, (
                source_dataset, target_dataset, transformation,
                datetime.now(timezone.utc).isoformat(), created_by
            ))
        
        self.logger.info(f"📈 Tracked lineage: {source_dataset} -> {target_dataset}")

# Global store instance
data_store = EnterpriseDataStore()

def store_dataset(*args, **kwargs):
    """Convenience function for storing datasets"""
    return data_store.store_dataset(*args, **kwargs)

def load_dataset(*args, **kwargs):
    """Convenience function for loading datasets"""
    return data_store.load_dataset(*args, **kwargs)

def list_datasets(*args, **kwargs):
    """Convenience function for listing datasets"""
    return data_store.list_datasets(*args, **kwargs)
