# MongoDB Atlas Connection Configuration
# Production-ready MongoDB connection setup

import os
from urllib.parse import quote_plus

# MongoDB Atlas Configuration
MONGODB_CONFIGS = {
    'local': {
        'uri': 'mongodb://localhost:27017/',
        'database': 'crypto',
        'description': 'Local MongoDB instance'
    },
    
    'atlas_free': {
        # MongoDB Atlas Free Tier Template
        # Replace with your actual connection string
        'uri_template': 'mongodb+srv://{username}:{password}@{cluster}.mongodb.net/{database}?retryWrites=true&w=majority',
        'database': 'crypto',
        'description': 'MongoDB Atlas Free Tier (512MB)'
    },
    
    'atlas_shared': {
        # MongoDB Atlas Shared Cluster Template  
        'uri_template': 'mongodb+srv://{username}:{password}@{cluster}.mongodb.net/{database}?retryWrites=true&w=majority',
        'database': 'crypto_production',
        'description': 'MongoDB Atlas Shared Cluster'
    }
}

def get_mongodb_uri(config_name='local', **kwargs):
    """
    Get MongoDB URI for specified configuration
    
    Args:
        config_name: 'local', 'atlas_free', 'atlas_shared'
        **kwargs: username, password, cluster for Atlas configs
    
    Returns:
        tuple: (uri, database_name)
    """
    config = MONGODB_CONFIGS.get(config_name)
    if not config:
        raise ValueError(f"Unknown config: {config_name}")
    
    if 'uri' in config:
        # Simple URI (local)
        return config['uri'], config['database']
    
    elif 'uri_template' in config:
        # Atlas template - need credentials
        required_params = ['username', 'password', 'cluster']
        missing_params = [p for p in required_params if p not in kwargs]
        
        if missing_params:
            raise ValueError(f"Missing required parameters for {config_name}: {missing_params}")
        
        # URL encode password to handle special characters
        encoded_password = quote_plus(kwargs['password'])
        
        uri = config['uri_template'].format(
            username=kwargs['username'],
            password=encoded_password,
            cluster=kwargs['cluster'],
            database=config['database']
        )
        
        return uri, config['database']
    
    else:
        raise ValueError(f"Invalid config format for {config_name}")

def setup_mongodb_atlas():
    """
    Interactive setup for MongoDB Atlas
    """
    print("🔧 MongoDB Atlas Setup")
    print("="*50)
    
    print("1. Go to https://cloud.mongodb.com")
    print("2. Create a free account and cluster")
    print("3. Create a database user")
    print("4. Add your IP to network access")
    print("5. Get your connection string")
    print()
    
    cluster = input("Enter your cluster name (e.g., cluster0): ").strip()
    username = input("Enter your database username: ").strip()
    password = input("Enter your database password: ").strip()
    
    try:
        uri, database = get_mongodb_uri(
            'atlas_free',
            username=username,
            password=password,
            cluster=cluster
        )
        
        print(f"\n✅ MongoDB Atlas Configuration:")
        print(f"🔗 URI: {uri[:50]}...")  # Show partial URI for security
        print(f"📊 Database: {database}")
        
        # Save to environment file
        env_content = f"""# MongoDB Atlas Configuration
MONGODB_URI={uri}
MONGODB_DATABASE={database}
MONGODB_CONFIG=atlas_free
"""
        
        env_path = os.path.join(os.path.dirname(__file__), '.env.mongodb')
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"💾 Configuration saved to: {env_path}")
        print("⚠️  Keep this file secure and don't commit to version control!")
        
        return uri, database
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None, None

# Environment variable setup
def load_mongodb_config():
    """Load MongoDB config from environment variables"""
    
    # Check for environment variables
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_database = os.getenv('MONGODB_DATABASE')
    
    if mongodb_uri and mongodb_database:
        return mongodb_uri, mongodb_database
    
    # Check for .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env.mongodb')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            
        env_vars = {}
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                env_vars[key] = value
        
        mongodb_uri = env_vars.get('MONGODB_URI')
        mongodb_database = env_vars.get('MONGODB_DATABASE')
        
        if mongodb_uri and mongodb_database:
            return mongodb_uri, mongodb_database
    
    # Fallback to local
    return get_mongodb_uri('local')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MongoDB Configuration Helper')
    parser.add_argument('--setup-atlas', action='store_true',
                       help='Setup MongoDB Atlas interactively')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test current MongoDB connection')
    
    args = parser.parse_args()
    
    if args.setup_atlas:
        setup_mongodb_atlas()
    
    elif args.test_connection:
        try:
            from app.database.mongo_client import CryptoMongoClient
            
            # Load config
            uri, database = load_mongodb_config()
            
            print(f"🔗 Testing connection to: {database}")
            client = CryptoMongoClient(uri, database)
            
            info = client.get_connection_info()
            print(f"✅ Connection successful: {info}")
            
            client.close()
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    else:
        # Show current config
        try:
            uri, database = load_mongodb_config()
            print(f"📊 Current MongoDB Config:")
            print(f"Database: {database}")
            print(f"URI: {uri[:30]}..." if len(uri) > 30 else uri)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("💡 Use --setup-atlas to configure MongoDB Atlas")