"""
Configuration management for HuggingFace-first phi2 analysis workflows.
Enables switching between local and remote data sources.
"""
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger

@dataclass
class DataSourceConfig:
    """Configuration for data source (local vs remote)."""
    type: str  # "local" or "huggingface"
    repo_id: Optional[str] = None
    local_path: Optional[str] = None
    cache_dir: Optional[str] = None
    max_cache_mb: int = 1000
    
    def __post_init__(self):
        if self.type == "huggingface" and not self.repo_id:
            raise ValueError("repo_id required for huggingface data source")
        if self.type == "local" and not self.local_path:
            raise ValueError("local_path required for local data source")

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    max_points_per_stratum: int = 5000
    pca_components: int = 10
    tda_max_points: int = 1000
    tda_max_dimension: int = 1
    batch_size: int = 1
    components: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = ["Q", "K", "V", "O", "FF1", "FF2"]

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    cleanup_after_analysis: bool = True
    temp_dir_prefix: str = "phi2_analysis_"
    max_concurrent_matrices: int = 4
    enable_garbage_collection: bool = True
    memory_warning_threshold_mb: int = 1000

@dataclass
class Phi2Config:
    """Complete configuration for phi2 analysis."""
    data_source: DataSourceConfig
    analysis: AnalysisConfig
    memory: MemoryConfig
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"✓ Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Phi2Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls(
            data_source=DataSourceConfig(**data['data_source']),
            analysis=AnalysisConfig(**data['analysis']),
            memory=MemoryConfig(**data['memory'])
        )
        logger.info(f"✓ Configuration loaded from {path}")
        return config

# Predefined configurations
def create_huggingface_config(repo_id: str = "totalorganfailure/phi2-weights") -> Phi2Config:
    """Create configuration for HuggingFace-first analysis."""
    return Phi2Config(
        data_source=DataSourceConfig(
            type="huggingface",
            repo_id=repo_id,
            max_cache_mb=1000
        ),
        analysis=AnalysisConfig(
            max_points_per_stratum=5000,
            pca_components=10,
            tda_max_points=1000,
            components=["Q", "K", "V", "O", "FF1", "FF2"]
        ),
        memory=MemoryConfig(
            cleanup_after_analysis=True,
            max_concurrent_matrices=4,
            memory_warning_threshold_mb=2000
        )
    )

def create_local_config(local_path: str = "phi2_analysis_organized") -> Phi2Config:
    """Create configuration for local file analysis."""
    return Phi2Config(
        data_source=DataSourceConfig(
            type="local",
            local_path=local_path
        ),
        analysis=AnalysisConfig(
            max_points_per_stratum=15000,  # Can handle more with local files
            pca_components=20,
            tda_max_points=5000,
            components=["Q", "K", "V", "O", "FF1", "FF2"]
        ),
        memory=MemoryConfig(
            cleanup_after_analysis=False,  # Keep local files
            max_concurrent_matrices=8,
            memory_warning_threshold_mb=4000
        )
    )

def create_minimal_config(repo_id: str = "totalorganfailure/phi2-weights") -> Phi2Config:
    """Create minimal configuration for resource-constrained environments."""
    return Phi2Config(
        data_source=DataSourceConfig(
            type="huggingface",
            repo_id=repo_id,
            max_cache_mb=200
        ),
        analysis=AnalysisConfig(
            max_points_per_stratum=1000,
            pca_components=5,
            tda_max_points=500,
            batch_size=1,
            components=["Q", "K", "V", "O"]  # Skip MLP for minimal
        ),
        memory=MemoryConfig(
            cleanup_after_analysis=True,
            max_concurrent_matrices=2,
            memory_warning_threshold_mb=500
        )
    )

class ConfigManager:
    """Manages configuration switching and validation."""
    
    def __init__(self, config_path: str = "phi2_config.json"):
        self.config_path = config_path
        self.current_config: Optional[Phi2Config] = None
        self.load_or_create_default()
    
    def load_or_create_default(self):
        """Load existing config or create default."""
        if os.path.exists(self.config_path):
            try:
                self.current_config = Phi2Config.load(self.config_path)
                logger.info(f"✓ Loaded existing configuration")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self.current_config = create_huggingface_config()
                self.save_current()
        else:
            self.current_config = create_huggingface_config()
            self.save_current()
    
    def save_current(self):
        """Save current configuration."""
        if self.current_config:
            self.current_config.save(self.config_path)
    
    def switch_to_remote(self, repo_id: str = "totalorganfailure/phi2-weights"):
        """Switch to HuggingFace remote configuration."""
        self.current_config = create_huggingface_config(repo_id)
        self.save_current()
        logger.info(f"🔄 Switched to remote configuration: {repo_id}")
    
    def switch_to_local(self, local_path: str = "phi2_analysis_organized"):
        """Switch to local file configuration."""
        self.current_config = create_local_config(local_path)
        self.save_current()
        logger.info(f"🔄 Switched to local configuration: {local_path}")
    
    def switch_to_minimal(self, repo_id: str = "totalorganfailure/phi2-weights"):
        """Switch to minimal resource configuration."""
        self.current_config = create_minimal_config(repo_id)
        self.save_current()
        logger.info(f"🔄 Switched to minimal configuration")
    
    def get_config(self) -> Phi2Config:
        """Get current configuration."""
        return self.current_config
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        if not self.current_config:
            return {"valid": False, "error": "No configuration loaded"}
        
        validation = {"valid": True, "warnings": []}
        
        # Check data source
        if self.current_config.data_source.type == "huggingface":
            if not self.current_config.data_source.repo_id:
                validation["valid"] = False
                validation["error"] = "repo_id required for huggingface source"
        
        elif self.current_config.data_source.type == "local":
            if not self.current_config.data_source.local_path:
                validation["valid"] = False
                validation["error"] = "local_path required for local source"
            elif not os.path.exists(self.current_config.data_source.local_path):
                validation["warnings"].append(f"Local path does not exist: {self.current_config.data_source.local_path}")
        
        # Check analysis parameters
        if self.current_config.analysis.max_points_per_stratum < 100:
            validation["warnings"].append("Very low max_points_per_stratum may affect analysis quality")
        
        if self.current_config.analysis.tda_max_points < 50:
            validation["warnings"].append("Very low tda_max_points may affect TDA quality")
        
        # Check memory settings
        if self.current_config.memory.max_concurrent_matrices > 8:
            validation["warnings"].append("High max_concurrent_matrices may cause memory issues")
        
        return validation

def get_current_config() -> Phi2Config:
    """Get current configuration (convenience function)."""
    manager = ConfigManager()
    return manager.get_config()

# CLI-style functions for easy switching
def use_remote_data(repo_id: str = "totalorganfailure/phi2-weights"):
    """Switch to remote HuggingFace data source."""
    manager = ConfigManager()
    manager.switch_to_remote(repo_id)
    return manager.get_config()

def use_local_data(local_path: str = "phi2_analysis_organized"):
    """Switch to local data source."""
    manager = ConfigManager()
    manager.switch_to_local(local_path)
    return manager.get_config()

def use_minimal_resources(repo_id: str = "totalorganfailure/phi2-weights"):
    """Switch to minimal resource configuration."""
    manager = ConfigManager()
    manager.switch_to_minimal(repo_id)
    return manager.get_config()

if __name__ == "__main__":
    # Test configuration management
    logger.info("🧪 Testing configuration management")
    
    # Create and test configurations
    hf_config = create_huggingface_config()
    local_config = create_local_config()
    minimal_config = create_minimal_config()
    
    print(f"HuggingFace config: {hf_config.data_source.type}")
    print(f"Local config: {local_config.data_source.type}")
    print(f"Minimal config: {minimal_config.analysis.max_points_per_stratum}")
    
    # Test config manager
    manager = ConfigManager("test_config.json")
    validation = manager.validate_config()
    print(f"Config validation: {validation}")
    
    # Test switching
    use_remote_data()
    config = get_current_config()
    print(f"Current config type: {config.data_source.type}")
    
    print("✅ Configuration management ready!")