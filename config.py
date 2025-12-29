"""
Configuration module for SQL safety and validation settings
"""

from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class Config:
    """Configuration for SQL safety layer"""
    
    # Read-only mode
    read_only_mode: bool = True
    allow_write_operations: bool = False
    
    # Query limits
    max_result_limit: int = 1000
    auto_inject_limit: bool = True
    default_limit: int = 100
    
    # Join restrictions
    allow_joins: bool = True
    max_joins: int = 5
    warn_join_threshold: int = 3
    
    # Complexity limits
    max_subquery_depth: int = 3
    max_union_operations: int = 3
    
    # Protection settings
    block_system_tables: bool = True
    block_select_star: bool = False  # Warning only
    protected_schemas: List[str] = field(default_factory=lambda: [
        'information_schema', 'pg_catalog', 'mysql',
        'sys', 'msdb', 'master', 'tempdb'
    ])
    
    # Performance settings
    enable_explain: bool = True
    query_timeout_seconds: int = 30
    max_query_length: int = 10000
    
    # Audit settings
    log_all_queries: bool = True
    log_blocked_queries: bool = True
    alert_on_dangerous_patterns: bool = True
    
    # Allowed operations (when not in read-only mode)
    allowed_write_tables: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'read_only_mode': self.read_only_mode,
            'allow_write_operations': self.allow_write_operations,
            'max_result_limit': self.max_result_limit,
            'auto_inject_limit': self.auto_inject_limit,
            'default_limit': self.default_limit,
            'allow_joins': self.allow_joins,
            'max_joins': self.max_joins,
            'warn_join_threshold': self.warn_join_threshold,
            'max_subquery_depth': self.max_subquery_depth,
            'max_union_operations': self.max_union_operations,
            'block_system_tables': self.block_system_tables,
            'block_select_star': self.block_select_star,
            'protected_schemas': self.protected_schemas,
            'enable_explain': self.enable_explain,
            'query_timeout_seconds': self.query_timeout_seconds,
            'max_query_length': self.max_query_length,
            'log_all_queries': self.log_all_queries,
            'log_blocked_queries': self.log_blocked_queries,
            'alert_on_dangerous_patterns': self.alert_on_dangerous_patterns,
            'allowed_write_tables': self.allowed_write_tables
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations
PRESET_CONFIGS = {
    'maximum_safety': Config(
        read_only_mode=True,
        max_result_limit=100,
        auto_inject_limit=True,
        max_joins=3,
        max_subquery_depth=2,
        block_system_tables=True,
        enable_explain=True
    ),
    
    'production': Config(
        read_only_mode=True,
        max_result_limit=1000,
        auto_inject_limit=True,
        max_joins=5,
        max_subquery_depth=3,
        block_system_tables=True,
        enable_explain=True
    ),
    
    'development': Config(
        read_only_mode=False,
        allow_write_operations=True,
        max_result_limit=5000,
        auto_inject_limit=True,
        max_joins=10,
        max_subquery_depth=4,
        block_system_tables=True,
        enable_explain=True
    ),
    
    'permissive': Config(
        read_only_mode=False,
        allow_write_operations=True,
        max_result_limit=10000,
        auto_inject_limit=False,
        allow_joins=True,
        max_joins=20,
        block_system_tables=False,
        enable_explain=False
    )
}


def get_preset_config(preset_name: str) -> Config:
    """
    Get a preset configuration
    
    Args:
        preset_name: Name of preset ('maximum_safety', 'production', 'development', 'permissive')
        
    Returns:
        Config instance
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]