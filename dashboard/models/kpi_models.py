"""
KPI (Key Performance Indicator) models
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class KPI:
    """
    KPI definition with parameter support
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: What this KPI measures
        sql_template: SQL query (may contain {time_range_days} parameter)
        category: financial | operational | customer | product | custom
        chart_type: metric | line | bar | pie | table
        has_time_parameter: Whether KPI uses time range parameter
        default_time_range_days: Default days for time-based KPIs
        created_at: Creation timestamp
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    sql_template: str = ""
    category: str = "custom"
    chart_type: str = "metric"
    has_time_parameter: bool = False
    default_time_range_days: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_sql(self, time_range_days: Optional[int] = None) -> str:
        """
        Get executable SQL with parameters replaced
        
        Args:
            time_range_days: Override default time range
        
        Returns:
            Executable SQL string
        """
        sql = self.sql_template
        
        if self.has_time_parameter and '{time_range_days}' in sql:
            days = time_range_days or self.default_time_range_days or 30
            sql = sql.replace('{time_range_days}', str(days))
        
        return sql
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate KPI configuration
        
        Returns:
            (is_valid, error_message)
        """
        if not self.name:
            return False, "Name is required"
        
        if not self.sql_template:
            return False, "SQL template is required"
        
        if self.category not in ['financial', 'operational', 'customer', 'product', 'custom']:
            return False, f"Invalid category: {self.category}"
        
        if self.chart_type not in ['metric', 'line', 'bar', 'pie', 'table']:
            return False, f"Invalid chart type: {self.chart_type}"
        
        if self.has_time_parameter and self.default_time_range_days is None:
            return False, "Time-based KPI must have default_time_range_days"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'sql_template': self.sql_template,
            'category': self.category,
            'chart_type': self.chart_type,
            'has_time_parameter': self.has_time_parameter,
            'default_time_range_days': self.default_time_range_days,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KPI':
        """Create KPI from dictionary"""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            description=data['description'],
            sql_template=data['sql_template'],
            category=data.get('category', 'custom'),
            chart_type=data.get('chart_type', 'metric'),
            has_time_parameter=data.get('has_time_parameter', False),
            default_time_range_days=data.get('default_time_range_days'),
            created_at=created_at or datetime.now()
        )
    
    def __repr__(self) -> str:
        return f"KPI({self.name}, {self.category}, {self.chart_type})"