"""
Dashboard configuration models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class Dashboard:
    """
    Dashboard configuration
    
    Attributes:
        id: Unique identifier
        name: Dashboard name
        database_id: Associated database ID
        kpi_ids: List of KPI IDs in this dashboard
        layout: 2D layout grid [[kpi1, kpi2], [kpi3]]
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    database_id: str = ""
    kpi_ids: List[str] = field(default_factory=list)
    layout: List[List[str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_kpi(self, kpi_id: str):
        """Add KPI to dashboard"""
        if kpi_id not in self.kpi_ids:
            self.kpi_ids.append(kpi_id)
            self.updated_at = datetime.now()
    
    def remove_kpi(self, kpi_id: str):
        """Remove KPI from dashboard"""
        if kpi_id in self.kpi_ids:
            self.kpi_ids.remove(kpi_id)
            # Remove from layout as well
            self.layout = [
                [kid for kid in row if kid != kpi_id]
                for row in self.layout
            ]
            # Remove empty rows
            self.layout = [row for row in self.layout if row]
            self.updated_at = datetime.now()
    
    def reorder_kpis(self, new_order: List[str]):
        """Reorder KPIs"""
        # Validate all KPIs are present
        if set(new_order) != set(self.kpi_ids):
            raise ValueError("New order must contain all existing KPIs")
        
        self.kpi_ids = new_order
        self.updated_at = datetime.now()
    
    def get_kpi_count(self) -> int:
        """Get number of KPIs"""
        return len(self.kpi_ids)
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate dashboard configuration
        
        Returns:
            (is_valid, error_message)
        """
        if not self.name:
            return False, "Name is required"
        
        if not self.database_id:
            return False, "Database ID is required"
        
        if not self.kpi_ids:
            return False, "Dashboard must have at least one KPI"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'name': self.name,
            'database_id': self.database_id,
            'kpi_ids': self.kpi_ids,
            'layout': self.layout,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dashboard':
        """Create Dashboard from dictionary"""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            database_id=data['database_id'],
            kpi_ids=data.get('kpi_ids', []),
            layout=data.get('layout', []),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now()
        )
    
    def __repr__(self) -> str:
        return f"Dashboard({self.name}, {len(self.kpi_ids)} KPIs)"
