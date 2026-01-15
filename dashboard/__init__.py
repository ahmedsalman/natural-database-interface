"""
Dashboard module for ChatDB
Provides schema analysis, KPI management, and visualization
"""
from dashboard.schema_analyzer import SchemaAnalyzer
from dashboard.kpi_engine import KPIEngine
from dashboard.models.schema_models import (
    ColumnInfo,
    TableInfo,
    Relationship,
    SchemaMetadata,
    BusinessContext
)

from dashboard.models.kpi_models import KPI

from dashboard.models.dashboard_models import Dashboard

from dashboard.storage import DashboardStorage, get_storage

__all__ = [
    # Schema models
    'ColumnInfo',
    'TableInfo',
    'Relationship',
    'SchemaMetadata',
    'BusinessContext',
    # KPI models
    'KPI',
    # Dashboard models
    'Dashboard',
    # Storage
    'DashboardStorage',
    'get_storage',

    'SchemaAnalyzer',
    'KPIEngine',
]

__version__ = '2.0.0'