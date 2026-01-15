"""
Simple JSON-based storage for dashboards and KPIs
Handles persistence to filesystem
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict
import logging

from dashboard.models.kpi_models import KPI
from dashboard.models.dashboard_models import Dashboard

logger = logging.getLogger(__name__)


class DashboardStorage:
    """Handles persistence of dashboards and KPIs"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.dashboards_dir = self.data_dir / "dashboards"
        self.kpis_dir = self.data_dir / "kpis"
        
        # Create directories if they don't exist
        self.dashboards_dir.mkdir(parents=True, exist_ok=True)
        self.kpis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dashboard storage initialized at {self.data_dir}")
    
    # ========================================
    # KPI Operations
    # ========================================
    
    def save_kpi(self, kpi: KPI) -> bool:
        """
        Save KPI to file
        
        Args:
            kpi: KPI instance to save
            
        Returns:
            True if successful
        """
        try:
            # Validate KPI
            is_valid, error = kpi.validate()
            if not is_valid:
                logger.error(f"Cannot save invalid KPI: {error}")
                return False
            
            filepath = self.kpis_dir / f"{kpi.id}.json"
            with open(filepath, 'w') as f:
                json.dump(kpi.to_dict(), f, indent=2)
            
            logger.info(f"Saved KPI: {kpi.name} ({kpi.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save KPI {kpi.id}: {e}")
            return False
    
    def load_kpi(self, kpi_id: str) -> Optional[KPI]:
        """
        Load KPI from file
        
        Args:
            kpi_id: KPI identifier
            
        Returns:
            KPI instance or None if not found
        """
        try:
            filepath = self.kpis_dir / f"{kpi_id}.json"
            if not filepath.exists():
                logger.warning(f"KPI not found: {kpi_id}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            kpi = KPI.from_dict(data)
            logger.debug(f"Loaded KPI: {kpi.name} ({kpi_id})")
            return kpi
            
        except Exception as e:
            logger.error(f"Failed to load KPI {kpi_id}: {e}")
            return None
    
    def delete_kpi(self, kpi_id: str) -> bool:
        """
        Delete KPI file
        
        Args:
            kpi_id: KPI identifier
            
        Returns:
            True if successful
        """
        try:
            filepath = self.kpis_dir / f"{kpi_id}.json"
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted KPI: {kpi_id}")
                return True
            else:
                logger.warning(f"KPI not found for deletion: {kpi_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete KPI {kpi_id}: {e}")
            return False
    
    def list_kpis(self) -> List[KPI]:
        """
        List all KPIs
        
        Returns:
            List of KPI instances
        """
        kpis = []
        try:
            for filepath in self.kpis_dir.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    kpis.append(KPI.from_dict(data))
                except Exception as e:
                    logger.error(f"Failed to load KPI from {filepath}: {e}")
            
            logger.debug(f"Listed {len(kpis)} KPIs")
            return kpis
            
        except Exception as e:
            logger.error(f"Failed to list KPIs: {e}")
            return []
    
    def get_kpis_by_category(self, category: str) -> List[KPI]:
        """
        Get KPIs filtered by category
        
        Args:
            category: Category name
            
        Returns:
            List of KPIs in category
        """
        all_kpis = self.list_kpis()
        return [kpi for kpi in all_kpis if kpi.category == category]
    
    # ========================================
    # Dashboard Operations
    # ========================================
    
    def save_dashboard(self, dashboard: Dashboard) -> bool:
        """
        Save dashboard to file
        
        Args:
            dashboard: Dashboard instance to save
            
        Returns:
            True if successful
        """
        try:
            # Validate dashboard
            is_valid, error = dashboard.validate()
            if not is_valid:
                logger.error(f"Cannot save invalid dashboard: {error}")
                return False
            
            filepath = self.dashboards_dir / f"{dashboard.id}.json"
            with open(filepath, 'w') as f:
                json.dump(dashboard.to_dict(), f, indent=2)
            
            logger.info(f"Saved dashboard: {dashboard.name} ({dashboard.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dashboard {dashboard.id}: {e}")
            return False
    
    def load_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """
        Load dashboard from file
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            Dashboard instance or None if not found
        """
        try:
            filepath = self.dashboards_dir / f"{dashboard_id}.json"
            if not filepath.exists():
                logger.warning(f"Dashboard not found: {dashboard_id}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            dashboard = Dashboard.from_dict(data)
            logger.debug(f"Loaded dashboard: {dashboard.name} ({dashboard_id})")
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to load dashboard {dashboard_id}: {e}")
            return None
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """
        Delete dashboard file
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            True if successful
        """
        try:
            filepath = self.dashboards_dir / f"{dashboard_id}.json"
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted dashboard: {dashboard_id}")
                return True
            else:
                logger.warning(f"Dashboard not found for deletion: {dashboard_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete dashboard {dashboard_id}: {e}")
            return False
    
    def list_dashboards(self) -> List[Dashboard]:
        """
        List all dashboards
        
        Returns:
            List of Dashboard instances
        """
        dashboards = []
        try:
            for filepath in self.dashboards_dir.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    dashboards.append(Dashboard.from_dict(data))
                except Exception as e:
                    logger.error(f"Failed to load dashboard from {filepath}: {e}")
            
            logger.debug(f"Listed {len(dashboards)} dashboards")
            return dashboards
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            return []
    
    def get_dashboards_for_database(self, database_id: str) -> List[Dashboard]:
        """
        Get all dashboards for a specific database
        
        Args:
            database_id: Database identifier
            
        Returns:
            List of dashboards for database
        """
        all_dashboards = self.list_dashboards()
        return [d for d in all_dashboards if d.database_id == database_id]
    
    # ========================================
    # Bulk Operations
    # ========================================
    
    def clear_all(self):
        """Clear all stored data (use with caution!)"""
        try:
            # Delete all KPIs
            for filepath in self.kpis_dir.glob("*.json"):
                filepath.unlink()
            
            # Delete all dashboards
            for filepath in self.dashboards_dir.glob("*.json"):
                filepath.unlink()
            
            logger.warning("Cleared all dashboard data")
            
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")


# Global storage instance
_storage = None

def get_storage() -> DashboardStorage:
    """
    Get global storage instance (singleton pattern)
    
    Returns:
        DashboardStorage instance
    """
    global _storage
    if _storage is None:
        _storage = DashboardStorage()
    return _storage


# Convenience functions
def save_kpi(kpi: KPI) -> bool:
    """Save KPI using global storage"""
    return get_storage().save_kpi(kpi)


def load_kpi(kpi_id: str) -> Optional[KPI]:
    """Load KPI using global storage"""
    return get_storage().load_kpi(kpi_id)


def list_kpis() -> List[KPI]:
    """List all KPIs using global storage"""
    return get_storage().list_kpis()


def save_dashboard(dashboard: Dashboard) -> bool:
    """Save dashboard using global storage"""
    return get_storage().save_dashboard(dashboard)


def load_dashboard(dashboard_id: str) -> Optional[Dashboard]:
    """Load dashboard using global storage"""
    return get_storage().load_dashboard(dashboard_id)


def list_dashboards() -> List[Dashboard]:
    """List all dashboards using global storage"""
    return get_storage().list_dashboards()