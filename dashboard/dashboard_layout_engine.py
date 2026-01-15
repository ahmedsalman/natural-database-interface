"""
Dashboard Layout Engine
AI-optimized layout generation for KPIs (Q5: Approved algorithm)
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from dashboard.models.kpi_models import KPI
from dashboard.models.dashboard_models import Dashboard

logger = logging.getLogger(__name__)


@dataclass
class LayoutCell:
    """Represents a single cell in the dashboard grid"""
    kpi_id: str
    width: int  # 1 = half width, 2 = full width
    height: int  # Relative height units


class DashboardLayoutEngine:
    """
    Generates optimal dashboard layouts
    
    Layout algorithm (APPROVED - Q5):
    1. Row 1: Metric cards (up to 3, 1/3 width each)
    2. Remaining rows: Charts
       - Simple charts (bar, pie): 2 per row (1/2 width)
       - Complex charts (line, table): Full width
    """
    
    # Layout constants
    GRID_COLUMNS = 12  # Bootstrap-style 12-column grid
    METRIC_CARD_COLS = 4  # Each metric takes 4 cols (3 per row)
    HALF_WIDTH_COLS = 6  # Half-width charts
    FULL_WIDTH_COLS = 12  # Full-width charts
    
    def __init__(self):
        """Initialize layout engine"""
        logger.info("Dashboard layout engine initialized")
    
    def generate_layout(
        self,
        kpis: List[KPI],
        dashboard_width: int = 1366  # Desktop width (approved)
    ) -> List[List[str]]:
        """
        Generate optimal layout for KPIs
        
        Args:
            kpis: List of KPI objects
            dashboard_width: Target dashboard width in pixels
            
        Returns:
            2D layout grid: [[kpi_id1, kpi_id2], [kpi_id3], ...]
        """
        if not kpis:
            return []
        
        layout = []
        
        # Step 1: Separate KPIs by type
        metric_kpis = [kpi for kpi in kpis if kpi.chart_type == 'metric']
        line_kpis = [kpi for kpi in kpis if kpi.chart_type == 'line']
        table_kpis = [kpi for kpi in kpis if kpi.chart_type == 'table']
        simple_kpis = [
            kpi for kpi in kpis 
            if kpi.chart_type in ['bar', 'pie']
        ]
        
        # Step 2: Row 1 - Metric cards (up to 3)
        if metric_kpis:
            metric_row = [kpi.id for kpi in metric_kpis[:3]]
            if metric_row:
                layout.append(metric_row)
            
            # If more than 3 metrics, add additional row(s)
            remaining_metrics = metric_kpis[3:]
            while remaining_metrics:
                batch = [kpi.id for kpi in remaining_metrics[:3]]
                layout.append(batch)
                remaining_metrics = remaining_metrics[3:]
        
        # Step 3: Remaining KPIs - Mixed strategy
        # Priority: Line charts (full width) → Simple charts (2 per row) → Tables (full width)
        
        # Add line charts (full width)
        for kpi in line_kpis:
            layout.append([kpi.id])
        
        # Add simple charts (2 per row)
        current_row = []
        for kpi in simple_kpis:
            current_row.append(kpi.id)
            
            if len(current_row) == 2:
                layout.append(current_row)
                current_row = []
        
        # Add remaining simple chart if odd number
        if current_row:
            layout.append(current_row)
        
        # Add tables (full width)
        for kpi in table_kpis:
            layout.append([kpi.id])
        
        logger.info(f"Generated layout: {len(layout)} rows for {len(kpis)} KPIs")
        return layout
    
    def optimize_layout(
        self,
        layout: List[List[str]],
        kpis_map: Dict[str, KPI]
    ) -> List[List[str]]:
        """
        Optimize layout for better visual balance
        
        Args:
            layout: Initial layout
            kpis_map: Map of kpi_id -> KPI object
            
        Returns:
            Optimized layout
        """
        # Optimization strategies:
        # 1. Avoid single items in row if possible
        # 2. Group related KPIs (same category)
        # 3. Alternate between full-width and split rows
        
        optimized = []
        
        for row in layout:
            # If single item in row (not metric, not intentionally full-width)
            if len(row) == 1:
                kpi_id = row[0]
                kpi = kpis_map.get(kpi_id)
                
                if kpi and kpi.chart_type not in ['metric', 'line', 'table']:
                    # This could potentially be paired
                    # For now, keep as is (future enhancement)
                    pass
            
            optimized.append(row)
        
        return optimized
    
    def get_streamlit_columns(self, row: List[str]) -> List[int]:
        """
        Get Streamlit column widths for a row
        
        Args:
            row: List of KPI IDs in row
            
        Returns:
            List of column widths (integers)
        """
        num_items = len(row)
        
        if num_items == 0:
            return []
        elif num_items == 1:
            return [1]  # Full width
        elif num_items == 2:
            return [1, 1]  # Equal split
        elif num_items == 3:
            return [1, 1, 1]  # Three equal parts (for metrics)
        else:
            # More than 3: distribute evenly
            return [1] * num_items
    
    def calculate_row_height(
        self,
        row: List[str],
        kpis_map: Dict[str, KPI]
    ) -> int:
        """
        Calculate appropriate height for a row
        
        Args:
            row: List of KPI IDs in row
            kpis_map: Map of kpi_id -> KPI object
            
        Returns:
            Height in pixels
        """
        if not row:
            return 0
        
        # Get chart types in row
        chart_types = []
        for kpi_id in row:
            kpi = kpis_map.get(kpi_id)
            if kpi:
                chart_types.append(kpi.chart_type)
        
        # Determine height based on chart types
        if 'metric' in chart_types:
            return 200  # Metric cards are shorter
        elif 'table' in chart_types:
            return 500  # Tables need more space
        else:
            return 400  # Standard chart height
    
    def validate_layout(
        self,
        layout: List[List[str]],
        kpis: List[KPI]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that layout is correct
        
        Args:
            layout: Layout to validate
            kpis: List of KPIs
            
        Returns:
            (is_valid, error_message)
        """
        kpi_ids = {kpi.id for kpi in kpis}
        layout_ids = set()
        
        for row in layout:
            for kpi_id in row:
                if kpi_id in layout_ids:
                    return False, f"Duplicate KPI in layout: {kpi_id}"
                
                if kpi_id not in kpi_ids:
                    return False, f"Unknown KPI in layout: {kpi_id}"
                
                layout_ids.add(kpi_id)
        
        # Check all KPIs are in layout
        missing = kpi_ids - layout_ids
        if missing:
            return False, f"KPIs missing from layout: {missing}"
        
        return True, None
    
    def get_layout_description(
        self,
        layout: List[List[str]],
        kpis_map: Dict[str, KPI]
    ) -> str:
        """
        Get human-readable layout description
        
        Args:
            layout: Dashboard layout
            kpis_map: Map of kpi_id -> KPI object
            
        Returns:
            Description string
        """
        lines = []
        lines.append(f"Dashboard Layout ({len(layout)} rows):")
        lines.append("")
        
        for i, row in enumerate(layout, 1):
            row_kpis = [kpis_map.get(kpi_id) for kpi_id in row if kpi_id in kpis_map]
            
            if len(row) == 1:
                kpi = row_kpis[0]
                lines.append(f"Row {i}: [{kpi.chart_type.upper()}] {kpi.name} (full width)")
            else:
                kpi_names = [f"[{kpi.chart_type.upper()}] {kpi.name}" for kpi in row_kpis]
                lines.append(f"Row {i}: {' | '.join(kpi_names)}")
        
        return "\n".join(lines)


class ResponsiveLayoutEngine(DashboardLayoutEngine):
    """
    Extended layout engine with responsive capabilities
    (Not in MVP - desktop only approved, but here for future)
    """
    
    BREAKPOINTS = {
        'mobile': 768,
        'tablet': 1024,
        'desktop': 1366,
        'large': 1920
    }
    
    def generate_responsive_layout(
        self,
        kpis: List[KPI],
        breakpoint: str = 'desktop'
    ) -> List[List[str]]:
        """
        Generate layout for specific breakpoint
        
        Args:
            kpis: List of KPIs
            breakpoint: 'mobile', 'tablet', 'desktop', or 'large'
            
        Returns:
            Layout optimized for breakpoint
        """
        # For MVP: Always use desktop layout
        return self.generate_layout(kpis, self.BREAKPOINTS[breakpoint])


class LayoutPresets:
    """
    Pre-defined layout templates
    """
    
    @staticmethod
    def executive_summary(kpis: List[KPI]) -> List[List[str]]:
        """
        Executive summary layout:
        - Top row: 3-4 key metrics
        - Middle: 1-2 trend charts
        - Bottom: Supporting details
        """
        metrics = [kpi for kpi in kpis if kpi.chart_type == 'metric']
        charts = [kpi for kpi in kpis if kpi.chart_type in ['line', 'bar']]
        tables = [kpi for kpi in kpis if kpi.chart_type == 'table']
        
        layout = []
        
        # Top: Metrics
        if metrics:
            layout.append([kpi.id for kpi in metrics[:4]])
        
        # Middle: Charts
        for chart in charts[:2]:
            layout.append([chart.id])
        
        # Bottom: Tables
        for table in tables[:2]:
            layout.append([table.id])
        
        return layout
    
    @staticmethod
    def detailed_analysis(kpis: List[KPI]) -> List[List[str]]:
        """
        Detailed analysis layout:
        - All KPIs at full width for maximum detail
        """
        return [[kpi.id] for kpi in kpis]
    
    @staticmethod
    def comparison_view(kpis: List[KPI]) -> List[List[str]]:
        """
        Comparison layout:
        - All charts side-by-side for easy comparison
        """
        layout = []
        current_row = []
        
        for kpi in kpis:
            current_row.append(kpi.id)
            
            if len(current_row) == 2:
                layout.append(current_row)
                current_row = []
        
        if current_row:
            layout.append(current_row)
        
        return layout


# Convenience function
def generate_dashboard_layout(kpis: List[KPI]) -> List[List[str]]:
    """
    Quick function to generate layout
    
    Args:
        kpis: List of KPI objects
        
    Returns:
        Layout grid
    """
    engine = DashboardLayoutEngine()
    return engine.generate_layout(kpis)
