"""
Dashboard Display Page
Renders dashboards with interactive controls and export functionality
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional, List
from datetime import datetime
import io

from dashboard.models.dashboard_models import Dashboard
from dashboard.models.kpi_models import KPI
from dashboard.storage import get_storage
from dashboard.visualization_engine import VisualizationEngine, ChartStyler
from dashboard.kpi_executor import KPIExecutor
from dashboard.dashboard_layout_engine import DashboardLayoutEngine
from common import DatabaseProps

logger = logging.getLogger(__name__)


def show_dashboard_view_page():
    """Main dashboard view page"""
    st.set_page_config(
        page_title="Dashboard - ChatDB",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar: Dashboard selection
    with st.sidebar:
        st.header("Dashboards")
        
        storage = get_storage()
        dashboards = storage.list_dashboards()
        
        if not dashboards:
            st.info("No dashboards created yet")
            if st.button("➕ Create Dashboard"):
                st.switch_page("pages/dashboard_config.py")
            return
        
        # Load dashboard details
        dashboard_options = {}
        for dash in dashboards:
            dash = storage.load_dashboard(dash.id)
            if dash:
                dashboard_options[dash.id] = dash.name +" | "+ dash.id
        
        # Select dashboard
        selected_id = st.selectbox(
            "Select Dashboard",
            options=list(dashboard_options.keys()),
            format_func=lambda x: dashboard_options[x],
            key="selected_dashboard_id"
        )
        
        st.divider()
        
        # Quick actions
        st.subheader("Actions")
        
        if st.button("➕ New Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard_config.py")
        
        if st.button("✏️ Edit Current", use_container_width=True):
            st.session_state.editing_dashboard_id = selected_id
            st.switch_page("pages/dashboard_config.py")
    
    # Load and display dashboard
    if selected_id:
        render_dashboard(selected_id)


def render_dashboard(dashboard_id: str):
    """Render complete dashboard"""
    storage = get_storage()
    dashboard = storage.load_dashboard(dashboard_id)
    
    if not dashboard:
        st.error("Dashboard not found")
        return
    
    # Header
    st.title(dashboard.name)
    st.caption(f"Database: {dashboard.database_id} | Created: {dashboard.created_at.strftime('%Y-%m-%d')}")
    
    # Controls bar
    show_dashboard_controls(dashboard)
    
    st.divider()
    
    # Load KPIs
    kpis = load_dashboard_kpis(dashboard)
    
    if not kpis:
        st.warning("No KPIs configured for this dashboard")
        if st.button("Add KPIs"):
            st.session_state.editing_dashboard_id = dashboard_id
            st.switch_page("pages/dashboard_config.py")
        return
    
    # Get time range from session state
    time_range = st.session_state.get(f'time_range_{dashboard_id}', 30)
    
    # Execute queries
    results = execute_dashboard_queries(dashboard, kpis, time_range)
    
    # Render dashboard
    render_dashboard_layout(dashboard, kpis, results, time_range)


def show_dashboard_controls(dashboard: Dashboard):
    """Show interactive controls for dashboard"""
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Time range selector
    with col1:
        time_options = {
            7: "Last 7 days",
            30: "Last 30 days",
            90: "Last 90 days",
            365: "Last year",
            1825: "Last 5 year",
            18250: "Last 50 year"
        }
        
        current_range = st.session_state.get(f'time_range_{dashboard.id}', 30)
        
        selected_range = st.selectbox(
            "Time Range",
            options=list(time_options.keys()),
            format_func=lambda x: time_options[x],
            index=list(time_options.keys()).index(current_range),
            key=f"time_selector_{dashboard.id}"
        )
        
        # Store selection
        st.session_state[f'time_range_{dashboard.id}'] = selected_range
    
    # Refresh button
    with col2:
        st.write("")  # Spacer for alignment
        if st.button("🔄 Refresh", use_container_width=True):
            # Clear cache for this dashboard
            st.cache_data.clear()
            st.session_state[f'force_refresh_{dashboard.id}'] = True
            st.rerun()
    
    # Export button
    with col3:
        st.write("")  # Spacer
        if st.button("📥 Export PDF", use_container_width=True):
            export_dashboard_pdf(dashboard)
    
    # Settings
    with col4:
        st.write("")  # Spacer
        with st.popover("⚙️ Settings"):
            show_dashboard_settings(dashboard)


def show_dashboard_settings(dashboard: Dashboard):
    """Show dashboard settings in popover"""
    
    st.markdown("### Display Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Chart Theme",
        options=['plotly_white', 'plotly', 'plotly_dark', 'simple_white'],
        key=f"theme_{dashboard.id}"
    )
    
    st.session_state[f'dashboard_theme_{dashboard.id}'] = theme
    
    # Auto-refresh
    auto_refresh = st.checkbox(
        "Auto-refresh (5 min)",
        key=f"auto_refresh_{dashboard.id}"
    )
    
    if auto_refresh:
        st.caption("Dashboard will refresh automatically")
    
    # Cache info
    st.divider()
    st.markdown("### Cache")
    
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared")


def load_dashboard_kpis(dashboard: Dashboard) -> List[KPI]:
    """Load all KPIs for dashboard"""
    
    storage = get_storage()
    kpis = []
    
    for kpi_id in dashboard.kpi_ids:
        kpi = storage.load_kpi(kpi_id)
        if kpi:
            kpis.append(kpi)
        else:
            logger.warning(f"KPI not found: {kpi_id}")
    
    return kpis


@st.cache_data(ttl=300, show_spinner=False)
def execute_dashboard_queries(
    dashboard: Dashboard,
    kpis: List[KPI],
    time_range: int
) -> dict:
    """
    Execute all KPI queries with caching
    
    Args:
        dashboard: Dashboard object
        kpis: List of KPIs
        time_range: Time range in days
        
    Returns:
        Dictionary mapping kpi_id to result DataFrame
    """
    
    # Get database URI
    if 'databases' not in st.session_state:
        return {}
    
    db_props: DatabaseProps = st.session_state.databases.get(dashboard.database_id)
    if not db_props:
        return {}
    
    # Check if force refresh
    force_refresh = st.session_state.get(f'force_refresh_{dashboard.id}', False)
    if force_refresh:
        st.session_state[f'force_refresh_{dashboard.id}'] = False
    
    # Execute queries
    executor = KPIExecutor(db_props.uri, use_cache=True)
    
    try:
        results = executor.execute_multiple(
            kpis,
            time_range_days=time_range,
            bypass_cache=force_refresh
        )
        return results
    finally:
        executor.close()


def render_dashboard_layout(
    dashboard: Dashboard,
    kpis: List[KPI],
    results: dict,
    time_range: int
):
    """Render dashboard using layout"""
    # Get theme
    theme = st.session_state.get(f'dashboard_theme_{dashboard.id}', 'plotly_white')
    viz_engine = VisualizationEngine(theme=theme)
    # Create KPI map for quick lookup
    kpis_map = {kpi.id: kpi for kpi in kpis}
    
    # Generate layout if not exists
    if not dashboard.layout:
        layout_engine = DashboardLayoutEngine()
        dashboard.layout = layout_engine.generate_layout(kpis)
    
    # Render each row
    for row_idx, row in enumerate(dashboard.layout):
        # Create columns based on number of items in row
        if len(row) == 1:
            # Full width
            cols = st.columns(len(row))
            render_kpi_chart(
                kpis_map[row[0]],
                results.get(row[0], pd.DataFrame()),
                viz_engine,
                time_range,
                container=st.container()
            )
            import pdb; pdb.set_trace()
        else:
            # Multiple columns
            cols = st.columns(len(row))
            
            for col, kpi_id in zip(cols, row):
                kpi = kpis_map.get(kpi_id)
                if kpi:
                    data = results.get(kpi_id, pd.DataFrame())
                    render_kpi_chart(kpi, data, viz_engine, time_range, container=col)
        
        # Add spacing between rows
        if row_idx < len(dashboard.layout) - 1:
            st.markdown("<br>", unsafe_allow_html=True)


def render_kpi_chart(
    kpi: KPI,
    data: pd.DataFrame,
    viz_engine: VisualizationEngine,
    time_range: int,
    container
):
    """Render a single KPI chart"""
    with container:
        try:
            if data.empty:
                st.info(f"📊 {kpi.name}\n\nNo data available")
                return
            
            # Generate chart
            fig = viz_engine.generate_chart(kpi, data, time_range)
            
            # Apply styling
            fig = ChartStyler.optimize_for_dashboard(fig)
            
            # Display
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{kpi.id}")
            
        except Exception as e:
            logger.error(f"Failed to render chart for {kpi.name}: {e}")
            st.error(f"❌ Error loading {kpi.name}")
            with st.expander("Error Details"):
                st.code(str(e))


def export_dashboard_pdf(dashboard: Dashboard):
    """Export dashboard to PDF (A4, current page)"""
    
    st.info("📄 PDF Export")
    st.markdown("""
    ### Export Options
    
    PDF export will capture the current dashboard view as displayed.
    
    **Note**: For best results:
    1. Adjust time range and settings as desired
    2. Wait for all charts to load
    3. Click the export button below
    """)
    
    # Export implementation placeholder
    st.warning("⚠️ PDF export functionality will be implemented using a headless browser or chart-to-image conversion")
    
    # Mockup of export process
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as PDF", type="primary"):
            st.info("Generating PDF...")
            
            # Placeholder: Would use plotly's kaleido or selenium here
            # For now, show what would happen
            st.success("✓ PDF would be generated here")
            
            # Offer download button (mockup)
            st.download_button(
                label="📥 Download PDF",
                data=b"PDF content would be here",
                file_name=f"{dashboard.name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        st.markdown("**What's included:**")
        st.caption("• Current dashboard layout")
        st.caption("• All visible charts")
        st.caption("• Current time range")
        st.caption("• Dashboard name and date")


def show_kpi_details_modal(kpi: KPI, data: pd.DataFrame):
    """Show detailed view of KPI in modal"""
    
    with st.expander(f"📊 {kpi.name} - Details", expanded=False):
        
        # KPI metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Category", kpi.category.title())
        with col2:
            st.metric("Chart Type", kpi.chart_type.title())
        with col3:
            st.metric("Data Points", len(data) if not data.empty else 0)
        
        # Description
        st.markdown("### Description")
        st.write(kpi.description)
        
        # SQL Query
        st.markdown("### SQL Query")
        st.code(kpi.sql_template, language="sql")
        
        # Raw data
        if not data.empty:
            st.markdown("### Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Download data
            csv = data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{kpi.name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def show_empty_dashboard_state():
    """Show empty state when no dashboards"""
    
    st.info("### 📊 No Dashboards Yet")
    
    st.markdown("""
    You haven't created any dashboards yet. Get started by creating your first dashboard!
    
    **What you can do:**
    1. Select a database connection
    2. Provide business context
    3. Generate AI-powered KPI suggestions
    4. Select KPIs for your dashboard
    5. View and interact with your data
    """)
    
    if st.button("➕ Create Your First Dashboard", type="primary"):
        st.switch_page("pages/dashboard_config.py")


def show_error_state(error_message: str):
    """Show error state"""
    
    st.error(f"❌ {error_message}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Try Again"):
            st.rerun()
    
    with col2:
        if st.button("← Back to Dashboards"):
            if 'selected_dashboard_id' in st.session_state:
                del st.session_state.selected_dashboard_id
            st.rerun()


# Auto-refresh functionality
def setup_auto_refresh(dashboard_id: str, interval_minutes: int = 5):
    """Setup auto-refresh for dashboard"""
    
    if st.session_state.get(f'auto_refresh_{dashboard_id}', False):
        # Use Streamlit's experimental_rerun with timer
        import time
        
        # Store last refresh time
        last_refresh_key = f'last_refresh_{dashboard_id}'
        last_refresh = st.session_state.get(last_refresh_key, 0)
        
        current_time = time.time()
        if current_time - last_refresh > interval_minutes * 60:
            st.session_state[last_refresh_key] = current_time
            st.session_state[f'force_refresh_{dashboard.id}'] = True
            st.rerun()


# Main entry point
if __name__ == "__main__":
    show_dashboard_view_page()
