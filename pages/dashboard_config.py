"""
Dashboard Configuration Page
Allows users to create and configure dashboards with KPI selection
"""

import streamlit as st
import logging
from typing import List, Optional, Dict
from datetime import datetime

from dashboard.models.kpi_models import KPI
from dashboard.models.dashboard_models import Dashboard
from dashboard.models.schema_models import BusinessContext, SchemaMetadata
from dashboard.storage import get_storage
from dashboard.schema_analyzer import SchemaAnalyzer
from dashboard.kpi_engine import KPIEngine
from dashboard.dashboard_layout_engine import DashboardLayoutEngine
from dashboard.business_context_ui import business_context_widget
from common import DatabaseProps

logger = logging.getLogger(__name__)


def show_dashboard_config_page():
    """Main dashboard configuration page"""
    
    st.set_page_config(
        page_title="Dashboard Configuration - ChatDB",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Dashboard Configuration")
    st.markdown("Create and configure your custom dashboard")
    
    # Check if databases exist
    if 'databases' not in st.session_state or not st.session_state.databases:
        st.warning("⚠️ No databases configured. Please add a database connection first.")
        if st.button("Go to Database Settings"):
            st.switch_page("pages/settings.py")
        return
    
    # Initialize storage
    storage = get_storage()
    
    # Sidebar: Navigation
    with st.sidebar:
        st.header("Navigation")
        
        mode = st.radio(
            "Select Mode",
            options=["Create New Dashboard", "Edit Existing Dashboard"],
            key="config_mode"
        )
        
        st.divider()
        
        # Show existing dashboards
        if mode == "Edit Existing Dashboard":
            existing_dashboards = storage.list_dashboards()
            if existing_dashboards:
                st.subheader("Your Dashboards")
                for dashboard_id in existing_dashboards:
                    dashboard = storage.load_dashboard(dashboard_id)
                    if dashboard:
                        if st.button(
                            f"📊 {dashboard.name}",
                            key=f"load_{dashboard_id}",
                            use_container_width=True
                        ):
                            st.session_state.editing_dashboard_id = dashboard_id
                            st.rerun()
            else:
                st.info("No dashboards yet")
    
    # Main content area
    if mode == "Create New Dashboard":
        show_create_dashboard_flow()
    else:
        show_edit_dashboard_flow()


def show_create_dashboard_flow():
    """Flow for creating a new dashboard"""
    
    # Initialize session state for wizard
    if 'dashboard_wizard_step' not in st.session_state:
        st.session_state.dashboard_wizard_step = 1
    
    # Progress indicator
    steps = ["Database & Context", "Generate KPIs", "Select KPIs", "Configure & Save"]
    current_step = st.session_state.dashboard_wizard_step
    
    st.progress(current_step / len(steps))
    st.caption(f"Step {current_step}/{len(steps)}: {steps[current_step - 1]}")
    
    st.divider()
    
    # Route to appropriate step
    if current_step == 1:
        show_step1_database_context()
    elif current_step == 2:
        show_step2_generate_kpis()
    elif current_step == 3:
        show_step3_select_kpis()
    elif current_step == 4:
        show_step4_configure_save()


def show_step1_database_context():
    """Step 1: Select database and collect business context"""
    
    st.subheader("Step 1: Database & Business Context")
    
    # Database selection
    st.markdown("### Select Database")
    
    database_options = list(st.session_state.databases.keys())
    
    if not database_options:
        st.error("No databases available")
        return
    
    selected_db = st.selectbox(
        "Database",
        options=database_options,
        key="selected_database",
        help="Select the database to create dashboard for"
    )
    
    # Show database info
    if selected_db:
        db_props: DatabaseProps = st.session_state.databases[selected_db]      
        with st.expander("📋 Database Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Database Type", db_props.db_type)
            with col2:
                st.metric("Connection", "Active" if db_props.uri else "Not configured")
    
    st.divider()
    
    # Business context collection
    st.markdown("### Business Context")
    st.info("Provide context about your business to help generate relevant KPIs")
    context = business_context_widget(
        database_id=selected_db,
        key="dashboard_creation_context"
    )
    
    # Navigation
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col3:
        if context:
            if st.button("Continue to KPI Generation →", type="primary", use_container_width=True):
                st.session_state.dashboard_wizard_step = 2
                st.session_state.wizard_database = selected_db
                st.session_state.wizard_context = context
                st.rerun()
        else:
            st.button("Continue to KPI Generation →", disabled=True, use_container_width=True)
            st.caption("Complete business context to continue")


def show_step2_generate_kpis():
    """Step 2: Generate KPI suggestions"""
    
    st.subheader("Step 2: Generate KPI Suggestions")
    
    # Retrieve saved context
    database_id = st.session_state.get('wizard_database')
    context: BusinessContext = st.session_state.get('wizard_context')
    
    if not database_id or not context:
        st.error("Missing required information. Please restart from Step 1.")
        if st.button("← Back to Step 1"):
            st.session_state.dashboard_wizard_step = 1
            st.rerun()
        return
    
    # Show summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Database", database_id)
    with col2:
        st.metric("Industry", context.industry)
    
    st.divider()
    
    # Check if already generated
    if 'wizard_kpis' in st.session_state and st.session_state.wizard_kpis:
        st.success(f"✓ {len(st.session_state.wizard_kpis)} KPIs generated previously")
        
        if st.button("🔄 Regenerate KPIs"):
            del st.session_state.wizard_kpis
            del st.session_state.wizard_metadata
            st.rerun()
        
        # Skip to next step
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.dashboard_wizard_step = 1
                st.rerun()
        with col3:
            if st.button("Continue to KPI Selection →", type="primary", use_container_width=True):
                st.session_state.dashboard_wizard_step = 3
                st.rerun()
        return
    
    # Generate KPIs button
    st.markdown("### Generate KPI Suggestions")
    st.info("This will analyze your database schema and generate relevant KPI suggestions based on your business context. This may take 1-2 minutes.")
    
    if st.button("🚀 Analyze Schema & Generate KPIs", type="primary", use_container_width=True):
        run_kpi_generation(database_id, context)


def run_kpi_generation(database_id: str, context: BusinessContext):
    """Execute KPI generation process"""
    # Get database URI
    db_props: DatabaseProps = st.session_state.databases[database_id]
    database_uri = db_props.uri
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Analyze schema
        status_text.info("📊 Analyzing database schema...")
        progress_bar.progress(0.1)
        
        analyzer = SchemaAnalyzer(database_uri)
        
        if not analyzer.connect():
            st.error("❌ Failed to connect to database")
            return
        
        def schema_progress(current, total, message):
            progress = 0.1 + (0.3 * current / total)
            progress_bar.progress(progress)
            status_text.info(f"📊 {message}")
        
        metadata = analyzer.analyze_database(
            database_id=database_id,
            progress_callback=schema_progress
        )
        
        analyzer.close()
        
        if not metadata:
            st.error("❌ Schema analysis failed")
            return
        
        st.success(f"✓ Analyzed {metadata.get_table_count()} tables")
        progress_bar.progress(0.4)
        
        # Step 2: Generate KPIs
        status_text.info("🤖 Generating KPI suggestions with AI...")
        progress_bar.progress(0.5)
        engine = KPIEngine()
        
        def kpi_progress(current, total, message):
            progress = 0.5 + (0.4 * current / total)
            progress_bar.progress(progress)
            status_text.info(f"🤖 {message}")

        result = engine.suggest_kpis(
            schema_metadata=metadata,
            business_context=context,
            database_uri=database_uri,
            progress_callback=kpi_progress
        )
        
        # Store results
        st.session_state.wizard_kpis = result['kpis']
        st.session_state.wizard_metadata = metadata
        st.session_state.wizard_validation_failures = result['validation_failures']
        
        progress_bar.progress(1.0)
        status_text.success(
            f"✅ Generated {len(result['kpis'])} valid KPIs "
            f"(from {result['total_suggested']} suggestions)"
        )
        
        # Show summary
        if result['validation_failures']:
            with st.expander(f"⚠️ {len(result['validation_failures'])} KPIs failed validation"):
                for failure in result['validation_failures'][:5]:
                    st.caption(f"• {failure['name']}: {failure['error'][:100]}")
        
        # Auto-advance after 2 seconds
        import time
        time.sleep(2)
        st.session_state.dashboard_wizard_step = 3
        st.rerun()
        
    except Exception as e:
        logger.error(f"KPI generation failed: {e}")
        st.error(f"❌ KPI generation failed: {e}")
        progress_bar.empty()
        status_text.empty()


def show_step3_select_kpis():
    """Step 3: Select KPIs for dashboard"""
    
    st.subheader("Step 3: Select KPIs")
    
    # Check if KPIs exist
    if 'wizard_kpis' not in st.session_state or not st.session_state.wizard_kpis:
        st.warning("No KPIs available. Please generate KPIs first.")
        if st.button("← Back to Generate KPIs"):
            st.session_state.dashboard_wizard_step = 2
            st.rerun()
        return
    
    kpis: List[KPI] = st.session_state.wizard_kpis
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total KPIs", len(kpis))
    with col2:
        categories = set(kpi.category for kpi in kpis)
        st.metric("Categories", len(categories))
    with col3:
        chart_types = set(kpi.chart_type for kpi in kpis)
        st.metric("Chart Types", len(chart_types))
    
    st.divider()
    
    # Selection mode
    selection_mode = st.radio(
        "Selection Mode",
        options=["Select All", "Select by Category", "Select Individually"],
        horizontal=True
    )
    
    # Initialize selection
    if 'selected_kpi_ids' not in st.session_state:
        st.session_state.selected_kpi_ids = []
    
    if selection_mode == "Select All":
        show_select_all_mode(kpis)
    elif selection_mode == "Select by Category":
        show_select_by_category_mode(kpis)
    else:
        show_select_individual_mode(kpis)
    
    # Navigation
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.dashboard_wizard_step = 2
            st.rerun()
    
    with col3:
        if st.session_state.selected_kpi_ids:
            if st.button(
                f"Continue with {len(st.session_state.selected_kpi_ids)} KPIs →",
                type="primary",
                use_container_width=True
            ):
                st.session_state.dashboard_wizard_step = 4
                st.rerun()
        else:
            st.button("Continue →", disabled=True, use_container_width=True)
            st.caption("Select at least one KPI to continue")


def show_select_all_mode(kpis: List[KPI]):
    """Select all KPIs mode"""
    
    st.info(f"All {len(kpis)} KPIs will be included in your dashboard")
    
    if st.button("Select All KPIs", use_container_width=True):
        st.session_state.selected_kpi_ids = [kpi.id for kpi in kpis]
        st.success(f"✓ Selected all {len(kpis)} KPIs")
        st.rerun()
    
    # Preview
    with st.expander("Preview All KPIs"):
        for kpi in kpis[:10]:
            st.markdown(f"**{kpi.name}** ({kpi.category}) - {kpi.chart_type}")


def show_select_by_category_mode(kpis: List[KPI]):
    """Select by category mode"""
    
    # Group by category
    by_category: Dict[str, List[KPI]] = {}
    for kpi in kpis:
        if kpi.category not in by_category:
            by_category[kpi.category] = []
        by_category[kpi.category].append(kpi)
    
    st.markdown("### Select Categories")
    
    for category, category_kpis in by_category.items():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected = st.checkbox(
                f"**{category.title()}** ({len(category_kpis)} KPIs)",
                key=f"category_{category}",
                value=any(kpi.id in st.session_state.selected_kpi_ids for kpi in category_kpis)
            )
        
        with col2:
            with st.expander("View"):
                for kpi in category_kpis:
                    st.caption(f"• {kpi.name}")
        
        if selected:
            for kpi in category_kpis:
                if kpi.id not in st.session_state.selected_kpi_ids:
                    st.session_state.selected_kpi_ids.append(kpi.id)
        else:
            for kpi in category_kpis:
                if kpi.id in st.session_state.selected_kpi_ids:
                    st.session_state.selected_kpi_ids.remove(kpi.id)


def show_select_individual_mode(kpis: List[KPI]):
    """Select individual KPIs mode"""
    
    st.markdown("### Select Individual KPIs")
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        filter_category = st.multiselect(
            "Filter by Category",
            options=list(set(kpi.category for kpi in kpis)),
            default=[]
        )
    
    with col2:
        filter_chart = st.multiselect(
            "Filter by Chart Type",
            options=list(set(kpi.chart_type for kpi in kpis)),
            default=[]
        )
    
    # Apply filters
    filtered_kpis = kpis
    if filter_category:
        filtered_kpis = [kpi for kpi in filtered_kpis if kpi.category in filter_category]
    if filter_chart:
        filtered_kpis = [kpi for kpi in filtered_kpis if kpi.chart_type in filter_chart]
    
    st.caption(f"Showing {len(filtered_kpis)} of {len(kpis)} KPIs")
    
    # Display KPIs
    for kpi in filtered_kpis:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                selected = st.checkbox(
                    f"**{kpi.name}**",
                    key=f"kpi_{kpi.id}",
                    value=kpi.id in st.session_state.selected_kpi_ids
                )
                st.caption(kpi.description[:100] + "..." if len(kpi.description) > 100 else kpi.description)
            
            with col2:
                st.caption(f"📊 {kpi.chart_type}")
                st.caption(f"🏷️ {kpi.category}")
            
            with col3:
                with st.expander("SQL"):
                    st.code(kpi.sql_template[:200] + "...", language="sql")
            
            if selected and kpi.id not in st.session_state.selected_kpi_ids:
                st.session_state.selected_kpi_ids.append(kpi.id)
            elif not selected and kpi.id in st.session_state.selected_kpi_ids:
                st.session_state.selected_kpi_ids.remove(kpi.id)
            
            st.divider()


def show_step4_configure_save():
    """Step 4: Configure dashboard and save"""
    
    st.subheader("Step 4: Configure & Save Dashboard")
    
    # Get selected KPIs
    all_kpis: List[KPI] = st.session_state.wizard_kpis
    selected_ids = st.session_state.selected_kpi_ids
    selected_kpis = [kpi for kpi in all_kpis if kpi.id in selected_ids]
    
    if not selected_kpis:
        st.warning("No KPIs selected")
        if st.button("← Back to Selection"):
            st.session_state.dashboard_wizard_step = 3
            st.rerun()
        return
    
    # Dashboard configuration
    st.markdown("### Dashboard Details")
    
    dashboard_name = st.text_input(
        "Dashboard Name *",
        value="My Dashboard",
        max_chars=100,
        help="Give your dashboard a descriptive name"
    )
    
    dashboard_description = st.text_area(
        "Description (optional)",
        max_chars=500,
        help="Optional description for this dashboard"
    )
    
    st.divider()
    
    # Preview layout
    st.markdown("### Layout Preview")
    
    layout_engine = DashboardLayoutEngine()
    preview_layout = layout_engine.generate_layout(selected_kpis)
    
    kpis_map = {kpi.id: kpi for kpi in selected_kpis}
    layout_description = layout_engine.get_layout_description(preview_layout, kpis_map)
    
    with st.expander("📐 View Layout Structure", expanded=True):
        st.code(layout_description, language="text")
    
    st.info(f"Dashboard will have {len(preview_layout)} rows with {len(selected_kpis)} KPIs")
    
    st.divider()
    
    # Save dashboard
    st.markdown("### Save Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.dashboard_wizard_step = 3
            st.rerun()
    
    with col3:
        if st.button("💾 Save Dashboard", type="primary", use_container_width=True):
            save_dashboard(
                name=dashboard_name,
                description=dashboard_description,
                selected_kpis=selected_kpis,
                layout=preview_layout
            )


def save_dashboard(
    name: str,
    description: Optional[str],
    selected_kpis: List[KPI],
    layout: List[List[str]]
):
    """Save dashboard to storage"""
    
    try:
        storage = get_storage()
        
        # Save all KPIs first
        for kpi in selected_kpis:
            storage.save_kpi(kpi)
        
        # Create dashboard
        database_id = st.session_state.wizard_database
        dashboard = Dashboard(
            name=name,
            database_id=database_id,
            kpi_ids=[kpi.id for kpi in selected_kpis],
            layout=layout
        )
        
        # Validate
        is_valid, error = dashboard.validate()
        if not is_valid:
            st.error(f"Dashboard validation failed: {error}")
            return
        
        # Save
        success = storage.save_dashboard(dashboard)
        
        if success:
            st.success(f"✅ Dashboard '{name}' saved successfully!")
            
            # # Clear wizard state
            # for key in list(st.session_state.keys()):
            #     if key.startswith('wizard_') or key.startswith('dashboard_wizard_'):
            #         del st.session_state[key]
            
            # Offer to view dashboard
            st.balloons()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Dashboard", type="primary", use_container_width=True):
                    st.session_state.current_dashboard_id = dashboard.id
                    st.switch_page("pages/dashboard_view.py")
            
            with col2:
                if st.button("➕ Create Another Dashboard", use_container_width=True):
                    st.rerun()
        else:
            st.error("Failed to save dashboard")
            
    except Exception as e:
        logger.error(f"Failed to save dashboard: {e}")
        st.error(f"Failed to save dashboard: {e}")


def show_edit_dashboard_flow():
    """Flow for editing existing dashboard"""
    
    st.subheader("Edit Dashboard")
    
    # Check if dashboard selected
    if 'editing_dashboard_id' not in st.session_state:
        st.info("Select a dashboard from the sidebar to edit")
        return
    
    dashboard_id = st.session_state.editing_dashboard_id
    storage = get_storage()
    dashboard = storage.load_dashboard(dashboard_id)
    
    if not dashboard:
        st.error("Dashboard not found")
        del st.session_state.editing_dashboard_id
        return
    
    # Dashboard info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric("Dashboard", dashboard.name)
    with col2:
        st.metric("KPIs", len(dashboard.kpi_ids))
    with col3:
        st.metric("Database", dashboard.database_id)
    
    st.divider()
    
    # Edit tabs
    tab1, tab2, tab3 = st.tabs(["Details", "KPIs", "Delete"])
    
    with tab1:
        show_edit_details_tab(dashboard)
    
    with tab2:
        show_edit_kpis_tab(dashboard)
    
    with tab3:
        show_delete_dashboard_tab(dashboard)


def show_edit_details_tab(dashboard: Dashboard):
    """Edit dashboard details"""
    
    st.markdown("### Dashboard Details")
    
    new_name = st.text_input(
        "Dashboard Name",
        value=dashboard.name,
        max_chars=100
    )
    
    if st.button("Update Name", type="primary"):
        dashboard.name = new_name
        dashboard.updated_at = datetime.now()
        
        storage = get_storage()
        if storage.save_dashboard(dashboard):
            st.success("Dashboard updated!")
            st.rerun()
        else:
            st.error("Failed to update dashboard")


def show_edit_kpis_tab(dashboard: Dashboard):
    """Edit dashboard KPIs"""
    
    st.markdown("### Manage KPIs")
    
    storage = get_storage()
    kpis = [storage.load_kpi(kpi_id) for kpi_id in dashboard.kpi_ids]
    kpis = [k for k in kpis if k]
    
    # Display current KPIs
    for kpi in kpis:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{kpi.name}**")
            st.caption(f"{kpi.category} • {kpi.chart_type}")
        
        with col2:
            with st.expander("SQL"):
                st.code(kpi.sql_template[:100] + "...", language="sql")
        
        with col3:
            if st.button("🗑️ Remove", key=f"remove_{kpi.id}"):
                dashboard.remove_kpi(kpi.id)
                storage.save_dashboard(dashboard)
                st.rerun()
        
        st.divider()
    
    # Add KPI option
    st.markdown("### Add KPI")
    st.info("Adding custom KPIs from chat will be available in the main chat interface")


def show_delete_dashboard_tab(dashboard: Dashboard):
    """Delete dashboard"""
    
    st.markdown("### Delete Dashboard")
    st.warning("⚠️ This action cannot be undone!")
    
    st.markdown(f"You are about to delete: **{dashboard.name}**")
    st.caption(f"Created: {dashboard.created_at}")
    st.caption(f"KPIs: {len(dashboard.kpi_ids)}")
    
    confirm = st.checkbox("I understand this action is permanent")
    
    if confirm:
        if st.button("🗑️ Delete Dashboard", type="primary"):
            storage = get_storage()
            if storage.delete_dashboard(dashboard.id):
                st.success("Dashboard deleted")
                del st.session_state.editing_dashboard_id
                st.rerun()
            else:
                st.error("Failed to delete dashboard")


# Main entry point
if __name__ == "__main__":
    show_dashboard_config_page()
