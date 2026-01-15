"""
Business Context UI Component
Collects business context for KPI generation
"""

import streamlit as st
from typing import Optional, Dict
from dashboard.models.schema_models import BusinessContext


# Industry options (as approved)
INDUSTRIES = [
    "E-commerce / Retail",
    "SaaS / Subscription",
    "Manufacturing",
    "Financial Services",
    "Healthcare",
    "Marketing / Advertising",
    "Logistics / Supply Chain",
    "Other"
]

# Business model options (as approved)
BUSINESS_MODELS = [
    "B2C",
    "B2B",
    "B2B2C",
    "Marketplace"
]

# Primary goals options (as approved)
PRIMARY_GOALS = [
    "Increase Revenue",
    "Reduce Costs",
    "Improve Customer Satisfaction",
    "Optimize Operations"
]


def show_business_context_form(
    database_id: str,
    existing_context: Optional[BusinessContext] = None
) -> Optional[BusinessContext]:
    """
    Display business context collection form
    
    Args:
        database_id: Database identifier
        existing_context: Existing context to pre-fill (optional)
        
    Returns:
        BusinessContext if form submitted, None otherwise
    """
    
    st.subheader("📊 Business Context")
    
    st.info("""
    Help us understand your business to generate relevant KPIs. 
    This information guides the AI in suggesting metrics that matter to your specific use case.
    """)
    
    # Pre-fill values if editing existing context
    default_industry = existing_context.industry if existing_context else INDUSTRIES[0]
    default_model = existing_context.business_model if existing_context else BUSINESS_MODELS[0]
    default_goals = existing_context.primary_goals if existing_context else []
    default_other = existing_context.other if existing_context else ""
    
    with st.form(key=f"business_context_form_{database_id}"):
        # Industry (required)
        industry = st.selectbox(
            "Industry *",
            options=INDUSTRIES,
            index=INDUSTRIES.index(default_industry) if default_industry in INDUSTRIES else 0,
            help="What industry does your business operate in?"
        )
        
        # If "Other" selected, show text input
        other_industry = ""
        if industry == "Other":
            other_industry = st.text_input(
                "Specify your industry",
                value=default_other if default_industry == "Other" else "",
                max_chars=100,
                placeholder="e.g., Real Estate, Education, Gaming",
                help="Maximum 100 characters"
            )
        
        # Business Model (required)
        business_model = st.selectbox(
            "Business Model *",
            options=BUSINESS_MODELS,
            index=BUSINESS_MODELS.index(default_model) if default_model in BUSINESS_MODELS else 0,
            help="How do you primarily do business?"
        )
        
        # Primary Goals (required, multi-select)
        primary_goals = st.multiselect(
            "Primary Business Goals *",
            options=PRIMARY_GOALS,
            default=default_goals,
            help="Select all that apply (at least one required)"
        )
        
        # Additional context (optional)
        st.markdown("**Additional Context** (optional)")
        other_context = st.text_area(
            "Any other information that might help generate relevant KPIs?",
            value=default_other if default_industry != "Other" else "",
            max_chars=100,
            placeholder="e.g., Focus on subscription retention, expanding to new markets, etc.",
            help="Maximum 100 characters",
            label_visibility="collapsed"
        )
        
        # Required field indicator
        st.caption("* Required fields")
        
        # Submit button
        col1, col2 = st.columns([1, 3])
        with col1:
            submitted = st.form_submit_button(
                "Continue",
                type="primary",
                use_container_width=True
            )
        with col2:
            if existing_context:
                st.caption("Updating existing context")
        
        # Validation and submission
        if submitted:
            # Validate required fields
            if not industry:
                st.error("Industry is required")
                return None
            
            if industry == "Other" and not other_industry.strip():
                st.error("Please specify your industry")
                return None
            
            if not business_model:
                st.error("Business model is required")
                return None
            
            if not primary_goals:
                st.error("Please select at least one primary goal")
                return None
            
            # Create BusinessContext
            final_industry = other_industry.strip() if industry == "Other" else industry
            final_other = other_context.strip() if other_context else None
            
            context = BusinessContext(
                industry=final_industry,
                business_model=business_model,
                primary_goals=primary_goals,
                other=final_other
            )
            
            return context
    
    return None


def display_business_context_summary(context: BusinessContext):
    """
    Display a summary of the business context
    
    Args:
        context: BusinessContext to display
    """
    st.markdown("### 📋 Business Context Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Industry", context.industry)
        st.metric("Business Model", context.business_model)
    
    with col2:
        st.markdown("**Primary Goals:**")
        for goal in context.primary_goals:
            st.markdown(f"- {goal}")
    
    if context.other:
        st.info(f"**Additional Context:** {context.other}")


def get_or_create_business_context(
    database_id: str,
    force_new: bool = False
) -> Optional[BusinessContext]:
    """
    Get existing business context or create new one
    
    Args:
        database_id: Database identifier
        force_new: Force creation of new context
        
    Returns:
        BusinessContext or None
    """
    session_key = f"business_context_{database_id}"
    
    # Check if context already exists in session
    if not force_new and session_key in st.session_state:
        return st.session_state[session_key]
    
    # Show form to create new context
    context = show_business_context_form(
        database_id,
        existing_context=st.session_state.get(session_key)
    )
    
    # Save to session if created
    if context:
        st.session_state[session_key] = context
    
    return context


def edit_business_context(database_id: str) -> Optional[BusinessContext]:
    """
    Edit existing business context
    
    Args:
        database_id: Database identifier
        
    Returns:
        Updated BusinessContext or None
    """
    session_key = f"business_context_{database_id}"
    existing = st.session_state.get(session_key)
    
    st.markdown("### ✏️ Edit Business Context")
    
    if existing:
        display_business_context_summary(existing)
        st.divider()
    
    # Show form
    updated_context = show_business_context_form(database_id, existing)
    
    if updated_context:
        st.session_state[session_key] = updated_context
        st.success("Business context updated!")
        st.rerun()
    
    return updated_context


def clear_business_context(database_id: str):
    """
    Clear business context for a database
    
    Args:
        database_id: Database identifier
    """
    session_key = f"business_context_{database_id}"
    if session_key in st.session_state:
        del st.session_state[session_key]


# Streamlit component for inline use
def business_context_widget(
    database_id: str,
    key: str = None,
    show_summary: bool = True
) -> Optional[BusinessContext]:
    """
    Inline business context widget
    
    Args:
        database_id: Database identifier
        key: Unique key for widget
        show_summary: Whether to show summary after submission
        
    Returns:
        BusinessContext or None
    """
    widget_key = key or f"bc_widget_{database_id}"
    session_key = f"business_context_{database_id}"
    
    # Check if already have context
    if session_key in st.session_state:
        context = st.session_state[session_key]
        
        if show_summary:
            display_business_context_summary(context)
            
            # Edit button
            if st.button("✏️ Edit", key=f"{widget_key}_edit"):
                del st.session_state[session_key]
                st.rerun()
        
        return context
    
    # Show form
    context = show_business_context_form(database_id)
    
    if context:
        st.session_state[session_key] = context
        st.rerun()
    
    return None
