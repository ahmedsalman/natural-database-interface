"""
Enhanced chats interface with SQL safety visualization and error recovery
"""

import json
import re
import warnings
import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from sqlalchemy.exc import DBAPIError, NoSuchColumnError, NoSuchTableError

from agent import get_enhanced_executor
from backup import backup_conversation, load_conversation
from common import Conversation, init_session_state
from multi_database import QueryBlockedError
from config import PRESET_CONFIGS
from error_recovery import (
    ErrorRecoveryEngine, SQLErrorClassifier, ErrorSeverity
)

# Convenience functions
_global_classifier = SQLErrorClassifier()
_global_recovery_engine = ErrorRecoveryEngine(_global_classifier)

warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'expire_cache' was never awaited")

st.set_page_config(
    page_title="Chats",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
init_session_state()

# Initialize safety preset in session state
if "safety_preset" not in st.session_state:
    st.session_state.safety_preset = "production"


def new_chat_button_on_click():
    st.session_state.current_conversation = ""


def set_conversation(conversation_id):
    st.session_state.current_conversation = conversation_id


def retry_chat(prompt: str, stream: bool):
    st.session_state.retry = {"stream": stream, "prompt": prompt}


def conversation_exists(id: str) -> bool:
    return id != "" and id in st.session_state.conversations


def conversation_valid(id: str):
    if conversation_exists(id):
        conversation: Conversation = st.session_state.conversations[id]
        return all([x in st.session_state.databases for x in conversation.database_ids])
    return False


def display_query(database, query, results):
    """Display query with enhanced formatting"""
    with st.expander("🔍 View SQL query..."):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Database:** `{database}`")
        with col2:
            # Syntax highlighting for SQL
            st.code(query, language="sql")
        
        if results:
            st.markdown("**Results:**")
            st.dataframe(results, use_container_width=True)
        else:
            st.info("No results returned")


def display_error_context(error_ctx):
    """Display detailed error context with recovery information"""
    severity_icons = {
        ErrorSeverity.LOW: "ℹ️",
        ErrorSeverity.MEDIUM: "⚠️",
        ErrorSeverity.HIGH: "🚨",
        ErrorSeverity.CRITICAL: "🔴"
    }
    
    severity_colors = {
        ErrorSeverity.LOW: "blue",
        ErrorSeverity.MEDIUM: "orange",
        ErrorSeverity.HIGH: "red",
        ErrorSeverity.CRITICAL: "red"
    }
    
    icon = severity_icons.get(error_ctx.severity, "❓")
    
    with st.expander(f"{icon} Error Details - {error_ctx.category.value.upper()}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Error Classification:**")
            st.markdown(f"- **Type:** {error_ctx.error_type}")
            st.markdown(f"- **Category:** {error_ctx.category.value}")
            st.markdown(f"- **Severity:** :{severity_colors[error_ctx.severity]}[{error_ctx.severity.name}]")
            st.markdown(f"- **Retry Count:** {error_ctx.retry_count}")
        
        with col2:
            st.markdown("**Recovery Strategy:**")
            st.markdown(f"- **Strategy:** {error_ctx.strategy.value}")
            if error_ctx.table_name:
                st.markdown(f"- **Table:** `{error_ctx.table_name}`")
            if error_ctx.column_name:
                st.markdown(f"- **Column:** `{error_ctx.column_name}`")
        
        if error_ctx.hint:
            st.info(f"💡 **Hint:** {error_ctx.hint}")
        
        if error_ctx.original_query:
            st.markdown("**Failed Query:**")
            st.code(error_ctx.original_query, language="sql")


def display_error_statistics():
    """Display error statistics in sidebar"""
    if not st.session_state.current_conversation:
        return
    
    try:
        stats = _global_recovery_engine.get_statistics(st.session_state.current_conversation)
        
        if stats['total_errors'] == 0:
            return
        
        with st.expander("📊 Error Statistics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Errors", stats['total_errors'])
                st.metric("Resolved", stats['resolved'])
            
            with col2:
                resolution_rate = stats['resolution_rate'] * 100
                st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
            
            if stats['by_category']:
                st.markdown("**By Category:**")
                for category, count in stats['by_category'].items():
                    st.markdown(f"- {category}: {count}")
    except Exception as e:
        # Silently fail if error recovery not initialized yet
        pass


def display_safety_info():
    """Display safety mode information in sidebar"""
    with st.sidebar:
        st.divider()
        st.markdown("### 🛡️ Safety Settings")
        
        current_preset = st.session_state.safety_preset
        config = PRESET_CONFIGS[current_preset]
        
        # Safety mode indicator
        if config.read_only_mode:
            st.success("✅ READ-ONLY Mode Active", icon="🔒")
        else:
            st.warning("⚠️ READ-WRITE Mode Active", icon="🔓")
        
        # Display key settings
        with st.expander("Current Settings"):
            st.markdown(f"""
            - **Preset:** {current_preset}
            - **Max Results:** {config.max_result_limit}
            - **Auto-Limit:** {'Enabled' if config.auto_inject_limit else 'Disabled'}
            - **Max JOINs:** {config.max_joins}
            - **System Tables:** {'Protected' if config.block_system_tables else 'Accessible'}
            """)
        
        # Preset selector
        new_preset = st.selectbox(
            "Change Safety Preset",
            options=list(PRESET_CONFIGS.keys()),
            index=list(PRESET_CONFIGS.keys()).index(current_preset),
            help="Select security level for database queries"
        )
        
        if new_preset != current_preset:
            st.session_state.safety_preset = new_preset
            st.rerun()
        
        # Error statistics toggle
        if st.session_state.current_conversation:
            st.checkbox(
                "Show Error Statistics",
                value=st.session_state.show_error_stats,
                key="show_error_stats"
            )
            
            if st.session_state.show_error_stats:
                display_error_statistics()


def display_query_blocked_error(error: QueryBlockedError):
    """Display blocked query error with helpful information"""
    st.error("🚫 Query Blocked by Safety System", icon="🛡️")
    
    error_msg = str(error)
    
    with st.expander("Why was this blocked?"):
        st.markdown(error_msg)
        
        st.markdown("""
        **Common reasons for blocked queries:**
        - Using UPDATE, DELETE, INSERT, or DROP commands in read-only mode
        - Accessing system tables or protected schemas
        - Query exceeds complexity limits (too many JOINs)
        - Query missing required LIMIT clause
        
        **What you can do:**
        - Rephrase your question to use SELECT queries only
        - Simplify complex queries
        - Ask for specific columns instead of all data
        - Request the administrator to adjust safety settings if needed
        """)


# Sidebar
with st.sidebar:
    st.markdown("## 💬 Chats")
    
    st.button("➕ New chat", on_click=new_chat_button_on_click)
    
    upload_file = st.file_uploader("Restore conversation from JSON")
    
    if upload_file:
        conversation = load_conversation(json.load(upload_file))
        st.session_state.conversations[conversation.id] = conversation
        st.toast("Conversation restored!", icon="✅")
    
    st.divider()
    
    if conversation_exists(st.session_state.current_conversation):
        st.markdown("## Current conversation")
        
        conversation_id = st.session_state.current_conversation
        with st.expander(conversation_id):
            with st.empty():
                if st.button("Backup conversation"):
                    backup_file = json.dumps(backup_conversation(conversation_id))
                    
                    no_whitespace_name = re.sub(r"\s+", "_", conversation_id)
                    if st.download_button(
                        "Download backup JSON",
                        data=backup_file,
                        file_name=f"chatdb_{no_whitespace_name}.json"
                    ):
                        st.toast("Download started.", icon="✅")
        
        st.divider()
    
    st.markdown("## Select conversation")
    for conversation_id in st.session_state.conversations.keys():
        st.button(conversation_id, on_click=set_conversation, args=[conversation_id])

# Display safety information
display_safety_info()

# Main view
if not conversation_exists(st.session_state.current_conversation):
    st.title("New conversation")
    
    # Display form for creating a new conversation
    with st.form("new_conversation_form"):
        conversation_id = st.text_input("Conversation title")
        agent_model = st.text_input(
            "Agent model",
            value="gpt-3.5-turbo",
            help="OpenAI model. See https://platform.openai.com/docs/models"
        )
        
        database_ids = st.multiselect(
            "Select databases",
            tuple(st.session_state.databases.keys())
        )
        
        if st.form_submit_button():
            if conversation_id in st.session_state.conversations:
                st.error("Conversation title has to be unique!", icon="🚨")
            else:
                st.session_state.conversations[conversation_id] = Conversation(
                    conversation_id, agent_model, database_ids
                )
                set_conversation(conversation_id)

elif not conversation_valid(st.session_state.current_conversation):
    st.title(st.session_state.current_conversation)
    st.markdown("### Could not load conversation due to missing parameters!\n\nDid you forget to restore the settings?")

elif not st.session_state.openai_key:
    st.error("OpenAI API key not set. Go to ⚙️ Settings page!", icon="🚨")

else:
    conversation_id = st.session_state.current_conversation
    conversation: Conversation = st.session_state.conversations[conversation_id]
    
    # Title with safety indicator and error count
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title(conversation_id)
    with col2:
        config = PRESET_CONFIGS[st.session_state.safety_preset]
        if config.read_only_mode:
            st.success("🔒 READ-ONLY")
        else:
            st.warning("🔓 READ-WRITE")
    with col3:
        # Show error indicator if errors exist
        try:
            stats = _global_recovery_engine.get_statistics(conversation_id)
            if stats['total_errors'] > 0:
                resolved_pct = int(stats['resolution_rate'] * 100)
                if resolved_pct >= 70:
                    st.success(f"{resolved_pct}% resolved")
                elif resolved_pct >= 40:
                    st.warning(f"{resolved_pct}% resolved")
                else:
                    st.error(f"{resolved_pct}% resolved")
        except:
            pass
    
    # Display chat messages from history
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            
            for database, query, results in message.query_results:
                display_query(database, query, results)
    
    # Initialize the enhanced agent executor
    executor = get_enhanced_executor(
        conversation_id,
        conversation.last_update_timestamp,
        st.session_state.safety_preset
    )
    
    if len(conversation.messages) == 0:
        # Add initial message
        role = "assistant"
        content = "How can I help you today? I can query your databases safely and will learn from any errors we encounter."
        conversation.add_message(role, content)
        
        with st.chat_message(role):
            st.markdown(content)
    
    use_streaming = True
    prompt = st.chat_input("Your query")
    
    # Allow retrying if the prompt failed last time
    if not prompt and st.session_state.retry:
        use_streaming = st.session_state.retry["stream"]
        prompt = st.session_state.retry["prompt"]
        st.session_state.retry = None
    
    # Accept user input
    if prompt:
        # Display message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        conversation.add_message("user", prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Show progress indicator
            with status_placeholder:
                with st.status("Processing query...", expanded=True) as status:
                    st.write("🔄 Executing query with error recovery...")
                    
                    # Execute with intelligent recovery
                    response, success, error_ctx = executor.execute_with_recovery(
                        prompt,
                        max_retries=3,
                        streaming=use_streaming
                    )
                    
                    if success:
                        status.update(label="✅ Query successful!", state="complete")
                    else:
                        status.update(label="❌ Query failed after retries", state="error")
            
            # Clear status after completion
            status_placeholder.empty()
            
            # Display response
            message_placeholder.markdown(response)
            
            # Display error context if available
            if error_ctx and not success:
                display_error_context(error_ctx)
                
                # Show retry buttons
                st.markdown("**Try Again:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.button("🔄 Retry", on_click=retry_chat, args=[prompt, True])
                with col2:
                    st.button("🔄 Retry without streaming", on_click=retry_chat, args=[prompt, False])
            
            # Show expandable elements for every SQL query generated by this prompt
            query_results = []
            for database, query, results in conversation.query_results_queue:
                query_results.append((database, query, results))
                display_query(database, query, results)
            
            conversation.query_results_queue = []
            
            # Add assistant message to chat history
            conversation.add_message("assistant", response, query_results)