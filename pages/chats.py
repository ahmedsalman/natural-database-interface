"""
Enhanced chats interface with SQL safety visualization
"""

import json
import re
import warnings
import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from sqlalchemy.exc import DBAPIError, NoSuchColumnError, NoSuchTableError

from agent import get_agent
from backup import backup_conversation, load_conversation
from common import Conversation, init_session_state
from multi_database import NoSuchDatabaseError, QueryBlockedError
from config import PRESET_CONFIGS

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
    
    # Title with safety indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(conversation_id)
    with col2:
        config = PRESET_CONFIGS[st.session_state.safety_preset]
        if config.read_only_mode:
            st.success("🔒 READ-ONLY")
        else:
            st.warning("🔓 READ-WRITE")
    
    # Display chat messages from history
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            
            for database, query, results in message.query_results:
                display_query(database, query, results)
    
    # Initialize the agent with safety preset
    get_agent(
        conversation_id,
        conversation.last_update_timestamp,
        st.session_state.safety_preset
    )
    
    if len(conversation.messages) == 0:
        # Add initial message
        role = "assistant"
        content = "How can I help you today? I can query your databases safely and securely."
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
        
        # Retrieve agent
        agent = get_agent(
            conversation_id,
            conversation.last_update_timestamp,
            st.session_state.safety_preset
        )
        
        # Initialize auto retry count
        auto_retry_count = 3
        show_retry_buttons = False
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            exception: str
            system_message: str
            
            while True:
                try:
                    exception = ""
                    system_message = ""
                    
                    if use_streaming:
                        # Incrementally display response as it is streamed from the agent
                        for response in agent.stream_chat(prompt).response_gen:
                            full_response += response
                            message_placeholder.markdown(full_response + "▌")
                    else:
                        # Receive the whole response before displaying it
                        message_placeholder.markdown("*Thinking...*")
                        full_response = agent.chat(prompt).response
                
                # Handle query blocked by safety validation
                except QueryBlockedError as e:
                    display_query_blocked_error(e)
                    
                    full_response = "[System] Your query was blocked by the safety system.\n\n"
                    full_response += f"Reason: {str(e)}\n\n"
                    full_response += "Please rephrase your question or check the safety settings."
                    
                    show_retry_buttons = False  # Don't show retry for blocked queries
                    break
                
                # Give the agent some useful info about the error and what it needs to do to avoid it
                except NoSuchColumnError as e:
                    exception = e
                    system_message = f"Error: {type(e).__name__}\n"
                    system_message += "Use describe_tables() function to retrieve details about the table."
                
                except NoSuchTableError as e:
                    exception = e
                    system_message = f"Error: {type(e).__name__}\n"
                    system_message += "Use list_tables() function to get a list of the tables."
                
                except NoSuchDatabaseError as e:
                    exception = e
                    system_message = f"Error: {type(e).__name__}\n"
                    system_message += "Use list_databases() function to get a list of the databases."
                
                except DBAPIError as e:
                    exception = e.orig
                    system_message = f"Error: {type(e.orig).__name__}\n"
                    system_message += "Use describe_tables() function to retrieve details about the table."
                
                except Exception as e:
                    # This is NOT an exception the agent should see
                    
                    # Show the error to the user and add a "retry" button
                    full_response = "[System] An error has occurred:\n\n"
                    full_response += "```" + str(e).replace("\n", "\n\n") + "```"
                    
                    show_retry_buttons = True
                else:
                    if full_response == "":
                        # Something wrong happened
                        full_response = "[System] An error has occurred, possibly related to streaming."
                        show_retry_buttons = True
                
                if exception:
                    # Let the agent know about the error
                    agent._memory.put(
                        ChatMessage(
                            content=system_message,
                            role=MessageRole.SYSTEM,
                        )
                    )
                    
                    # Give the agent another chance to try the tool that was recommended in the previous error
                    if auto_retry_count > 0:
                        auto_retry_count -= 1
                        continue
                    
                    # Show the error to the user
                    full_response = "[System] An SQL error has occurred:\n\n"
                    full_response += f'Error type: "{type(exception).__name__}"\n\n'
                    full_response += "```" + str(exception).replace("\n", "\n\n") + "```"
                    
                    show_retry_buttons = True
                
                break
            
            # Display full message once it is retrieved
            message_placeholder.markdown(full_response)
            
            if show_retry_buttons:
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
            conversation.add_message("assistant", full_response, query_results)