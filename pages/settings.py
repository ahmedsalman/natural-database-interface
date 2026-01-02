import json

import streamlit as st

# For clarity
from cryptography.fernet import InvalidToken as InvalidEncryptionKey

from backup import backup_settings, load_settings
from common import DatabaseProps, init_session_state, set_api_key, get_api_key, has_api_key
from model_config import LLMProvider, PROVIDER_CONFIGS, get_models_by_provider

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
)

NEW_DATABASE_TEXT = "Add new database"

# Initialize session state variables
init_session_state()

st.title("⚙️ Settings")

st.divider()

# ========================================
# API Keys Section with Multi-Provider Support
# ========================================
st.markdown("## 🔑 API Keys")

st.info("""
**Multi-Provider Support:** ChatDB now supports multiple AI providers. 
Configure the API keys for the providers you want to use.
""")

# OpenAI API Key
with st.expander("🤖 OpenAI API Key", expanded=not has_api_key(LLMProvider.OPENAI)):
    st.markdown("""
    **OpenAI Models:** GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo
    
    Get your API key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    """)
    
    with st.form("openai_key_form", clear_on_submit=True):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key (starts with sk-...)"
        )
        
        if st.form_submit_button("Save OpenAI Key"):
            set_api_key(LLMProvider.OPENAI, openai_key)
            st.success("OpenAI API key saved!")
            st.rerun()
    
    if has_api_key(LLMProvider.OPENAI):
        st.success(" OpenAI API key is configured")
        
        # Show available models
        models = get_models_by_provider(LLMProvider.OPENAI)
        st.markdown(f"**Available Models:** {len(models)}")
        model_names = [m.name for m in models]
        st.caption(", ".join(model_names))
    else:
        st.warning(" OpenAI API key not set")

# Anthropic API Key
with st.expander(" Anthropic API Key (Claude)", expanded=not has_api_key(LLMProvider.ANTHROPIC)):
    st.markdown("""
    **Anthropic Models:** Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
    
    Get your API key from: [https://console.anthropic.com/](https://console.anthropic.com/)
    """)
    
    with st.form("anthropic_key_form", clear_on_submit=True):
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Your Anthropic API key (starts with sk-ant-...)"
        )
        
        if st.form_submit_button("Save Anthropic Key"):
            set_api_key(LLMProvider.ANTHROPIC, anthropic_key)
            st.success("Anthropic API key saved!")
            st.rerun()
    
    if has_api_key(LLMProvider.ANTHROPIC):
        st.success("Anthropic API key is configured")
        
        # Show available models
        models = get_models_by_provider(LLMProvider.ANTHROPIC)
        st.markdown(f"**Available Models:** {len(models)}")
        model_names = [m.name for m in models]
        st.caption(", ".join(model_names))
    else:
        st.info("Anthropic API key not set (optional)")

# API Key Status Summary
st.markdown("### API Key Status")
col1, col2 = st.columns(2)

with col1:
    if has_api_key(LLMProvider.OPENAI):
        st.success("OpenAI: Configured")
    else:
        st.error("OpenAI: Not Set")

with col2:
    if has_api_key(LLMProvider.ANTHROPIC):
        st.success("Anthropic: Configured")
    else:
        st.info("Anthropic: Optional")

st.divider()

# ========================================
# Databases Section
# ========================================
st.markdown("## 🗄️ Databases")
with st.expander("Configure"):
    database_selection = st.selectbox("Select database", (NEW_DATABASE_TEXT, *st.session_state.databases.keys()))

    id = ""
    uri = ""

    props = None
    if database_selection != NEW_DATABASE_TEXT:
        props = st.session_state.databases[database_selection]

        id = props.id
        uri = props.uri

    database_id = st.text_input(
        "Database identifier",
        value=id,
        help="Choose a proper, relevant name or just the database name. Used by the model to distinguish between different databases.",
    )

    database_uri = st.text_input(
        "Connection URI",
        value=uri,
        help="Use the format: `dialect://username:password@host:port/database`, where dialect can be postgresql, mysql, oracle, etc.",
    )

    if st.button("Submit", key="database_submit_button"):
        if props and props.id != database_id:
            # Remove existing database if we're going to rename it
            st.session_state.databases.pop(props.id)

        if not props and database_id in st.session_state.databases:
            # A new entry is being added, so it should have a unique id
            st.error("Database identifier has to be unique!")
        else:
            st.session_state.databases[database_id] = DatabaseProps(database_id, database_uri)
            st.success("Database saved!")

with st.expander("View databases"):
    if st.session_state.databases:
        st.table({k: {"URI": st.session_state.databases[k].get_uri_without_password()} for k in st.session_state.databases})
    else:
        st.info("No databases configured yet.")

st.divider()

# ========================================
# Backup & Restore Section
# ========================================
st.markdown("## Backup settings")

st.markdown("### Backup")
st.markdown("""
Export your settings including API keys, database connections, and conversations.
""")

password = st.text_input(
    "Encryption password",
    help="This will be used to encrypt your API keys before backup. If no password is provided, the data will still be encrypted but using a common encryption key",
    type="password",
    key="backup_password"
)

with st.empty():
    if st.button("Prepare backup"):
        # Prepare JSON file
        backup_file = json.dumps(backup_settings(password), indent=2)

        if password:
            st.info("Your backup is encrypted with the password you provided.")

        st.download_button("Download settings JSON", data=backup_file, file_name="chatdb_settings.json")

st.markdown("### Restore")
st.markdown("""
Import previously exported settings. This will restore API keys, databases, and the current conversation.
""")

upload_file = st.file_uploader("Restore settings from JSON")

if upload_file:
    backup_file = json.load(upload_file)

    loaded = False
    try:
        if "use_default_key" in backup_file and not backup_file["use_default_key"]:
            st.markdown("**Backup is encrypted!**")
            password = st.text_input(
                "Decryption password",
                help="This is the same password you used to encrypt your backup. Leave this empty if you did not use a password when backing up.",
                type="password",
                key="restore_password"
            )

            if st.button("Decrypt and restore"):
                load_settings(backup_file, password)
                loaded = True
        else:
            load_settings(backup_file, None)
            loaded = True
    except InvalidEncryptionKey:
        st.error("Invalid decryption key.")
    else:
        if loaded:
            st.success("Settings restored!")
            st.rerun()

st.divider()

# ========================================
# Help & Information
# ========================================
with st.expander(" Help & Information"):
    st.markdown("""
    ### Multi-Provider Support
    
    ChatDB now supports multiple AI providers:
    
    **OpenAI (Required for most users)**
    - GPT-4o: Recommended for best overall performance
    - GPT-4o-mini: Best value - excellent quality at low cost
    - GPT-4 Turbo: Premium option for maximum accuracy
    - GPT-3.5 Turbo: Legacy option (not recommended)
    
    **Anthropic (Optional)**
    - Claude 3.5 Sonnet: Excellent reasoning and SQL generation
    - Claude 3 Opus: Most capable Claude model
    - Claude 3 Sonnet: Balanced performance
    - Claude 3 Haiku: Fast and economical
    
    ### Getting Started
    
    1. **Configure at least one API key** (OpenAI or Anthropic)
    2. **Add your database connections** in the Databases section
    3. **Go to the Chats page** to start conversations
    4. **Select your preferred model** when creating a conversation
    
    ### Security
    
    - API keys are stored securely in your browser session
    - Use the backup feature with a password to encrypt exported data
    - Database URIs in backups are encrypted
    
    ### Cost Considerations
    
    Different models have different pricing:
    - **GPT-4o-mini**: $0.15/$0.60 per 1M tokens (best value)
    - **GPT-4o**: $2.50/$10 per 1M tokens (recommended)
    - **Claude 3.5 Sonnet**: $3/$15 per 1M tokens
    - **Claude 3 Opus**: $15/$75 per 1M tokens (premium)
    
    Most database queries use 1,000-5,000 tokens, so costs are typically fractions of a cent per query.
    """)

# Footer with version info
st.divider()
st.caption("ChatDB v2.0 - Multi-Provider Edition | Supports OpenAI GPT & Anthropic Claude")