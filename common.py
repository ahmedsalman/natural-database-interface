"""
Common classes and session state management with error recovery support
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import openai
import streamlit as st

from model_config import LLMProvider


class DatabaseProps:
    """Database connection properties"""
    id: str
    uri: str

    def __init__(self, id, uri) -> None:
        self.id = id
        self.uri = uri

    def get_uri_without_password(self) -> str:
        """Return URI with password masked"""
        match = re.search("(:(?!\/\/).+@)", self.uri)

        if not match:
            return self.uri

        # Use fixed password length
        return f'{self.uri[:match.start(0) + 1]}{"*" * 8}{self.uri[match.end(0) - 1:]}'

    @property
    def db_type(self) -> str:
        """
        Database engine derived from URI
        Examples: postgresql, mysql, sqlite
        """
        if not self.uri:
            return "Unknown"

        return urlparse(self.uri).scheme


class Message:
    """Chat message with optional query results"""
    role: str
    content: str
    query_results: List[Tuple[str, list]]

    def __init__(self, role, content, query_results=None) -> None:
        self.role = role
        self.content = content
        self.query_results = query_results or []


class Conversation:
    """Conversation with agent model and database connections"""
    id: str
    agent_model: str
    database_ids: List[str]
    messages: List[Message]
    query_results_queue: List[Tuple[str, str, list]]
    last_update_timestamp: float

    def __init__(
        self,
        id: str,
        agent_model: str,
        database_ids: List[str],
        messages: List[Message] = None,
    ) -> None:
        self.id = id
        self.agent_model = agent_model
        self.database_ids = list(database_ids)
        self.messages = list(messages) if messages else list()
        self.query_results_queue = list()
        self.update_timestamp()

    def add_message(self, role, content, query_results=None):
        """Add a message to the conversation"""
        self.messages.append(Message(role, content, query_results))

    def update_timestamp(self):
        """Update timestamp to invalidate agent cache"""
        self.last_update_timestamp = datetime.now().timestamp()


def init_session_state():
    """
    Initialize all session state variables including multi-provider support
    """
    # OpenAI API key (legacy - kept for backward compatibility)
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    
    # Multi-provider API keys
    if "api_keys" not in st.session_state:
        st.session_state.api_keys: Dict[LLMProvider, str] = {
            LLMProvider.OPENAI: "",
            LLMProvider.ANTHROPIC: ""
        }

    # Database connections
    if "databases" not in st.session_state:
        st.session_state.databases: Dict[str, DatabaseProps] = dict()

    # Conversations
    if "conversations" not in st.session_state:
        st.session_state.conversations: Dict[str, Conversation] = dict()

    # Current active conversation
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation: str = ""

    # Retry state for failed queries
    if "retry" not in st.session_state:
        st.session_state.retry = None

    # Error recovery and statistics state
    if "show_error_stats" not in st.session_state:
        st.session_state.show_error_stats = False
    
    # Error recovery enabled flag
    if "error_recovery_enabled" not in st.session_state:
        st.session_state.error_recovery_enabled = True
    
    # Track if error recovery is initialized
    if "error_recovery_initialized" not in st.session_state:
        st.session_state.error_recovery_initialized = False
    
    # Selected provider for UI
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = LLMProvider.OPENAI


def set_openai_api_key(api_key: str):
    """
    Set OpenAI API key in both openai module and session state
    Legacy function for backward compatibility
    """
    openai.api_key = api_key
    st.session_state.openai_key = api_key
    st.session_state.api_keys[LLMProvider.OPENAI] = api_key


def set_api_key(provider: LLMProvider, api_key: str):
    """
    Set API key for a specific provider
    
    Args:
        provider: LLM provider enum
        api_key: API key string
    """
    st.session_state.api_keys[provider] = api_key
    
    # Set in provider-specific modules
    if provider == LLMProvider.OPENAI:
        openai.api_key = api_key
        st.session_state.openai_key = api_key  # Backward compatibility


def get_api_key(provider: LLMProvider) -> Optional[str]:
    """
    Get API key for a specific provider
    
    Args:
        provider: LLM provider enum
        
    Returns:
        API key string or None if not set
    """
    return st.session_state.api_keys.get(provider, "")


def has_api_key(provider: LLMProvider) -> bool:
    """
    Check if API key is set for a provider
    
    Args:
        provider: LLM provider enum
        
    Returns:
        True if API key is set, False otherwise
    """
    key = get_api_key(provider)
    return bool(key and key.strip())


def get_all_api_keys_status() -> Dict[LLMProvider, bool]:
    """
    Get status of all API keys
    
    Returns:
        Dictionary mapping provider to key status (True if set)
    """
    return {
        provider: has_api_key(provider)
        for provider in LLMProvider
    }


def get_conversation(conversation_id: str) -> Conversation:
    """
    Get conversation by ID with validation
    
    Args:
        conversation_id: ID of the conversation to retrieve
        
    Returns:
        Conversation object
        
    Raises:
        KeyError: If conversation doesn't exist
    """
    if conversation_id not in st.session_state.conversations:
        raise KeyError(f"Conversation '{conversation_id}' not found")
    
    return st.session_state.conversations[conversation_id]


def conversation_exists(conversation_id: str) -> bool:
    """Check if a conversation exists"""
    return (
        conversation_id != "" and 
        conversation_id in st.session_state.conversations
    )


def is_conversation_valid(conversation_id: str) -> bool:
    """
    Check if conversation is valid (has all required databases)
    
    Args:
        conversation_id: ID of the conversation to validate
        
    Returns:
        True if conversation is valid, False otherwise
    """
    if not conversation_exists(conversation_id):
        return False
    
    conversation = st.session_state.conversations[conversation_id]
    return all(
        db_id in st.session_state.databases 
        for db_id in conversation.database_ids
    )


def create_conversation(
    conversation_id: str,
    agent_model: str,
    database_ids: List[str]
) -> Conversation:
    """
    Create a new conversation
    
    Args:
        conversation_id: Unique ID for the conversation
        agent_model: Model to use (e.g., gpt-4o, claude-sonnet-4-20250514)
        database_ids: List of database IDs to connect to
        
    Returns:
        Created Conversation object
        
    Raises:
        ValueError: If conversation_id already exists
    """
    if conversation_exists(conversation_id):
        raise ValueError(f"Conversation '{conversation_id}' already exists")
    
    conversation = Conversation(conversation_id, agent_model, database_ids)
    st.session_state.conversations[conversation_id] = conversation
    
    return conversation


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation
    
    Args:
        conversation_id: ID of conversation to delete
        
    Returns:
        True if deleted, False if conversation didn't exist
    """
    if conversation_exists(conversation_id):
        del st.session_state.conversations[conversation_id]
        
        # Clear current conversation if it was deleted
        if st.session_state.current_conversation == conversation_id:
            st.session_state.current_conversation = ""
        
        return True
    
    return False


def get_all_conversation_ids() -> List[str]:
    """Get list of all conversation IDs"""
    return list(st.session_state.conversations.keys())


def add_database(database_id: str, uri: str) -> DatabaseProps:
    """
    Add a new database connection
    
    Args:
        database_id: Unique ID for the database
        uri: Connection URI
        
    Returns:
        Created DatabaseProps object
        
    Raises:
        ValueError: If database_id already exists
    """
    if database_id in st.session_state.databases:
        raise ValueError(f"Database '{database_id}' already exists")
    
    db_props = DatabaseProps(database_id, uri)
    st.session_state.databases[database_id] = db_props
    
    return db_props


def update_database(database_id: str, uri: str) -> DatabaseProps:
    """
    Update an existing database connection
    
    Args:
        database_id: ID of database to update
        uri: New connection URI
        
    Returns:
        Updated DatabaseProps object
        
    Raises:
        KeyError: If database doesn't exist
    """
    if database_id not in st.session_state.databases:
        raise KeyError(f"Database '{database_id}' not found")
    
    st.session_state.databases[database_id].uri = uri
    
    return st.session_state.databases[database_id]


def delete_database(database_id: str) -> bool:
    """
    Delete a database connection
    
    Args:
        database_id: ID of database to delete
        
    Returns:
        True if deleted, False if database didn't exist
    """
    if database_id in st.session_state.databases:
        del st.session_state.databases[database_id]
        return True
    
    return False


def get_all_database_ids() -> List[str]:
    """Get list of all database IDs"""
    return list(st.session_state.databases.keys())


def clear_all_session_state():
    """Clear all session state (useful for logout/reset)"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize
    init_session_state()


def get_session_state_summary() -> Dict:
    """
    Get summary of current session state
    
    Returns:
        Dictionary with session state statistics
    """
    return {
        'has_openai_key': has_api_key(LLMProvider.OPENAI),
        'has_anthropic_key': has_api_key(LLMProvider.ANTHROPIC),
        'api_keys_status': get_all_api_keys_status(),
        'database_count': len(st.session_state.databases),
        'conversation_count': len(st.session_state.conversations),
        'current_conversation': st.session_state.current_conversation,
        'error_recovery_enabled': st.session_state.error_recovery_enabled,
        'show_error_stats': st.session_state.show_error_stats
    }