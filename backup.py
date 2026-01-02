import json

import jsonpickle
import streamlit as st

from common import Conversation, set_api_key, get_api_key
from encryption import DEFAULT_KEY, decrypt, decrypt_prop, encrypt, encrypt_prop, generate_key
from model_config import LLMProvider

BACKUP_PROPS = ["openai_key", "api_keys", "databases", "current_conversation"]


def backup_settings(password: str) -> dict:
    """
    Backup settings with multi-provider API key support
    
    Args:
        password: Encryption password (optional)
        
    Returns:
        Dictionary containing encrypted backup data
    """
    backup = dict()

    backup["use_default_key"] = not password
    backup["version"] = "2.0"
    enc_key = generate_key(password) if password else DEFAULT_KEY

    for prop in BACKUP_PROPS:
        if prop not in st.session_state:
            continue
            
        value = st.session_state[prop]

        # Handle api_keys dictionary (multi-provider)
        if prop == "api_keys":
            encrypted_keys = {}
            for provider, key in value.items():
                if key:  # Only backup if key is set
                    # Convert provider enum to string for JSON serialization
                    provider_str = provider.value if isinstance(provider, LLMProvider) else str(provider)
                    encrypted_keys[provider_str] = encrypt(key.encode("utf-8"), enc_key).decode("utf-8")
            value = encrypted_keys
        
        # Handle databases dictionary
        elif isinstance(value, dict) and prop == "databases":
            value = {k: json.loads(jsonpickle.encode(encrypt_prop(v, enc_key))) for k, v in value.items()}

        # Handle legacy openai_key for backward compatibility
        elif prop == "openai_key":
            if value:
                value = encrypt(value.encode("utf-8"), enc_key).decode("utf-8")

        backup[prop] = value

    return backup


def load_settings(backup: dict, password: str):
    """
    Load settings from backup with multi-provider support
    
    Args:
        backup: Backup dictionary
        password: Decryption password (optional)
    """
    enc_key = generate_key(password) if password else DEFAULT_KEY
    backup_version = backup.get("version", "1.0")  # Default to 1.0 for old backups

    for prop in BACKUP_PROPS:
        if prop not in backup:
            continue
            
        value = backup[prop]

        # Handle api_keys dictionary (multi-provider)
        if prop == "api_keys":
            decrypted_keys = {}
            for provider_str, encrypted_key in value.items():
                try:
                    # Convert string back to provider enum
                    provider = LLMProvider(provider_str)
                    decrypted_key = decrypt(encrypted_key.encode("utf-8"), enc_key).decode("utf-8")
                    decrypted_keys[provider] = decrypted_key
                    
                    # Set the API key using the proper method
                    set_api_key(provider, decrypted_key)
                except (ValueError, KeyError):
                    # Skip unknown providers
                    continue
            
            st.session_state[prop] = decrypted_keys
        
        # Handle databases dictionary
        elif isinstance(value, dict) and prop == "databases":
            value = {k: decrypt_prop(jsonpickle.decode(json.dumps(v)), enc_key) for k, v in value.items()}
            st.session_state[prop] = value

        # Handle legacy openai_key
        elif prop == "openai_key":
            if value:
                decrypted_key = decrypt(value.encode("utf-8"), enc_key).decode("utf-8")
                set_api_key(LLMProvider.OPENAI, decrypted_key)
                st.session_state[prop] = decrypted_key
        
        # Handle other properties
        else:
            st.session_state[prop] = value
    
    # Backward compatibility: If old backup only has openai_key, migrate to api_keys
    if backup_version == "1.0" and "openai_key" in backup and "api_keys" not in backup:
        if "openai_key" in st.session_state and st.session_state.openai_key:
            if "api_keys" not in st.session_state:
                st.session_state.api_keys = {}
            st.session_state.api_keys[LLMProvider.OPENAI] = st.session_state.openai_key


def backup_conversation(id: str) -> dict:
    """
    Backup a specific conversation
    
    Args:
        id: Conversation ID
        
    Returns:
        Dictionary containing conversation data
    """
    if id not in st.session_state.conversations:
        return None

    return json.loads(jsonpickle.encode(st.session_state.conversations[id]))


def load_conversation(backup: dict) -> Conversation:
    """
    Load a conversation from backup
    
    Args:
        backup: Backup dictionary
        
    Returns:
        Conversation object
    """
    # As this will create a new object, the timestamp will be updated
    return jsonpickle.decode(json.dumps(backup))


def export_full_backup(password: str = None) -> dict:
    """
    Create a comprehensive backup including all conversations
    
    Args:
        password: Encryption password (optional)
        
    Returns:
        Dictionary containing full backup
    """
    backup = backup_settings(password)
    
    # Add all conversations
    conversations_backup = {}
    for conv_id in st.session_state.conversations:
        conversations_backup[conv_id] = backup_conversation(conv_id)
    
    backup["conversations"] = conversations_backup
    backup["backup_type"] = "full"
    
    return backup


def import_full_backup(backup: dict, password: str = None):
    """
    Import a comprehensive backup including all conversations
    
    Args:
        backup: Full backup dictionary
        password: Decryption password (optional)
    """
    # Load settings first
    load_settings(backup, password)
    
    # Load conversations if present
    if "conversations" in backup:
        for conv_id, conv_data in backup["conversations"].items():
            conversation = load_conversation(conv_data)
            st.session_state.conversations[conv_id] = conversation