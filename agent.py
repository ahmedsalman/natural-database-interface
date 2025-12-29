"""
Enhanced agent module with SQL validation
"""

import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from common import Conversation, DatabaseProps
from multi_database import MultiDatabaseToolSpec, TrackingDatabaseToolSpec, QueryBlockedError
from config import Config, get_preset_config


@st.cache_resource(show_spinner="Loading LLM...")
def get_llm(model: str, api_key: str):
    """
    Get or create LLM instance
    
    Args:
        model: OpenAI model name
        api_key: OpenAI API key (used to invalidate cache)
        
    Returns:
        OpenAI LLM instance
    """
    _ = api_key  # Force cache invalidation on API key change
    return OpenAI(model=model)


@st.cache_resource(show_spinner="Connecting to database...")
def get_database_spec(
    database_id: str,
    safety_preset: str = 'production'
) -> TrackingDatabaseToolSpec:
    """
    Get or create database spec with configuration
    
    Args:
        database_id: Database identifier
        safety_preset: configuration preset
        
    Returns:
        TrackingDatabaseToolSpec instance
    """
    database: DatabaseProps = st.session_state.databases[database_id]
    
    # Get configuration
    config = get_preset_config(safety_preset)
    
    # Create safe database spec
    db_spec = TrackingDatabaseToolSpec(
        uri=database.uri,
        config=config
    )
    
    # Set the database name for query tracking
    db_spec.database_name = database_id
    
    return db_spec


def database_spec_handler(database, query, items):
    """
    Handler called after successful query execution
    
    Args:
        database: Database identifier
        query: SQL query that was executed
        items: Query result items
    """
    conversation = st.session_state.conversations[st.session_state.current_conversation]
    conversation.query_results_queue.append((database, query, items))


@st.cache_resource(show_spinner="Creating agent...")
def get_agent(
    conversation_id: str,
    last_update_timestamp: float,
    safety_preset: str = 'production'
):
    """
    Get or create agent with safety-enabled database tools
    
    Args:
        conversation_id: Conversation identifier
        last_update_timestamp: Timestamp to invalidate cache
        safety_preset: configuration preset
        
    Returns:
        OpenAIAgent instance with safe database tools
    """
    # Used for invalidating the cache when we want to force create a new agent
    _ = last_update_timestamp
    
    conversation: Conversation = st.session_state.conversations[conversation_id]
    
    # Create multi-database tool with configuration
    database_tools = MultiDatabaseToolSpec(
        handler=database_spec_handler,
        safety_preset=safety_preset
    )
    
    # Add database connections with checks
    for database_id in conversation.database_ids:
        db_spec = get_database_spec(database_id, safety_preset)
        database_tools.add_database_tool_spec(database_id, db_spec)
    
    # Convert to tool list
    tools = database_tools.to_tool_list()
    
    # Load chat history from the conversation's messages
    chat_history = list(map(
        lambda m: ChatMessage(role=m.role, content=m.content),
        conversation.messages
    ))
    
    # Create an LLM with the specified model
    llm = get_llm(conversation.agent_model, st.session_state.openai_key)
    
    # Create the Agent with enhanced system prompt for SQL
    system_prompt = """You are a helpful AI assistant that can query databases using SQL.

IMPORTANT SAFETY RULES:
1. You operate in READ-ONLY mode by default - no INSERT, UPDATE, DELETE, DROP, or other modification commands
2. Always use explicit column names instead of SELECT *
3. Your queries will automatically be limited to prevent excessive results
4. System tables and schemas are protected - you cannot access them
5. Complex queries with too many JOINs may be rejected
6. If a query is blocked, explain why and suggest an alternative approach

When generating SQL:
- Be specific about which tables and columns you need
- Use appropriate WHERE clauses to filter data
- Consider performance implications of your queries
- Ask clarifying questions if the user's request is ambiguous
- Explain your query choices to help users understand

If a query fails validation:
- Explain what safety rule was violated
- Suggest a safer alternative query
- Help the user understand database security best practices
"""
    
    # Create the agent
    agent = OpenAIAgent.from_tools(
        tools,
        llm=llm,
        chat_history=chat_history,
        system_prompt=system_prompt,
        verbose=True
    )
    
    return agent


def get_safety_statistics():
    """
    Get safety statistics from all active database connections
    
    Returns:
        Dictionary with aggregated safety statistics
    """
    conversation_id = st.session_state.current_conversation
    
    if not conversation_id or conversation_id not in st.session_state.conversations:
        return None
    
    conversation: Conversation = st.session_state.conversations[conversation_id]
    
    # This would need to access the database_tools from the agent
    # For now, return basic info
    return {
        'conversation_id': conversation_id,
        'database_count': len(conversation.database_ids),
        'databases': conversation.database_ids
    }