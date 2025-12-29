"""
Enhanced agent module with SQL validation and error recovery
"""

import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from typing import Optional, Tuple

from common import Conversation, DatabaseProps
from multi_database import MultiDatabaseToolSpec, TrackingDatabaseToolSpec, QueryBlockedError
from config import Config, get_preset_config
from error_recovery import (
    ErrorRecoveryEngine, SQLErrorClassifier, ErrorContext,
    RecoveryStrategy, ErrorCategory
)


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


@st.cache_resource(show_spinner="Initializing error recovery system...")
def get_error_recovery_engine() -> ErrorRecoveryEngine:
    """
    Get or create error recovery engine
    
    Returns:
        ErrorRecoveryEngine instance
    """
    classifier = SQLErrorClassifier()
    return ErrorRecoveryEngine(classifier)


@st.cache_resource(show_spinner="Initializing error recovery...")
def get_recovery_engine() -> ErrorRecoveryEngine:
    """Get or create error recovery engine"""
    classifier = SQLErrorClassifier()
    return ErrorRecoveryEngine(classifier)


@st.cache_resource(show_spinner="Creating agent...")
def get_agent(
    conversation_id: str,
    last_update_timestamp: float,
    safety_preset: str = 'production'
):
    """
    Get or create agent with database tools and error recovery
    
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
    
    # Enhanced system prompt with error recovery guidance
    system_prompt = """You are a helpful AI assistant that can query databases using SQL.

IMPORTANT SAFETY RULES:
1. You operate in READ-ONLY mode by default - no INSERT, UPDATE, DELETE, DROP, or other modification commands
2. Always use explicit column names instead of SELECT *
3. Your queries will automatically be limited to prevent excessive results
4. System tables and schemas are protected - you cannot access them
5. Complex queries with too many JOINs may be rejected
6. If a query is blocked, explain why and suggest an alternative approach

ERROR RECOVERY INTELLIGENCE:
When you encounter an error:
1. **Learn from it**: Understand what went wrong
2. **Use schema tools**: Call list_tables() or describe_tables() to verify names
3. **Simplify**: Break complex queries into simpler parts
4. **Adapt**: Modify your approach based on the specific error
5. **Remember**: Similar errors should be avoided in future queries

COMMON ERROR PATTERNS AND SOLUTIONS:
- "Table does not exist" → Call list_tables() to see available tables
- "Column does not exist" → Call describe_tables(table_name) to see columns
- "Syntax error" → Review SQL keywords and structure
- "Timeout" → Add LIMIT clause or reduce JOINs
- "Safety block" → Rephrase as SELECT query
- "Ambiguous column" → Use table aliases (e.g., users u, orders o)

When generating SQL:
- Be specific about which tables and columns you need
- Use appropriate WHERE clauses to filter data
- Consider performance implications of your queries
- Ask clarifying questions if the user's request is ambiguous
- Explain your query choices to help users understand
- If you tried a query that failed, ALWAYS try a different approach

RETRY STRATEGY:
1st attempt: Try the direct query
2nd attempt: Get schema information first, then try again
3rd attempt: Simplify the query (fewer JOINs, basic SELECT)
Final attempt: Ask user for clarification

Never give up after one error - use the tools and hints provided to adapt and succeed.
"""
    
    # Create the agent
    agent = OpenAIAgent.from_tools(
        tools,
        llm=llm,
        chat_history=chat_history,
        system_prompt=system_prompt,
        verbose=True
    )
    
    return agent, database_tools


class EnhancedAgentExecutor:
    """
    Wrapper for agent execution with intelligent error recovery
    """
    
    def __init__(
        self,
        agent: OpenAIAgent,
        recovery_engine: ErrorRecoveryEngine,
        conversation_id: str,
        database_tools: MultiDatabaseToolSpec
    ):
        """
        Initialize enhanced executor
        
        Args:
            agent: OpenAIAgent instance
            recovery_engine: Error recovery engine
            conversation_id: Current conversation ID
            database_tools: Database tools for schema queries
        """
        self.agent = agent
        self.recovery_engine = recovery_engine
        self.conversation_id = conversation_id
        self.database_tools = database_tools
        self.current_error_ctx: Optional[ErrorContext] = None
    
    def execute_with_recovery(
        self,
        prompt: str,
        max_retries: int = 3,
        streaming: bool = True
    ) -> Tuple[str, bool, Optional[ErrorContext]]:
        """
        Execute query with intelligent error recovery
        
        Args:
            prompt: User prompt
            max_retries: Maximum retry attempts
            streaming: Use streaming response
            
        Returns:
            Tuple of (response, success, error_context)
        """
        retry_count = 0
        last_error_ctx = None
        
        while retry_count <= max_retries:
            try:
                # Execute the query
                if streaming:
                    response = ""
                    for chunk in self.agent.stream_chat(prompt).response_gen:
                        response += chunk
                else:
                    response = self.agent.chat(prompt).response
                
                # Success! Record if we recovered from an error
                if last_error_ctx and retry_count > 0:
                    self.recovery_engine.record_success(
                        self.conversation_id,
                        last_error_ctx,
                        f"Succeeded after {retry_count} retries"
                    )
                
                return response, True, None
            
            except QueryBlockedError as e:
                # Handle blocks separately
                error_ctx, recovery_msg = self.recovery_engine.handle_error(
                    e, "", "", self.conversation_id, max_retries
                )
                
                # Inject recovery guidance
                self._inject_recovery_message(error_ctx, recovery_msg)
                
                last_error_ctx = error_ctx
                retry_count += 1
                
                if error_ctx.strategy == RecoveryStrategy.ABORT:
                    return self._format_error_response(error_ctx, recovery_msg), False, error_ctx
            
            except Exception as e:
                # Classify and handle error
                error_ctx, recovery_msg = self.recovery_engine.handle_error(
                    e, "", "", self.conversation_id, max_retries
                )
                
                error_ctx.retry_count = retry_count
                last_error_ctx = error_ctx
                
                # Check if we should retry
                if not self._should_retry(error_ctx, retry_count, max_retries):
                    return self._format_error_response(error_ctx, recovery_msg), False, error_ctx
                
                # Inject recovery guidance into agent
                self._inject_recovery_message(error_ctx, recovery_msg)
                
                # Apply recovery strategy
                self._apply_recovery_strategy(error_ctx)
                
                retry_count += 1
        
        # Max retries reached
        return self._format_error_response(
            last_error_ctx,
            "Maximum retry attempts reached. Please rephrase your question."
        ), False, last_error_ctx
    
    def _should_retry(self, error_ctx: ErrorContext, retry_count: int, max_retries: int) -> bool:
        """Determine if we should retry based on error context"""
        if retry_count >= max_retries:
            return False
        
        if error_ctx.strategy == RecoveryStrategy.ABORT:
            return False
        
        if error_ctx.severity.value >= 4:  # CRITICAL
            return False
        
        return True
    
    def _inject_recovery_message(self, error_ctx: ErrorContext, recovery_msg: str):
        """Inject recovery message into agent's memory"""
        system_message = f"""SYSTEM ERROR RECOVERY:

{recovery_msg}

Previous Query: {error_ctx.original_query}

IMPORTANT: Do NOT retry the exact same query. Analyze the error and adapt your approach.
"""
        
        self.agent._memory.put(
            ChatMessage(
                content=system_message,
                role=MessageRole.SYSTEM
            )
        )
    
    def _apply_recovery_strategy(self, error_ctx: ErrorContext):
        """Apply specific recovery strategy"""
        if error_ctx.strategy == RecoveryStrategy.RETRY_WITH_SCHEMA:
            # Inject hint to get schema info
            hint = f"Call list_tables() and describe_tables() before retrying."
            if error_ctx.table_name:
                hint = f"Call describe_tables('{error_ctx.table_name}') to see correct column names."
            
            self.agent._memory.put(
                ChatMessage(
                    content=f"HINT: {hint}",
                    role=MessageRole.SYSTEM
                )
            )
        
        elif error_ctx.strategy == RecoveryStrategy.RETRY_WITH_SIMPLIFIED:
            hint = "Simplify your query: remove JOINs, add LIMIT, use basic SELECT."
            self.agent._memory.put(
                ChatMessage(
                    content=f"HINT: {hint}",
                    role=MessageRole.SYSTEM
                )
            )
        
        elif error_ctx.strategy == RecoveryStrategy.RETRY_WITH_CORRECTION:
            suggestion = self.recovery_engine.suggest_fix(error_ctx)
            if suggestion:
                self.agent._memory.put(
                    ChatMessage(
                        content=f"SUGGESTED FIX: {suggestion}",
                        role=MessageRole.SYSTEM
                    )
                )
    
    def _format_error_response(self, error_ctx: ErrorContext, recovery_msg: str) -> str:
        """Format error response for display"""
        return f"""Error: {error_ctx.user_message}

**Error Details:**
- Type: {error_ctx.error_type}
- Category: {error_ctx.category.value}
- Severity: {error_ctx.severity.name}

**Recovery Information:**
{recovery_msg}

**What happened:**
{error_ctx.error_message[:200]}...

Try rephrasing your question or being more specific about what you need.
"""


def get_enhanced_executor(
    conversation_id: str,
    last_update_timestamp: float,
    safety_preset: str = 'production'
) -> EnhancedAgentExecutor:
    """
    Get enhanced agent executor with error recovery
    
    Args:
        conversation_id: Conversation ID
        last_update_timestamp: Cache invalidation timestamp
        safety_preset: configuration preset
        
    Returns:
        EnhancedAgentExecutor instance
    """
    agent, database_tools = get_agent(conversation_id, last_update_timestamp, safety_preset)
    recovery_engine = get_error_recovery_engine()
    
    return EnhancedAgentExecutor(
        agent=agent,
        recovery_engine=recovery_engine,
        conversation_id=conversation_id,
        database_tools=database_tools
    )


def get_error_statistics(conversation_id: str) -> dict:
    """
    Get error recovery statistics for a conversation
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Statistics dictionary
    """
    recovery_engine = get_error_recovery_engine()
    return recovery_engine.get_statistics(conversation_id)