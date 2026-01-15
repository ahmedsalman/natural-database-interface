"""
Enhanced agent module with SQL validation and error recovery
Supports SQL validation and error recovery
"""

import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from typing import Optional, Tuple, Union

from common import Conversation, DatabaseProps, get_api_key, has_api_key
from multi_database import MultiDatabaseToolSpec, TrackingDatabaseToolSpec, QueryBlockedError
from config import Config, get_preset_config
from error_recovery import (
    ErrorRecoveryEngine, SQLErrorClassifier, ErrorContext,
    RecoveryStrategy, ErrorCategory
)
from model_config import (
    get_model_config, get_provider_for_model, LLMProvider,
    validate_model_id
)


@st.cache_resource(show_spinner="Loading LLM...")
def get_llm(model: str, api_key: str):
    """
    Get or create LLM instance
    
    Args:
        model: Model identifier (e.g., gpt-4o, claude-sonnet-4-20250514)
        api_key: API key (used to invalidate cache)
        
    Returns:
        LLM instance
        
    Raises:
        ValueError: If model is not supported or API key is missing
    """
    _ = api_key  # Force cache invalidation on API key change
    
    # Validate model
    if not validate_model_id(model):
        raise ValueError(f"Unsupported model: {model}")
    
    # Get model configuration
    model_config = get_model_config(model)
    provider = model_config.provider
    
    # Check API key
    if not has_api_key(provider):
        provider_name = provider.value.capitalize()
        raise ValueError(f"{provider_name} API key not set. Please configure in Settings.")
    
    # Get the appropriate API key
    provider_api_key = get_api_key(provider)
    
    # Create LLM instance based on provider
    if provider == LLMProvider.OPENAI:
        return OpenAI(
            model=model,
            api_key=provider_api_key,
            temperature=0.1,  # Lower temperature for more consistent SQL generation
            max_tokens=4096
        )
    elif provider == LLMProvider.ANTHROPIC:
        return Anthropic(
            model=model,
            api_key=provider_api_key,
            temperature=0.1,
            max_tokens=4096
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


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


def get_system_prompt(provider: LLMProvider) -> str:
    """
    Get system prompt optimized for the specific provider
    
    Args:
        provider: LLM provider
        
    Returns:
        System prompt string
    """
    base_prompt = """You are a helpful AI assistant that can query databases using SQL.

CRITICAL EXECUTION RULE
When a user asks a question about the database, you MUST:
1. IMMEDIATELY use the database tools (load_data, list_tables, describe_tables)
2. NEVER just explain what you would do - ALWAYS DO IT
3. ALWAYS return actual query results - never just descriptions
4. Execute queries FIRST, then explain what you found

EXECUTION WORKFLOW (FOLLOW THIS EXACTLY):
Step 1: If you need schema info → Call list_tables() and/or describe_tables() RIGHT NOW
Step 2: Construct your SQL query
Step 3: IMMEDIATELY call load_data(database, query) - DO NOT WAIT
Step 4: Show the actual results to the user

NEVER ALLOWED:
"To retrieve X, we need to query Y..." WITHOUT actually querying
"Let's first check the table structure..." WITHOUT calling describe_tables()
"Let's proceed with the query..." WITHOUT calling load_data()
Explaining what you WILL do instead of DOING it
Describing the plan without executing it

ALWAYS REQUIRED:
Call list_tables() if you need to see available tables
Call describe_tables() if you need to see table structure  
Call load_data() to execute every SELECT query
Return actual data from the database
Execute BEFORE explaining

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
- "Table does not exist" → Call list_tables() to see available tables, THEN retry
- "Column does not exist" → Call describe_tables(table_name) to see columns, THEN retry
- "Syntax error" → Fix the SQL syntax and call load_data() again
- "Timeout" → Add LIMIT clause or reduce JOINs, THEN call load_data() again
- "Safety block" → Rephrase as SELECT query, THEN call load_data() again
- "Ambiguous column" → Use table aliases (e.g., users u, orders o), THEN call load_data() again

RETRY STRATEGY:
1st attempt: Try the direct query with load_data()
2nd attempt: Call list_tables()/describe_tables() to get schema, THEN call load_data() again
3rd attempt: Simplify the query, THEN call load_data() again
Final attempt: Ask user for clarification

Remember: Your PRIMARY JOB is to EXECUTE queries and return REAL DATA, not to explain what you could do.
"""
    
    # Provider-specific additions
    if provider == LLMProvider.ANTHROPIC:
        # Claude responds well to structured thinking
        base_prompt += """

CLAUDE-SPECIFIC GUIDANCE:
- Think step-by-step before constructing queries
- Use <thinking> tags internally if helpful for complex queries
- Be precise and methodical in error analysis
- Leverage your strong reasoning for query optimization
"""
    elif provider == LLMProvider.OPENAI:
        # GPT models benefit from explicit function calling guidance
        base_prompt += """

GPT-SPECIFIC GUIDANCE:
- Use function calling precisely - one tool call per action
- Be concise in your explanations
- Focus on efficiency in query execution
- Prioritize quick response with streaming
"""
    
    return base_prompt


@st.cache_resource(show_spinner="Creating agent...")
def get_agent(
    conversation_id: str,
    last_update_timestamp: float,
    safety_preset: str = 'production'
):
    """
    Get or create agent with database tools and error recovery
    Supports both OpenAI and Anthropic models
    
    Args:
        conversation_id: Conversation identifier
        last_update_timestamp: Timestamp to invalidate cache
        safety_preset: configuration preset
        
    Returns:
        Tuple of (Agent instance, database_tools)
        
    Raises:
        ValueError: If model is not supported or API key is missing
    """
    # Used for invalidating the cache when we want to force create a new agent
    _ = last_update_timestamp
    
    conversation: Conversation = st.session_state.conversations[conversation_id]
    
    # Get model and validate
    model_id = conversation.agent_model
    if not validate_model_id(model_id):
        raise ValueError(f"Invalid model: {model_id}")
    
    model_config = get_model_config(model_id)
    provider = model_config.provider
    
    # Check API key
    if not has_api_key(provider):
        provider_name = provider.value.capitalize()
        raise ValueError(f"{provider_name} API key not set")
    
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
    api_key = get_api_key(provider)
    llm = get_llm(model_id, api_key)
    
    # Get provider-optimized system prompt
    system_prompt = get_system_prompt(provider)
    
    # Create the agent
    # Note: OpenAIAgent works with both OpenAI and Anthropic models via LlamaIndex
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
    Supports both OpenAI and Anthropic models
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


def get_supported_models_info() -> dict:
    """
    Get information about all supported models
    
    Returns:
        Dictionary with model information
    """
    from model_config import MODEL_CATALOG
    
    return {
        model_id: {
            'name': config.name,
            'provider': config.provider.value,
            'tier': config.tier.value,
            'context_window': config.context_window,
            'cost_input': config.cost_per_1m_input,
            'cost_output': config.cost_per_1m_output,
            'recommended': config.recommended_for_sql
        }
        for model_id, config in MODEL_CATALOG.items()
    }