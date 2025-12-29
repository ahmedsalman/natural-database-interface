"""
Enhanced SQL Error Classification and Recovery System
Provides intelligent error handling with learning capabilities
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import defaultdict


class ErrorCategory(Enum):
    """Categories of SQL errors for intelligent handling"""
    SCHEMA_ERROR = "schema"           # Table/column doesn't exist
    SYNTAX_ERROR = "syntax"           # SQL syntax problems
    PERMISSION_ERROR = "permission"   # Access denied
    CONSTRAINT_ERROR = "constraint"   # FK, unique, check violations
    DATA_TYPE_ERROR = "datatype"      # Type mismatch
    TIMEOUT_ERROR = "timeout"         # Query timeout
    CONNECTION_ERROR = "connection"   # Database connection issues
    SAFETY_ERROR = "safety"           # Blocked by safety layer
    SEMANTIC_ERROR = "semantic"       # Logically incorrect query
    RESOURCE_ERROR = "resource"       # Out of memory, disk space
    UNKNOWN_ERROR = "unknown"         # Unclassified errors


class ErrorSeverity(Enum):
    """Severity levels for error classification"""
    LOW = 1       # Minor issues, easily recoverable
    MEDIUM = 2    # Requires retry with modifications
    HIGH = 3      # Significant issues, may need user input
    CRITICAL = 4  # Cannot auto-recover, requires intervention


class RecoveryStrategy(Enum):
    """Strategies for error recovery"""
    RETRY_AS_IS = "retry_as_is"                    # Just retry the same query
    RETRY_WITH_SCHEMA = "retry_with_schema"        # Get schema info first
    RETRY_WITH_SIMPLIFIED = "retry_simplified"     # Simplify the query
    RETRY_WITH_CORRECTION = "retry_corrected"      # Apply known correction
    ASK_USER = "ask_user"                          # Need user clarification
    ABORT = "abort"                                # Cannot recover


@dataclass
class ErrorPattern:
    """Pattern matching for specific error types"""
    pattern: str                    # Regex pattern to match
    category: ErrorCategory         # Error category
    severity: ErrorSeverity        # How serious is this
    strategy: RecoveryStrategy     # How to recover
    hint: str                      # Helpful hint for the agent
    user_message: str              # User-friendly message
    
    def matches(self, error_text: str) -> bool:
        """Check if error text matches this pattern"""
        return bool(re.search(self.pattern, error_text, re.IGNORECASE))


@dataclass
class ErrorContext:
    """Context information about an error"""
    timestamp: datetime
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    strategy: RecoveryStrategy
    
    # Query context
    original_query: str
    database_name: str
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    
    # Recovery context
    attempted_fixes: List[str] = field(default_factory=list)
    retry_count: int = 0
    hint: str = ""
    user_message: str = ""
    
    # Learning context
    successful_fix: Optional[str] = None
    resolution_time: Optional[datetime] = None


@dataclass
class ErrorMemory:
    """Memory of past errors for learning"""
    conversation_id: str
    errors: List[ErrorContext] = field(default_factory=list)
    successful_patterns: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def add_error(self, error: ErrorContext):
        """Record a new error"""
        self.errors.append(error)
    
    def record_success(self, error: ErrorContext, fix: str):
        """Record a successful fix"""
        error.successful_fix = fix
        error.resolution_time = datetime.now()
        
        # Store pattern for future learning
        pattern_key = f"{error.category.value}:{error.table_name or 'any'}"
        self.successful_patterns[pattern_key].append(fix)
    
    def get_similar_errors(self, error: ErrorContext) -> List[ErrorContext]:
        """Find similar past errors"""
        similar = []
        for past_error in self.errors:
            if (past_error.category == error.category and
                past_error.table_name == error.table_name):
                similar.append(past_error)
        return similar
    
    def get_successful_fixes(self, category: ErrorCategory, table: Optional[str] = None) -> List[str]:
        """Get previously successful fixes for this error type"""
        pattern_key = f"{category.value}:{table or 'any'}"
        return self.successful_patterns.get(pattern_key, [])
    
    def get_error_summary(self) -> Dict:
        """Get summary statistics"""
        total = len(self.errors)
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        resolved = 0
        
        for error in self.errors:
            by_category[error.category.value] += 1
            by_severity[error.severity.value] += 1
            if error.successful_fix:
                resolved += 1
        
        return {
            'total_errors': total,
            'resolved': resolved,
            'resolution_rate': resolved / total if total > 0 else 0,
            'by_category': dict(by_category),
            'by_severity': dict(by_severity)
        }


class SQLErrorClassifier:
    """
    Intelligent SQL error classifier with pattern matching
    """
    
    def __init__(self):
        """Initialize with comprehensive error patterns"""
        self.patterns = self._build_error_patterns()
    
    def _build_error_patterns(self) -> List[ErrorPattern]:
        """Build comprehensive error pattern database"""
        return [
            # Schema Errors - Table doesn't exist
            ErrorPattern(
                pattern=r"(table|relation).*?does not exist|no such table",
                category=ErrorCategory.SCHEMA_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_SCHEMA,
                hint="The table name is incorrect or doesn't exist. Use list_tables() to see available tables.",
                user_message="Table not found. Checking available tables..."
            ),
            
            # Schema Errors - Column doesn't exist
            ErrorPattern(
                pattern=r"(column|field).*?does not exist|unknown column",
                category=ErrorCategory.SCHEMA_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_SCHEMA,
                hint="The column name is incorrect. Use describe_tables() to see the correct column names.",
                user_message="Column not found. Checking table structure..."
            ),
            
            # Syntax Errors
            ErrorPattern(
                pattern=r"syntax error|syntaxerror|invalid syntax",
                category=ErrorCategory.SYNTAX_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_CORRECTION,
                hint="SQL syntax is incorrect. Review the query structure and SQL keywords.",
                user_message="SQL syntax error. Attempting to fix..."
            ),
            
            # Permission Errors
            ErrorPattern(
                pattern=r"permission denied|access denied|insufficient privileges",
                category=ErrorCategory.PERMISSION_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.ASK_USER,
                hint="You don't have permission to access this resource.",
                user_message="Access denied. This operation requires elevated permissions."
            ),
            
            # Constraint Violations
            ErrorPattern(
                pattern=r"foreign key|unique constraint|check constraint|violates",
                category=ErrorCategory.CONSTRAINT_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.ABORT,
                hint="Data constraint violation. The operation would violate database rules.",
                user_message="Data constraint violation detected."
            ),
            
            # Data Type Errors
            ErrorPattern(
                pattern=r"invalid input syntax for.*?type|type mismatch|cannot convert",
                category=ErrorCategory.DATA_TYPE_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_CORRECTION,
                hint="Data type mismatch. Check the column data types and adjust the query.",
                user_message="Data type mismatch. Adjusting query..."
            ),
            
            # Timeout Errors
            ErrorPattern(
                pattern=r"timeout|query.*?too long|execution time limit",
                category=ErrorCategory.TIMEOUT_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.RETRY_WITH_SIMPLIFIED,
                hint="Query took too long. Add LIMIT clause or simplify joins.",
                user_message="Query timeout. Simplifying query..."
            ),
            
            # Connection Errors
            ErrorPattern(
                pattern=r"connection.*?refused|lost connection|cannot connect",
                category=ErrorCategory.CONNECTION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                strategy=RecoveryStrategy.RETRY_AS_IS,
                hint="Database connection issue. Will retry the connection.",
                user_message="Database connection lost. Retrying..."
            ),
            
            # Safety Layer Blocks
            ErrorPattern(
                pattern=r"blocked by safety|not allowed in read-only|safety validation",
                category=ErrorCategory.SAFETY_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.RETRY_WITH_CORRECTION,
                hint="Query blocked by safety layer. Rephrase as a SELECT query.",
                user_message="Query blocked for safety. Generating alternative..."
            ),
            
            # Ambiguous References
            ErrorPattern(
                pattern=r"ambiguous|not unique",
                category=ErrorCategory.SEMANTIC_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_CORRECTION,
                hint="Ambiguous column reference. Use table aliases to clarify.",
                user_message="Ambiguous reference. Clarifying query..."
            ),
            
            # Division by Zero
            ErrorPattern(
                pattern=r"division by zero",
                category=ErrorCategory.SEMANTIC_ERROR,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.RETRY_WITH_CORRECTION,
                hint="Division by zero error. Add WHERE clause to filter zero values.",
                user_message="Division by zero. Adding safety check..."
            ),
            
            # Too Many Connections
            ErrorPattern(
                pattern=r"too many connections|connection pool",
                category=ErrorCategory.RESOURCE_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.RETRY_AS_IS,
                hint="Database has too many active connections. Will retry after brief wait.",
                user_message="Database busy. Will retry shortly..."
            ),
            
            # Lock Timeout
            ErrorPattern(
                pattern=r"lock timeout|deadlock",
                category=ErrorCategory.RESOURCE_ERROR,
                severity=ErrorSeverity.HIGH,
                strategy=RecoveryStrategy.RETRY_AS_IS,
                hint="Database lock conflict. Will retry the query.",
                user_message="Database lock detected. Retrying..."
            ),
        ]
    
    def classify(self, error: Exception, query: str = "", database: str = "") -> ErrorContext:
        """
        Classify an error and return context
        
        Args:
            error: The exception that occurred
            query: The SQL query that caused the error
            database: The database name
            
        Returns:
            ErrorContext with classification and recovery info
        """
        error_text = str(error)
        error_type = type(error).__name__
        
        # Try to match against known patterns
        for pattern in self.patterns:
            if pattern.matches(error_text):
                return ErrorContext(
                    timestamp=datetime.now(),
                    error_type=error_type,
                    error_message=error_text,
                    category=pattern.category,
                    severity=pattern.severity,
                    strategy=pattern.strategy,
                    original_query=query,
                    database_name=database,
                    table_name=self._extract_table_name(error_text, query),
                    column_name=self._extract_column_name(error_text, query),
                    hint=pattern.hint,
                    user_message=pattern.user_message
                )
        
        # Default classification for unknown errors
        return ErrorContext(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_text,
            category=ErrorCategory.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            strategy=RecoveryStrategy.RETRY_AS_IS,
            original_query=query,
            database_name=database,
            hint="An unexpected error occurred. Will attempt to retry.",
            user_message="Unexpected error. Analyzing..."
        )
    
    def _extract_table_name(self, error_text: str, query: str) -> Optional[str]:
        """Extract table name from error message or query"""
        # Try to extract from error message
        table_match = re.search(r'table\s+"?(\w+)"?', error_text, re.IGNORECASE)
        if table_match:
            return table_match.group(1)
        
        # Try to extract from query
        from_match = re.search(r'from\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            return from_match.group(1)
        
        return None
    
    def _extract_column_name(self, error_text: str, query: str) -> Optional[str]:
        """Extract column name from error message"""
        column_match = re.search(r'column\s+"?(\w+)"?', error_text, re.IGNORECASE)
        if column_match:
            return column_match.group(1)
        
        return None


class ErrorRecoveryEngine:
    """
    Intelligent error recovery with learning capabilities
    """
    
    def __init__(self, classifier: SQLErrorClassifier):
        """
        Initialize recovery engine
        
        Args:
            classifier: Error classifier instance
        """
        self.classifier = classifier
        self.memories: Dict[str, ErrorMemory] = {}
    
    def get_memory(self, conversation_id: str) -> ErrorMemory:
        """Get or create error memory for conversation"""
        if conversation_id not in self.memories:
            self.memories[conversation_id] = ErrorMemory(conversation_id)
        return self.memories[conversation_id]
    
    def handle_error(
        self,
        error: Exception,
        query: str,
        database: str,
        conversation_id: str,
        max_retries: int = 3
    ) -> Tuple[ErrorContext, str]:
        """
        Handle an error with intelligent recovery
        
        Args:
            error: The exception that occurred
            query: The SQL query
            database: Database name
            conversation_id: Current conversation ID
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (ErrorContext, recovery_message)
        """
        # Classify the error
        error_ctx = self.classifier.classify(error, query, database)
        
        # Get conversation memory
        memory = self.get_memory(conversation_id)
        
        # Check for similar past errors
        similar = memory.get_similar_errors(error_ctx)
        
        # Build recovery message
        recovery_message = self._build_recovery_message(error_ctx, similar, memory)
        
        # Record error
        memory.add_error(error_ctx)
        
        return error_ctx, recovery_message
    
    def _build_recovery_message(
        self,
        error_ctx: ErrorContext,
        similar_errors: List[ErrorContext],
        memory: ErrorMemory
    ) -> str:
        """
        Build intelligent recovery message with context
        
        Args:
            error_ctx: Current error context
            similar_errors: Similar past errors
            memory: Error memory
            
        Returns:
            Recovery message string
        """
        message_parts = []
        
        # Start with error classification
        message_parts.append(f"Error: {error_ctx.error_type}")
        message_parts.append(f"Category: {error_ctx.category.value}")
        message_parts.append("")
        
        # Add the specific hint
        message_parts.append(error_ctx.hint)
        message_parts.append("")
        
        # Check for previously successful fixes
        successful_fixes = memory.get_successful_fixes(
            error_ctx.category,
            error_ctx.table_name
        )
        
        if successful_fixes:
            message_parts.append("Based on past experience:")
            for fix in successful_fixes[-3:]:  # Last 3 successful fixes
                message_parts.append(f"  - {fix}")
            message_parts.append("")
        
        # Add recovery strategy
        strategy_messages = {
            RecoveryStrategy.RETRY_WITH_SCHEMA: 
                "I'll retrieve the schema information and try again.",
            RecoveryStrategy.RETRY_WITH_SIMPLIFIED:
                "I'll simplify the query and try again.",
            RecoveryStrategy.RETRY_WITH_CORRECTION:
                "I'll apply a correction and try again.",
            RecoveryStrategy.RETRY_AS_IS:
                "I'll retry the same query.",
            RecoveryStrategy.ASK_USER:
                "I need your help to resolve this.",
            RecoveryStrategy.ABORT:
                "I cannot automatically recover from this error."
        }
        
        message_parts.append(f"🔄 Strategy: {strategy_messages[error_ctx.strategy]}")
        
        # Add similar error count if relevant
        if similar_errors:
            resolved = sum(1 for e in similar_errors if e.successful_fix)
            message_parts.append("")
            message_parts.append(
                f"This type of error has occurred {len(similar_errors)} times "
                f"({resolved} resolved successfully)"
            )
        
        return "\n".join(message_parts)
    
    def suggest_fix(self, error_ctx: ErrorContext) -> Optional[str]:
        """
        Suggest a fix based on error classification
        
        Args:
            error_ctx: Error context
            
        Returns:
            Suggested fix string or None
        """
        suggestions = {
            ErrorCategory.SCHEMA_ERROR: (
                "Call list_tables() or describe_tables() to get correct names"
            ),
            ErrorCategory.SYNTAX_ERROR: (
                "Review SQL syntax. Check for missing commas, unmatched quotes, or incorrect keywords"
            ),
            ErrorCategory.TIMEOUT_ERROR: (
                "Add LIMIT clause, reduce JOINs, or add WHERE filters"
            ),
            ErrorCategory.SAFETY_ERROR: (
                "Rephrase as SELECT query. Avoid UPDATE/DELETE/DROP operations"
            ),
            ErrorCategory.DATA_TYPE_ERROR: (
                "Check data types. Use CAST() or ensure correct literal formats"
            ),
            ErrorCategory.SEMANTIC_ERROR: (
                "Use table aliases for clarity: FROM users u JOIN orders o"
            ),
        }
        
        return suggestions.get(error_ctx.category)
    
    def record_success(self, conversation_id: str, error_ctx: ErrorContext, fix: str):
        """
        Record a successful error resolution
        
        Args:
            conversation_id: Conversation ID
            error_ctx: Error that was resolved
            fix: What fixed it
        """
        memory = self.get_memory(conversation_id)
        memory.record_success(error_ctx, fix)
    
    def get_statistics(self, conversation_id: str) -> Dict:
        """
        Get error recovery statistics
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Statistics dictionary
        """
        memory = self.get_memory(conversation_id)
        return memory.get_error_summary()


# Convenience functions
_global_classifier = SQLErrorClassifier()
_global_recovery_engine = ErrorRecoveryEngine(_global_classifier)


def classify_error(error: Exception, query: str = "", database: str = "") -> ErrorContext:
    """Quick error classification"""
    return _global_classifier.classify(error, query, database)


def handle_error(
    error: Exception,
    query: str,
    database: str,
    conversation_id: str
) -> Tuple[ErrorContext, str]:
    """Quick error handling"""
    return _global_recovery_engine.handle_error(error, query, database, conversation_id)