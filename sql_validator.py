"""
SQL Query Validation and Safety Layer
Provides comprehensive validation for SQL queries before execution
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, DML, DDL


class QueryType(Enum):
    """Classification of SQL query types"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    DROP = "DROP"
    CREATE = "CREATE"
    ALTER = "ALTER"
    TRUNCATE = "TRUNCATE"
    GRANT = "GRANT"
    REVOKE = "REVOKE"
    UNKNOWN = "UNKNOWN"


class ValidationSeverity(Enum):
    """Severity levels for validation results"""
    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


@dataclass
class ValidationResult:
    """Result of SQL validation"""
    is_valid: bool
    severity: ValidationSeverity
    query_type: QueryType
    issues: List[str]
    modified_query: Optional[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class SQLValidator:
    """
    Comprehensive SQL query validator with configurable safety rules
    """
    
    # Dangerous SQL keywords that should be blocked
    DANGEROUS_KEYWORDS = [
        'DROP', 'TRUNCATE', 'DELETE', 'UPDATE', 'INSERT',
        'CREATE', 'ALTER', 'GRANT', 'REVOKE', 'EXEC',
        'EXECUTE', 'CALL', 'MERGE', 'REPLACE'
    ]
    
    # System tables/schemas to protect
    PROTECTED_OBJECTS = [
        'information_schema', 'pg_catalog', 'mysql',
        'sys', 'msdb', 'master', 'tempdb'
    ]
    
    def __init__(
        self,
        read_only_mode: bool = True,
        max_result_limit: int = 1000,
        auto_inject_limit: bool = True,
        allow_joins: bool = True,
        max_joins: int = 5,
        block_system_tables: bool = True,
        enable_explain: bool = True
    ):
        """
        Initialize SQL validator with configuration
        
        Args:
            read_only_mode: Only allow SELECT queries
            max_result_limit: Maximum rows to return
            auto_inject_limit: Automatically add LIMIT clause
            allow_joins: Allow JOIN operations
            max_joins: Maximum number of JOINs allowed
            block_system_tables: Block queries on system tables
            enable_explain: Enable EXPLAIN analysis for complex queries
        """
        self.read_only_mode = read_only_mode
        self.max_result_limit = max_result_limit
        self.auto_inject_limit = auto_inject_limit
        self.allow_joins = allow_joins
        self.max_joins = max_joins
        self.block_system_tables = block_system_tables
        self.enable_explain = enable_explain
    
    def validate(self, query: str) -> ValidationResult:
        """
        Main validation entry point
        
        Args:
            query: SQL query string to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        issues = []
        suggestions = []
        modified_query = query.strip()
        
        # Parse the query
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.BLOCKED,
                    query_type=QueryType.UNKNOWN,
                    issues=["Unable to parse SQL query"]
                )
            
            statement = parsed[0]
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.BLOCKED,
                query_type=QueryType.UNKNOWN,
                issues=[f"SQL parsing error: {str(e)}"]
            )
        
        # Determine query type
        query_type = self._identify_query_type(statement)
        
        # Check 1: Read-only mode enforcement
        if self.read_only_mode and query_type != QueryType.SELECT:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.BLOCKED,
                query_type=query_type,
                issues=[f"{query_type.value} operations are not allowed in read-only mode"],
                suggestions=["Only SELECT queries are permitted"]
            )
        
        # Check 2: Dangerous keywords
        dangerous_found = self._check_dangerous_keywords(query)
        if dangerous_found:
            issues.append(f"Dangerous keywords detected: {', '.join(dangerous_found)}")
            if not self.read_only_mode:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.BLOCKED,
                    query_type=query_type,
                    issues=issues,
                    suggestions=["Consider using safer alternatives"]
                )
        
        # Check 3: System table protection
        if self.block_system_tables:
            system_tables = self._check_system_tables(query)
            if system_tables:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.BLOCKED,
                    query_type=query_type,
                    issues=[f"Access to system tables not allowed: {', '.join(system_tables)}"]
                )
        
        # Check 4: SELECT * detection
        if self._has_select_star(statement):
            issues.append("SELECT * detected - consider specifying columns")
            suggestions.append("Explicitly list required columns for better performance")
        
        # Check 5: Missing LIMIT clause
        if query_type == QueryType.SELECT:
            has_limit = self._has_limit_clause(statement)
            if not has_limit:
                if self.auto_inject_limit:
                    modified_query = self._inject_limit_clause(modified_query)
                    suggestions.append(f"LIMIT {self.max_result_limit} automatically added")
                else:
                    issues.append("Query missing LIMIT clause")
                    suggestions.append(f"Add LIMIT clause (max: {self.max_result_limit})")
        
        # Check 6: JOIN complexity
        if self.allow_joins:
            join_count = self._count_joins(statement)
            if join_count > self.max_joins:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.BLOCKED,
                    query_type=query_type,
                    issues=[f"Too many JOINs ({join_count}). Maximum allowed: {self.max_joins}"],
                    suggestions=["Simplify query or use subqueries"]
                )
            elif join_count > 2:
                issues.append(f"Complex query with {join_count} JOINs")
                suggestions.append("Consider query optimization")
        
        # Check 7: Subquery depth
        subquery_depth = self._check_subquery_depth(statement)
        if subquery_depth > 3:
            issues.append(f"Deep subquery nesting (depth: {subquery_depth})")
            suggestions.append("Consider simplifying with CTEs or temp tables")
        
        # Check 8: UNION operations
        if 'UNION' in query.upper():
            union_count = query.upper().count('UNION')
            if union_count > 3:
                issues.append(f"Multiple UNION operations ({union_count})")
                suggestions.append("Consider alternative approaches for combining results")
        
        # Determine final severity
        severity = ValidationSeverity.SAFE
        if issues:
            severity = ValidationSeverity.WARNING
        
        return ValidationResult(
            is_valid=True,
            severity=severity,
            query_type=query_type,
            issues=issues,
            modified_query=modified_query if modified_query != query.strip() else None,
            suggestions=suggestions
        )
    
    def _identify_query_type(self, statement: Statement) -> QueryType:
        """Identify the type of SQL query"""
        first_token = statement.token_first(skip_ws=True, skip_cm=True)
        
        if first_token:
            token_value = first_token.value.upper()
            try:
                return QueryType[token_value]
            except KeyError:
                pass
        
        return QueryType.UNKNOWN
    
    def _check_dangerous_keywords(self, query: str) -> List[str]:
        """Check for dangerous SQL keywords"""
        query_upper = query.upper()
        found = []
        
        for keyword in self.DANGEROUS_KEYWORDS:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_upper):
                found.append(keyword)
        
        return found
    
    def _check_system_tables(self, query: str) -> List[str]:
        """Check if query accesses system tables"""
        query_lower = query.lower()
        found = []
        
        for protected in self.PROTECTED_OBJECTS:
            if protected.lower() in query_lower:
                found.append(protected)
        
        return found
    
    def _has_select_star(self, statement: Statement) -> bool:
        """Check if query uses SELECT *"""
        query_str = str(statement).upper()
        # Simple pattern matching for SELECT *
        return bool(re.search(r'SELECT\s+\*', query_str))
    
    def _has_limit_clause(self, statement: Statement) -> bool:
        """Check if query has a LIMIT clause"""
        query_str = str(statement).upper()
        return 'LIMIT' in query_str or 'TOP' in query_str or 'FETCH FIRST' in query_str
    
    def _inject_limit_clause(self, query: str) -> str:
        """Inject LIMIT clause into SELECT query"""
        query = query.strip()
        
        # Remove trailing semicolon if present
        if query.endswith(';'):
            query = query[:-1].strip()
        
        # Add LIMIT clause
        query = f"{query} LIMIT {self.max_result_limit}"
        
        return query
    
    def _count_joins(self, statement: Statement) -> int:
        """Count number of JOIN operations"""
        query_str = str(statement).upper()
        # Count various JOIN types
        joins = len(re.findall(r'\bJOIN\b', query_str))
        return joins
    
    def _check_subquery_depth(self, statement: Statement, depth: int = 0) -> int:
        """Check depth of nested subqueries"""
        max_depth = depth
        query_str = str(statement)
        
        # Simple parenthesis counting (not perfect but sufficient)
        open_parens = query_str.count('(')
        
        # Heuristic: each pair of parentheses might indicate a subquery
        estimated_depth = min(open_parens // 2, 5)  # Cap at 5 for sanity
        
        return max(depth, estimated_depth)


class QuerySafetyWrapper:
    """
    Wrapper class to integrate validation with existing database tools
    """
    
    def __init__(self, validator: SQLValidator):
        self.validator = validator
        self.query_log = []
    
    def validate_and_execute(
        self,
        query: str,
        execute_func: callable
    ) -> Tuple[ValidationResult, Optional[any]]:
        """
        Validate query and execute if safe
        
        Args:
            query: SQL query to validate
            execute_func: Function to execute the query
            
        Returns:
            Tuple of (ValidationResult, execution_result)
        """
        # Validate the query
        validation = self.validator.validate(query)
        
        # Log the query and validation result
        self.query_log.append({
            'query': query,
            'validation': validation,
            'timestamp': __import__('datetime').datetime.now()
        })
        
        # Block dangerous queries
        if validation.severity == ValidationSeverity.BLOCKED:
            raise SecurityError(
                f"Query blocked: {', '.join(validation.issues)}"
            )
        
        # Use modified query if available
        final_query = validation.modified_query or query
        
        # Execute the query
        try:
            result = execute_func(final_query)
            return validation, result
        except Exception as e:
            raise QueryExecutionError(f"Query execution failed: {str(e)}") from e
    
    def get_query_log(self) -> List[dict]:
        """Return query execution log"""
        return self.query_log


class SecurityError(Exception):
    """Raised when a query is blocked for security reasons"""
    pass


class QueryExecutionError(Exception):
    """Raised when query execution fails"""
    pass


# Convenience function for quick validation
def validate_query(
    query: str,
    read_only: bool = True,
    max_limit: int = 1000
) -> ValidationResult:
    """
    Quick validation function
    
    Args:
        query: SQL query to validate
        read_only: Enforce read-only mode
        max_limit: Maximum result limit
        
    Returns:
        ValidationResult
    """
    validator = SQLValidator(
        read_only_mode=read_only,
        max_result_limit=max_limit
    )
    return validator.validate(query)