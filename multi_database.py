"""
Enhanced multi-database tool with integrated SQL safety validation
"""

from typing import Callable, Dict, Iterable, List, Optional
from llama_index.tools.database import DatabaseToolSpec
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from sqlalchemy import text
from sqlalchemy.exc import InvalidRequestError
import logging

# Import safety modules
from sql_validator import SQLValidator, ValidationResult, ValidationSeverity, SecurityError
from config import Config, get_preset_config

# Setup logging
logger = logging.getLogger(__name__)


class NoSuchDatabaseError(InvalidRequestError):
    """Database does not exist or is not visible to a connection."""


class QueryBlockedError(Exception):
    """Raised when a query is blocked by safety validation"""


class TrackingDatabaseToolSpec(DatabaseToolSpec):
    """
    Enhanced DatabaseToolSpec with SQL validation and safety checks
    """
    
    handler: Callable[[str, str, Iterable], None]
    database_name: str
    config: Config
    validator: SQLValidator
    query_log: List[Dict]
    
    def __init__(
        self,
        uri: str = None,
        engine: any = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """
        Initialize with safety configuration
        
        Args:
            uri: Database connection URI
            engine: SQLAlchemy engine (alternative to uri)
            config: Safety configuration (defaults to 'production' preset)
            **kwargs: Additional arguments for DatabaseToolSpec
        """
        super().__init__(uri=uri, engine=engine, **kwargs)
        
        # Initialize safety configuration
        self.config = config or get_preset_config('production')
        
        # Initialize SQL validator
        self.validator = SQLValidator(
            read_only_mode=self.config.read_only_mode,
            max_result_limit=self.config.max_result_limit,
            auto_inject_limit=self.config.auto_inject_limit,
            allow_joins=self.config.allow_joins,
            max_joins=self.config.max_joins,
            block_system_tables=self.config.block_system_tables,
            enable_explain=self.config.enable_explain
        )
        
        # Initialize query log
        self.query_log = []
        
        logger.info(f"Initialized TrackingDatabaseToolSpec with {self.config.read_only_mode and 'READ-ONLY' or 'READ-WRITE'} mode")
    
    def set_handler(self, func: Callable) -> None:
        """Set query execution handler"""
        self.handler = func
    
    def set_database_name(self, database_name: str) -> None:
        """Set database identifier"""
        self.database_name = database_name
    
    def set_config(self, config: Config) -> None:
        """
        Update safety configuration
        
        Args:
            config: New Config instance
        """
        self.config = config
        # Reinitialize validator with new config
        self.validator = SQLValidator(
            read_only_mode=config.read_only_mode,
            max_result_limit=config.max_result_limit,
            auto_inject_limit=config.auto_inject_limit,
            allow_joins=config.allow_joins,
            max_joins=config.max_joins,
            block_system_tables=config.block_system_tables,
            enable_explain=config.enable_explain
        )
        logger.info(f"Updated safety config for {self.database_name}")
    
    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate SQL query before execution
        
        Args:
            query: SQL query string
            
        Returns:
            ValidationResult with validation details
        """
        return self.validator.validate(query)
    
    def load_data(self, query: str) -> List[Document]:
        """
        Query and load data with safety validation
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            List[Document]: Query results as documents
            
        Raises:
            QueryBlockedError: If query is blocked by safety checks
            SecurityError: If query contains dangerous patterns
        """
        import datetime
        
        # Validate the query
        validation = self.validate_query(query)
        
        # Log validation attempt
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'database': self.database_name,
            'original_query': query,
            'validation': validation,
            'executed': False
        }
        
        # Block dangerous queries
        if validation.severity == ValidationSeverity.BLOCKED:
            log_entry['blocked'] = True
            log_entry['block_reason'] = validation.issues
            self.query_log.append(log_entry)
            
            if self.config.log_blocked_queries:
                logger.warning(
                    f"Query BLOCKED for {self.database_name}: {validation.issues}"
                )
            
            raise QueryBlockedError(
                f"Query blocked by safety validation: {', '.join(validation.issues)}"
            )
        
        # Use modified query if validator made changes
        final_query = validation.modified_query or query
        
        # Log warnings
        if validation.severity == ValidationSeverity.WARNING:
            logger.warning(
                f"Query WARNING for {self.database_name}: {validation.issues}"
            )
        
        # Execute the query
        documents = []
        try:
            with self.sql_database.engine.connect() as connection:
                result = connection.execute(text(final_query))
                items = result.fetchall()
                
                # Call handler if set
                if self.handler:
                    self.handler(self.database_name, final_query, items)
                
                # Convert results to documents
                for item in items:
                    doc_str = ", ".join([str(entry) for entry in item])
                    documents.append(Document(text=doc_str))
            
            # Log successful execution
            log_entry['executed'] = True
            log_entry['final_query'] = final_query
            log_entry['result_count'] = len(documents)
            log_entry['warnings'] = validation.issues if validation.issues else []
            
            if self.config.log_all_queries:
                logger.info(
                    f"Query executed for {self.database_name}: "
                    f"{len(documents)} results, "
                    f"{len(validation.issues)} warnings"
                )
        
        except Exception as e:
            log_entry['error'] = str(e)
            logger.error(f"Query execution error for {self.database_name}: {str(e)}")
            raise
        
        finally:
            self.query_log.append(log_entry)
        
        return documents
    
    def get_query_log(self) -> List[Dict]:
        """
        Retrieve query execution log
        
        Returns:
            List of query log entries
        """
        return self.query_log
    
    def get_safety_stats(self) -> Dict:
        """
        Get safety statistics
        
        Returns:
            Dictionary with safety metrics
        """
        total_queries = len(self.query_log)
        blocked_queries = sum(1 for log in self.query_log if log.get('blocked', False))
        executed_queries = sum(1 for log in self.query_log if log.get('executed', False))
        queries_with_warnings = sum(1 for log in self.query_log if log.get('warnings', []))
        
        return {
            'total_queries': total_queries,
            'blocked_queries': blocked_queries,
            'executed_queries': executed_queries,
            'queries_with_warnings': queries_with_warnings,
            'safety_mode': 'READ-ONLY' if self.config.read_only_mode else 'READ-WRITE',
            'max_result_limit': self.config.max_result_limit
        }


class MultiDatabaseToolSpec(BaseToolSpec, BaseReader):
    """
    Enhanced multi-database tool with safety validation
    """
    
    database_specs: Dict[str, TrackingDatabaseToolSpec]
    handler: Callable[[str, str, Iterable], None]
    global_config: Config
    
    spec_functions = ["load_data", "describe_tables", "list_tables", "list_databases"]
    
    def __init__(
        self,
        database_toolspec_mapping: Optional[Dict[str, TrackingDatabaseToolSpec]] = None,
        handler: Optional[Callable[[str, str, Iterable], None]] = None,
        config: Optional[Config] = None,
        safety_preset: str = 'production'
    ) -> None:
        """
        Initialize multi-database tool with safety configuration
        
        Args:
            database_toolspec_mapping: Existing database specs
            handler: Query execution handler
            config: Safety configuration for all databases
            safety_preset: Preset name if config not provided
        """
        self.database_specs = database_toolspec_mapping or dict()
        self.handler = handler
        
        # Set global safety configuration
        if config:
            self.global_config = config
        else:
            self.global_config= get_preset_config(safety_preset)
        
        # Apply safety config to existing specs
        for spec in self.database_specs.values():
            spec.set_handler(self.handler)
            if not hasattr(spec, 'config'):
                spec.set_config(self.global_config)
        
        logger.info(
            f"Initialized MultiDatabaseToolSpec with {len(self.database_specs)} databases, "
            f"safety preset: {safety_preset}"
        )
    
    def add_connection(
        self,
        database_name: str,
        uri: str,
        config: Optional[Config] = None
    ) -> None:
        """
        Add a database connection with safety configuration
        
        Args:
            database_name: Identifier for the database
            uri: Connection URI
            config: Optional per-database safety config
        """
        config = config or self.global_config
        
        spec = TrackingDatabaseToolSpec(uri=uri, config=config)
        spec.set_handler(self.handler)
        spec.set_database_name(database_name)
        
        self.database_specs[database_name] = spec
        logger.info(f"Added database connection: {database_name}")
    
    def add_database_tool_spec(
        self,
        database_name: str,
        tool_spec: TrackingDatabaseToolSpec
    ) -> None:
        """
        Add an existing database tool spec
        
        Args:
            database_name: Identifier for the database
            tool_spec: TrackingDatabaseToolSpec instance
        """
        tool_spec.set_handler(self.handler)
        tool_spec.set_database_name(database_name)
        
        # Apply global config if spec doesn't have one
        if not hasattr(tool_spec, 'config'):
            tool_spec.set_config(self.global_config)
        
        self.database_specs[database_name] = tool_spec
        logger.info(f"Added database tool spec: {database_name}")
    
    def update_config(
        self,
        database_name: Optional[str] = None,
        config: Optional[Config] = None,
        preset: Optional[str] = None
    ) -> None:
        """
        Update safety configuration
        
        Args:
            database_name: Specific database to update (None for all)
            config: New Config instance
            preset: Preset name to use
        """
        if not config and not preset:
            raise ValueError("Either config or preset must be provided")
        
        new_config = config or get_preset_config(preset)
        
        if database_name:
            if database_name in self.database_specs:
                self.database_specs[database_name].set_config(new_config)
                logger.info(f"Updated safety config for {database_name}")
        else:
            # Update all databases
            self.global_config = new_config
            for spec in self.database_specs.values():
                spec.set_config(new_config)
            logger.info("Updated safety config for all databases")
    
    def load_data(self, database: str, query: str) -> List[Document]:
        """
        Query and load data with safety validation
        
        Args:
            database (str): Database name
            query (str): SQL query
            
        Returns:
            List[Document]: Query results
            
        Raises:
            NoSuchDatabaseError: If database doesn't exist
            QueryBlockedError: If query is blocked
        """
        if database not in self.database_specs:
            raise NoSuchDatabaseError(f"Database '{database}' does not exist.")
        
        return self.database_specs[database].load_data(query)
    
    def describe_tables(self, database: str, tables: Optional[List[str]] = None) -> str:
        """
        Describe tables in the database
        
        Args:
            database (str): Database name
            tables (List[str]): Table names
            
        Returns:
            str: Table descriptions
        """
        if database not in self.database_specs:
            raise NoSuchDatabaseError(f"Database '{database}' does not exist.")
        
        return self.database_specs[database].describe_tables(tables)
    
    def list_tables(self, database: str) -> List[str]:
        """
        List tables in the database
        
        Args:
            database (str): Database name
            
        Returns:
            List[str]: Table names
        """
        if database not in self.database_specs:
            raise NoSuchDatabaseError(f"Database '{database}' does not exist.")
        
        return self.database_specs[database].list_tables()
    
    def list_databases(self) -> List[str]:
        """
        List all available databases
        
        Returns:
            List[str]: Database names
        """
        return list(self.database_specs.keys())
    
    def get_all_query_logs(self) -> Dict[str, List[Dict]]:
        """
        Get query logs for all databases
        
        Returns:
            Dictionary mapping database names to their query logs
        """
        return {
            db_name: spec.get_query_log()
            for db_name, spec in self.database_specs.items()
        }
    
    def get_global_safety_stats(self) -> Dict:
        """
        Get aggregated safety statistics across all databases
        
        Returns:
            Dictionary with global safety metrics
        """
        all_stats = {
            db_name: spec.get_safety_stats()
            for db_name, spec in self.database_specs.items()
        }
        
        total_queries = sum(stats['total_queries'] for stats in all_stats.values())
        total_blocked = sum(stats['blocked_queries'] for stats in all_stats.values())
        total_executed = sum(stats['executed_queries'] for stats in all_stats.values())
        total_warnings = sum(stats['queries_with_warnings'] for stats in all_stats.values())
        
        return {
            'total_databases': len(self.database_specs),
            'total_queries': total_queries,
            'total_blocked': total_blocked,
            'total_executed': total_executed,
            'total_warnings': total_warnings,
            'per_database_stats': all_stats,
            'global_safety_mode': 'READ-ONLY' if self.global_config.read_only_mode else 'READ-WRITE'
        }