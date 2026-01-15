"""
Schema Analyzer - Database Schema Analysis with Business Context
Analyzes database structure and prepares metadata for KPI generation
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, Table, text
from sqlalchemy.exc import SQLAlchemyError

from dashboard.models.schema_models import (
    ColumnInfo,
    TableInfo,
    Relationship,
    SchemaMetadata,
    BusinessContext
)

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """
    Analyzes database schema with business context
    
    Limits (APPROVED):
    - Max tables: 100
    - Max rows per table: 50M
    - Sample size: 10,000 rows per table
    - Timeout: 60 seconds
    """
    
    MAX_TABLES = 100
    MAX_SAMPLE_ROWS = 10000
    ANALYSIS_TIMEOUT = 60  # seconds
    
    def __init__(self, database_uri: str):
        """
        Initialize schema analyzer
        
        Args:
            database_uri: SQLAlchemy database connection URI
        """
        self.database_uri = database_uri
        self.engine = None
        self.inspector = None
        
    def connect(self) -> bool:
        """
        Establish database connection
        
        Returns:
            True if successful
        """
        try:
            self.engine = create_engine(self.database_uri)
            self.inspector = inspect(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def analyze_database(
        self,
        database_id: str,
        progress_callback: Optional[callable] = None
    ) -> Optional[SchemaMetadata]:
        """
        Analyze complete database schema
        
        Args:
            database_id: Unique database identifier
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            SchemaMetadata or None if failed
        """
        if not self.engine or not self.inspector:
            if not self.connect():
                return None
        
        start_time = time.time()
        
        try:
            # Get all table names (limit to MAX_TABLES)
            all_tables = self.inspector.get_table_names()
            
            if len(all_tables) > self.MAX_TABLES:
                logger.warning(
                    f"Database has {len(all_tables)} tables. "
                    f"Analyzing first {self.MAX_TABLES} only."
                )
                all_tables = all_tables[:self.MAX_TABLES]
            
            tables = []
            total_tables = len(all_tables)
            
            # Analyze each table
            for idx, table_name in enumerate(all_tables):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.ANALYSIS_TIMEOUT:
                    logger.warning(f"Analysis timeout after {elapsed:.1f}s")
                    break
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        total_tables,
                        f"Analyzing table: {table_name}"
                    )
                
                # Analyze table
                table_info = self._analyze_table(table_name)
                if table_info:
                    tables.append(table_info)
                
            # Find relationships
            if progress_callback:
                progress_callback(total_tables, total_tables, "Finding relationships...")
            
            relationships = self._find_relationships(tables)
            
            # Create metadata
            analysis_time = time.time() - start_time
            
            metadata = SchemaMetadata(
                database_id=database_id,
                tables=tables,
                relationships=relationships,
                analysis_time=analysis_time
            )
            
            logger.info(
                f"Schema analysis complete: {len(tables)} tables, "
                f"{len(relationships)} relationships, {analysis_time:.1f}s"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            return None
    
    def _analyze_table(self, table_name: str) -> Optional[TableInfo]:
        """
        Analyze a single table
        
        Args:
            table_name: Name of table to analyze
            
        Returns:
            TableInfo or None if failed
        """
        try:
            # Get columns
            columns = self._get_columns(table_name)
            
            # Get row count
            row_count = self._get_row_count(table_name)
            
            # Get sample data
            sample_data = self._get_sample_data(table_name)
            
            table_info = TableInfo(
                name=table_name,
                row_count=row_count,
                columns=columns,
                sample_data=sample_data
            )
            
            logger.debug(f"Analyzed table: {table_name} ({row_count:,} rows)")
            return table_info
            
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return None
    
    def _get_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get column information for a table"""
        columns = []
        
        try:
            # Get column details from inspector
            cols = self.inspector.get_columns(table_name)
            
            # Get primary keys
            pk_constraint = self.inspector.get_pk_constraint(table_name)
            pk_columns = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            
            # Get foreign keys
            fk_constraints = self.inspector.get_foreign_keys(table_name)
            fk_map = {}
            for fk in fk_constraints:
                for col in fk.get('constrained_columns', []):
                    fk_map[col] = f"{fk['referred_table']}.{fk['referred_columns'][0]}"
            
            # Get unique constraints
            unique_constraints = self.inspector.get_unique_constraints(table_name)
            unique_columns = set()
            for constraint in unique_constraints:
                unique_columns.update(constraint.get('column_names', []))
            
            # Build column info
            for col in cols:
                col_name = col['name']
                col_type = str(col['type'])
                
                column_info = ColumnInfo(
                    name=col_name,
                    type=col_type,
                    nullable=col.get('nullable', True),
                    unique=col_name in unique_columns,
                    primary_key=col_name in pk_columns,
                    foreign_key=fk_map.get(col_name)
                )
                
                columns.append(column_info)
            
            return columns
            
        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")
            return []
    
    def _get_row_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                return result.scalar()
                
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def _get_sample_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Get sample data from table
        
        Args:
            table_name: Table to sample
            
        Returns:
            DataFrame with sample data (max MAX_SAMPLE_ROWS rows)
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT {self.MAX_SAMPLE_ROWS}"
            
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            
            logger.debug(f"Sampled {len(df)} rows from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get sample data for {table_name}: {e}")
            return None
    
    def _find_relationships(self, tables: List[TableInfo]) -> List[Relationship]:
        """
        Find foreign key relationships between tables
        
        Args:
            tables: List of analyzed tables
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for table in tables:
            for column in table.columns:
                if column.foreign_key:
                    # Parse foreign key: "table.column"
                    parts = column.foreign_key.split('.')
                    if len(parts) == 2:
                        to_table, to_column = parts
                        
                        relationship = Relationship(
                            from_table=table.name,
                            from_column=column.name,
                            to_table=to_table,
                            to_column=to_column
                        )
                        
                        relationships.append(relationship)
        
        return relationships
    
    def format_schema_for_llm(self, metadata: SchemaMetadata) -> str:
        """
        Format schema metadata for LLM prompt
        
        Args:
            metadata: Schema metadata
            
        Returns:
            Formatted string for LLM
        """
        output = []
        output.append(f"Database: {metadata.database_id}")
        output.append(f"Tables: {metadata.get_table_count()}")
        output.append(f"Total Rows: {metadata.get_total_rows():,}")
        output.append("\n" + "="*60 + "\n")
        
        # Format each table
        for table in metadata.tables:
            output.append(f"\nTABLE: {table.name}")
            output.append(f"Rows: {table.row_count:,}")
            output.append("\nColumns:")
            
            for col in table.columns:
                col_desc = f"  - {col.name} ({col.type})"
                
                if col.primary_key:
                    col_desc += " [PRIMARY KEY]"
                if col.foreign_key:
                    col_desc += f" [FK -> {col.foreign_key}]"
                if col.unique:
                    col_desc += " [UNIQUE]"
                if not col.nullable:
                    col_desc += " [NOT NULL]"
                
                output.append(col_desc)
            
            # Show sample data statistics
            if table.sample_data is not None and not table.sample_data.empty:
                output.append("\nSample Data Statistics:")
                
                # Numeric columns
                numeric_cols = table.get_numeric_columns()
                if numeric_cols:
                    output.append("  Numeric columns:")
                    for col in numeric_cols[:5]:  # Limit to 5
                        if col.name in table.sample_data.columns:
                            series = table.sample_data[col.name]
                            output.append(
                                f"    {col.name}: "
                                f"min={series.min()}, max={series.max()}, "
                                f"mean={series.mean():.2f}"
                            )
                
                # Date columns
                date_cols = table.get_date_columns()
                if date_cols:
                    output.append("  Date columns:")
                    for col in date_cols[:3]:
                        if col.name in table.sample_data.columns:
                            series = pd.to_datetime(table.sample_data[col.name], errors="coerce")
                            output.append(
                                f"    {col.name}: "
                                f"range={series.min()} to {series.max()}"
                            )
            
            output.append("\n" + "-"*60)
        
        # Relationships
        if metadata.relationships:
            output.append("\n\nRELATIONSHIPS:")
            for rel in metadata.relationships:
                output.append(f"  {rel}")
        
        return "\n".join(output)
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


class SchemaAnalyzerError(Exception):
    """Base exception for schema analyzer errors"""
    pass


class TimeoutError(SchemaAnalyzerError):
    """Raised when analysis exceeds timeout"""
    pass


class ConnectionError(SchemaAnalyzerError):
    """Raised when database connection fails"""
    pass
