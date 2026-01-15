"""
Schema metadata models
Represents database schema information for KPI generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


@dataclass
class ColumnInfo:
    """Information about a database column"""
    name: str
    type: str
    nullable: bool
    unique: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    
    def is_numeric(self) -> bool:
        """Check if column is numeric"""
        numeric_types = [
            'integer', 'int', 'bigint', 'smallint', 'tinyint',
            'decimal', 'numeric', 'float', 'double', 'real',
            'money', 'number'
        ]
        return any(t in self.type.lower() for t in numeric_types)
    
    def is_date(self) -> bool:
        """Check if column is date/time"""
        date_types = ['date', 'time', 'timestamp', 'datetime']
        return any(t in self.type.lower() for t in date_types)
    
    def is_text(self) -> bool:
        """Check if column is text"""
        text_types = ['char', 'varchar', 'text', 'string', 'clob']
        return any(t in self.type.lower() for t in text_types)
    
    def __repr__(self) -> str:
        fk_info = f" -> {self.foreign_key}" if self.foreign_key else ""
        pk_info = " [PK]" if self.primary_key else ""
        return f"{self.name}: {self.type}{pk_info}{fk_info}"


@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    row_count: int
    columns: List[ColumnInfo]
    sample_data: Optional[pd.DataFrame] = None
    
    def get_numeric_columns(self) -> List[ColumnInfo]:
        """Get all numeric columns"""
        return [col for col in self.columns if col.is_numeric()]
    
    def get_date_columns(self) -> List[ColumnInfo]:
        """Get all date/time columns"""
        return [col for col in self.columns if col.is_date()]
    
    def get_text_columns(self) -> List[ColumnInfo]:
        """Get all text columns"""
        return [col for col in self.columns if col.is_text()]
    
    def get_column(self, column_name: str) -> Optional[ColumnInfo]:
        """Get column by name"""
        for col in self.columns:
            if col.name.lower() == column_name.lower():
                return col
        return None
    
    def __repr__(self) -> str:
        return f"Table({self.name}, {self.row_count:,} rows, {len(self.columns)} columns)"


@dataclass
class Relationship:
    """Foreign key relationship between tables"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    
    def __repr__(self) -> str:
        return f"{self.from_table}.{self.from_column} -> {self.to_table}.{self.to_column}"


@dataclass
class SchemaMetadata:
    """Complete database schema metadata"""
    database_id: str
    tables: List[TableInfo]
    relationships: List[Relationship]
    analysis_time: float
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get table info by name"""
        for table in self.tables:
            if table.name.lower() == table_name.lower():
                return table
        return None
    
    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """Get all relationships involving a table"""
        relationships = []
        for rel in self.relationships:
            if rel.from_table == table_name or rel.to_table == table_name:
                relationships.append(rel)
        return relationships
    
    def get_table_count(self) -> int:
        """Get number of tables"""
        return len(self.tables)
    
    def get_total_rows(self) -> int:
        """Get total row count across all tables"""
        return sum(table.row_count for table in self.tables)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'database_id': self.database_id,
            'tables': [
                {
                    'name': t.name,
                    'row_count': t.row_count,
                    'columns': [
                        {
                            'name': c.name,
                            'type': c.type,
                            'nullable': c.nullable,
                            'unique': c.unique,
                            'primary_key': c.primary_key,
                            'foreign_key': c.foreign_key
                        }
                        for c in t.columns
                    ]
                }
                for t in self.tables
            ],
            'relationships': [
                {
                    'from_table': r.from_table,
                    'from_column': r.from_column,
                    'to_table': r.to_table,
                    'to_column': r.to_column
                }
                for r in self.relationships
            ],
            'analysis_time': self.analysis_time,
            'analyzed_at': self.analyzed_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return f"SchemaMetadata({self.database_id}, {len(self.tables)} tables, {len(self.relationships)} relationships)"


@dataclass
class BusinessContext:
    """Business context for KPI suggestions"""
    industry: str
    business_model: str
    primary_goals: List[str]
    other: Optional[str] = None
    
    def to_prompt_text(self) -> str:
        """Convert to text for LLM prompt"""
        text = f"Industry: {self.industry}\n"
        text += f"Business Model: {self.business_model}\n"
        text += f"Primary Goals: {', '.join(self.primary_goals)}\n"
        if self.other:
            text += f"Additional Context: {self.other}\n"
        return text
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'industry': self.industry,
            'business_model': self.business_model,
            'primary_goals': self.primary_goals,
            'other': self.other
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BusinessContext':
        """Create from dictionary"""
        return cls(
            industry=data['industry'],
            business_model=data['business_model'],
            primary_goals=data['primary_goals'],
            other=data.get('other')
        )
    
    def __repr__(self) -> str:
        return f"BusinessContext({self.industry}, {self.business_model})"