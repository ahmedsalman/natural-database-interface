"""
KPI Engine - AI-Powered KPI Generation with Business Context
Generates and validates KPI suggestions using LLM
"""

import json
import logging
import re
from typing import List, Dict, Optional, Tuple
import openai
from anthropic import Anthropic

from dashboard.models.schema_models import SchemaMetadata, BusinessContext
from dashboard.models.kpi_models import KPI
from model_config import get_model_config, get_provider_for_model, LLMProvider
from common import get_api_key

logger = logging.getLogger(__name__)


class KPIEngine:
    """
    Generates and validates KPI suggestions
    
    Process (APPROVED):
    1. Analyze schema + business context
    2. Generate 20 KPI suggestions via LLM
    3. Pre-validate each KPI SQL
    4. Return only valid KPIs
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Best for schema analysis
    SUGGESTION_COUNT = 20
    
    def __init__(self, model_id: str = None):
        """
        Initialize KPI engine
        
        Args:
            model_id: LLM model to use (default: Claude 3.5 Sonnet)
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.model_config = get_model_config(self.model_id)
        self.provider = get_provider_for_model(self.model_id)
        
        if not self.model_config:
            raise ValueError(f"Unknown model: {self.model_id}")
        
        # Initialize LLM client
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on provider"""
        api_key = get_api_key(self.provider)
        
        if not api_key:
            raise ValueError(f"API key not set for {self.provider.value}")
        
        if self.provider == LLMProvider.OPENAI:
            openai.api_key = api_key
            self.client = None  # Use openai module directly
        elif self.provider == LLMProvider.ANTHROPIC:
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def suggest_kpis(
        self,
        schema_metadata: SchemaMetadata,
        business_context: BusinessContext,
        database_uri: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, any]:
        """
        Generate KPI suggestions with validation
        
        Args:
            schema_metadata: Database schema information
            business_context: Business context for suggestions
            database_uri: Database connection URI for validation
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            {
                'kpis': [KPI],  # Only validated KPIs
                'total_suggested': int,
                'validation_failures': [{'name': str, 'error': str}]
            }
        """
        # Detect database type from URI
        db_type = self._detect_database_type(database_uri)
        logger.info(f"Detected database type: {db_type}")
        
        # Step 1: Generate suggestions
        if progress_callback:
            progress_callback(0, 3, "Generating KPI suggestions...")
        
        suggested_kpis = self._generate_suggestions(
            schema_metadata,
            business_context,
            db_type
        )
        
        logger.info(f"Generated {len(suggested_kpis)} KPI suggestions")
        
        # Step 2: Pre-validate each KPI
        if progress_callback:
            progress_callback(1, 3, f"Validating {len(suggested_kpis)} KPIs...")
        
        valid_kpis = []
        failures = []
        
        from dashboard.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer(database_uri)
        
        if not analyzer.connect():
            logger.error("Failed to connect for validation")
            return {
                'kpis': [],
                'total_suggested': len(suggested_kpis),
                'validation_failures': [{'name': 'CONNECTION', 'error': 'Database connection failed'}]
            }
        
        for idx, kpi in enumerate(suggested_kpis):
            if progress_callback:
                progress_callback(
                    idx + 1,
                    len(suggested_kpis),
                    f"Validating KPI {idx+1}/{len(suggested_kpis)}: {kpi.name}"
                )
            
            try:
                # Validate SQL
                self._validate_kpi_sql(kpi, analyzer)
                valid_kpis.append(kpi)
                logger.debug(f"✓ KPI valid: {kpi.name}")
                
            except Exception as e:
                import pdb; pdb.set_trace()
                failures.append({
                    'name': kpi.name,
                    'error': str(e)
                })
                logger.warning(f"✗ KPI invalid: {kpi.name} - {e}")
        
        analyzer.close()
        
        # Step 3: Return results
        if progress_callback:
            progress_callback(3, 3, f"Complete: {len(valid_kpis)}/{len(suggested_kpis)} valid")
        
        logger.info(
            f"KPI validation complete: {len(valid_kpis)}/{len(suggested_kpis)} valid"
        )
        
        return {
            'kpis': valid_kpis,
            'total_suggested': len(suggested_kpis),
            'validation_failures': failures
        }
    
    def _generate_suggestions(
        self,
        schema_metadata: SchemaMetadata,
        business_context: BusinessContext,
        db_type: str
    ) -> List[KPI]:
        """
        Generate KPI suggestions using LLM
        
        Args:
            schema_metadata: Database schema
            business_context: Business context
            
        Returns:
            List of suggested KPIs (unvalidated)
        """
        # Format schema for LLM
        from dashboard.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer("")
        schema_text = analyzer.format_schema_for_llm(schema_metadata)
        
        # Build prompt
        prompt = self._build_kpi_generation_prompt(
            schema_text,
            business_context,
            db_type
        )
        
        # Call LLM
        response_text = self._call_llm(prompt)
        
        # Parse response
        kpis = self._parse_kpi_response(response_text)
        
        return kpis
    
    def _build_kpi_generation_prompt(
        self,
        schema_text: str,
        business_context: BusinessContext,
        db_type: str
    ) -> str:
        """
        Build KPI generation prompt with database-specific SQL syntax
        
        Args:
            schema_text: Formatted schema
            business_context: Business context
            db_type: Database type (postgresql, mysql, sqlserver, oracle, sqlite)
            
        Returns:
            Prompt text
        """
        # Database-specific SQL syntax for date/time operations
        sql_examples = self._get_sql_syntax_examples(db_type)
        time_filter_example = self._get_time_filter_example(db_type)
        example_kpi = self._get_example_kpi_sql(db_type)
        
        prompt = f"""You are a data analytics expert. Generate {self.SUGGESTION_COUNT} relevant Key Performance Indicators (KPIs) for this database.

DATABASE TYPE: {db_type.upper()}

DATABASE SCHEMA:
{schema_text}

BUSINESS CONTEXT:
{business_context.to_prompt_text()}

REQUIREMENTS:
1. Generate exactly {self.SUGGESTION_COUNT} KPIs
2. Cover different categories: financial, operational, customer, product
3. Use ACTUAL table and column names from the schema
4. For time-based KPIs, use placeholder: {{{{time_range_days}}}}
5. Include appropriate aggregations (SUM, AVG, COUNT, etc.)
6. Add LIMIT clauses to queries returning multiple rows
7. Ensure SQL is valid {db_type.upper()} syntax

For each KPI, provide:
- name: Clear, descriptive name (e.g., "Total Revenue", "Customer Acquisition Rate")
- description: What it measures and why it matters for this business
- sql: Valid {db_type.upper()} SQL query using exact table/column names
- category: one of [financial, operational, customer, product]
- chart_type: one of [metric, line, bar, pie, table]
- has_time_parameter: true if SQL uses {{{{time_range_days}}}}
- default_time_range_days: 30, 90, 365, or null

CRITICAL: {db_type.upper()} SQL SYNTAX RULES
{sql_examples}

IMPORTANT SQL RULES:
- Use EXACT table and column names from schema (case-sensitive for PostgreSQL)
- For time-based queries: {time_filter_example}
- Include proper JOINs when using multiple tables
- Add appropriate WHERE clauses for filtering
- Use LIMIT for queries returning multiple rows (or TOP for SQL Server, ROWNUM for Oracle)
- Test logic: Would this SQL actually work on {db_type.upper()}?

OUTPUT FORMAT:
Return ONLY a valid JSON array with no preamble or explanation:
[
  {{{{
    "name": "Total Revenue",
    "description": "Sum of all completed order amounts. Critical for tracking overall business performance.",
    "sql": "{example_kpi}",
    "category": "financial",
    "chart_type": "metric",
    "has_time_parameter": true,
    "default_time_range_days": 30
  }}}},
  ...
]

Generate {self.SUGGESTION_COUNT} KPIs now as a JSON array:"""

        return prompt
    
    def _detect_database_type(self, database_uri: str) -> str:
        """
        Detect database type from connection URI
        
        Args:
            database_uri: Database connection string
            
        Returns:
            Database type: 'postgresql', 'mysql', 'sqlserver', 'oracle', or 'sqlite'
        """
        uri_lower = database_uri.lower()
        
        if uri_lower.startswith('postgresql://') or uri_lower.startswith('postgres://'):
            return 'postgresql'
        elif uri_lower.startswith('mysql://') or uri_lower.startswith('mariadb://'):
            return 'mysql'
        elif uri_lower.startswith('mssql://') or uri_lower.startswith('sqlserver://'):
            return 'sqlserver'
        elif uri_lower.startswith('oracle://'):
            return 'oracle'
        elif uri_lower.startswith('sqlite://'):
            return 'sqlite'
        else:
            # Default to PostgreSQL
            logger.warning(f"Unknown database type in URI: {database_uri}, defaulting to PostgreSQL")
            return 'postgresql'
    
    def _get_sql_syntax_examples(self, db_type: str) -> str:
        """
        Get database-specific SQL syntax examples
        
        Args:
            db_type: Database type
            
        Returns:
            SQL syntax examples as string
        """
        if db_type == 'postgresql':
            return """- Date arithmetic: NOW() - INTERVAL '30 days'
- String quotes: Use single quotes for strings
- Date extraction: EXTRACT(DAY FROM date_column)
- Date difference: date_column1 - date_column2 (returns interval)
- Cast: CAST(column AS type) or column::type
- String concatenation: column1 || column2
- Example: WHERE rental_date > NOW() - INTERVAL '{{{{time_range_days}}}} days'"""
        
        elif db_type == 'mysql':
            return """- Date arithmetic: NOW() - INTERVAL {{{{time_range_days}}}} DAY
- String quotes: Use single quotes for strings
- Date extraction: DAY(date_column), MONTH(date_column), YEAR(date_column)
- Date difference: DATEDIFF(date1, date2) returns days
- Cast: CAST(column AS type) or CONVERT(column, type)
- String concatenation: CONCAT(column1, column2)
- Example: WHERE rental_date > NOW() - INTERVAL {{{{time_range_days}}}} DAY"""
        
        elif db_type == 'sqlserver':
            return """- Date arithmetic: DATEADD(day, -{{{{time_range_days}}}}, GETDATE())
- String quotes: Use single quotes for strings
- Date extraction: DAY(date_column), MONTH(date_column), YEAR(date_column)
- Date difference: DATEDIFF(day, date1, date2)
- Cast: CAST(column AS type) or CONVERT(type, column)
- String concatenation: column1 + column2 or CONCAT(column1, column2)
- LIMIT replacement: SELECT TOP 10 * FROM table
- Example: WHERE rental_date > DATEADD(day, -{{{{time_range_days}}}}, GETDATE())"""
        
        elif db_type == 'oracle':
            return """- Date arithmetic: SYSDATE - {{{{time_range_days}}}} or SYSDATE - INTERVAL '{{{{time_range_days}}}}' DAY
- String quotes: Use single quotes for strings
- Date extraction: EXTRACT(DAY FROM date_column)
- Date difference: date1 - date2 (returns number of days)
- Cast: CAST(column AS type)
- String concatenation: column1 || column2
- LIMIT replacement: WHERE ROWNUM <= 10
- Example: WHERE rental_date > SYSDATE - {{{{time_range_days}}}}"""
        
        else:  # sqlite
            return """- Date arithmetic: datetime('now', '-{{{{time_range_days}}}} days')
- String quotes: Use single quotes for strings
- Date extraction: strftime('%d', date_column)
- Date difference: julianday(date1) - julianday(date2)
- Cast: CAST(column AS type)
- String concatenation: column1 || column2
- Example: WHERE rental_date > datetime('now', '-{{{{time_range_days}}}} days')"""
    
    def _get_time_filter_example(self, db_type: str) -> str:
        """
        Get database-specific time filter example
        
        Args:
            db_type: Database type
            
        Returns:
            Example WHERE clause for time filtering
        """
        if db_type == 'postgresql':
            return "WHERE date_column > NOW() - INTERVAL '{{{{time_range_days}}}} days'"
        elif db_type == 'mysql':
            return "WHERE date_column > NOW() - INTERVAL {{{{time_range_days}}}} DAY"
        elif db_type == 'sqlserver':
            return "WHERE date_column > DATEADD(day, -{{{{time_range_days}}}}, GETDATE())"
        elif db_type == 'oracle':
            return "WHERE date_column > SYSDATE - {{{{time_range_days}}}}"
        else:  # sqlite
            return "WHERE date_column > datetime('now', '-{{{{time_range_days}}}} days')"
    
    def _get_example_kpi_sql(self, db_type: str) -> str:
        """
        Get database-specific example KPI SQL
        
        Args:
            db_type: Database type
            
        Returns:
            Example SQL query
        """
        if db_type == 'postgresql':
            return "SELECT SUM(amount) as revenue FROM orders WHERE status = 'completed' AND order_date > NOW() - INTERVAL '{{{{time_range_days}}}} days'"
        elif db_type == 'mysql':
            return "SELECT SUM(amount) as revenue FROM orders WHERE status = 'completed' AND order_date > NOW() - INTERVAL {{{{time_range_days}}}} DAY"
        elif db_type == 'sqlserver':
            return "SELECT SUM(amount) as revenue FROM orders WHERE status = 'completed' AND order_date > DATEADD(day, -{{{{time_range_days}}}}, GETDATE())"
        elif db_type == 'oracle':
            return "SELECT SUM(amount) as revenue FROM orders WHERE status = 'completed' AND order_date > SYSDATE - {{{{time_range_days}}}}"
        else:  # sqlite
            return "SELECT SUM(amount) as revenue FROM orders WHERE status = 'completed' AND order_date > datetime('now', '-{{{{time_range_days}}}} days')"
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt
        
        Args:
            prompt: Prompt text
            
        Returns:
            LLM response text
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                response = openai.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data analytics expert. Generate KPI suggestions as valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                return response.content[0].text
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise KPIGenerationError(f"Failed to generate KPIs: {e}")
    
    def _parse_kpi_response(self, response_text: str) -> List[KPI]:
        """
        Parse LLM response into KPI objects
        
        Args:
            response_text: LLM response
            
        Returns:
            List of KPI objects
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response_text.strip()
            
            # Remove ```json and ``` if present
            if cleaned.startswith('```'):
                # Find first [ and last ]
                start = cleaned.find('[')
                end = cleaned.rfind(']') + 1
                if start != -1 and end > start:
                    cleaned = cleaned[start:end]
            
            # Parse JSON
            kpis_data = json.loads(cleaned)
            
            if not isinstance(kpis_data, list):
                raise ValueError("Response is not a JSON array")
            
            # Convert to KPI objects
            kpis = []
            for kpi_data in kpis_data:
                try:
                    kpi = KPI(
                        name=kpi_data['name'],
                        description=kpi_data['description'],
                        sql_template=kpi_data['sql'].strip(),
                        category=kpi_data.get('category', 'custom'),
                        chart_type=kpi_data.get('chart_type', 'metric'),
                        has_time_parameter=kpi_data.get('has_time_parameter', False),
                        default_time_range_days=kpi_data.get('default_time_range_days')
                    )
                    
                    # Validate KPI structure
                    is_valid, error = kpi.validate()
                    if is_valid:
                        kpis.append(kpi)
                    else:
                        logger.warning(f"Invalid KPI structure: {kpi.name} - {error}")
                        
                except KeyError as e:
                    logger.warning(f"Missing required field in KPI: {e}")
                except Exception as e:
                    logger.warning(f"Failed to parse KPI: {e}")
            
            logger.info(f"Parsed {len(kpis)} KPIs from LLM response")
            return kpis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            raise KPIGenerationError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Failed to parse KPI response: {e}")
            raise KPIGenerationError(f"Failed to parse KPIs: {e}")
    
    def _validate_kpi_sql(self, kpi: KPI, analyzer) -> bool:
        """
        Validate KPI SQL by executing test query
        
        Args:
            kpi: KPI to validate
            analyzer: SchemaAnalyzer instance with active connection
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if SQL is invalid
        """
        try:
            # Get executable SQL
            test_sql = kpi.get_sql()
            
            # Add LIMIT 1 for testing
            if 'LIMIT' not in test_sql.upper():
                test_sql = f"{test_sql.rstrip(';')} LIMIT 1"
            
            # Execute test query
            with analyzer.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(test_sql))
                
                # Fetch one row to ensure query works
                row = result.fetchone()
            
            return True
            
        except Exception as e:
            # Clean up error message
            error_msg = str(e).split('\n')[0]  # First line only
            raise ValidationError(f"SQL validation failed: {error_msg}")
    
    def create_custom_kpi(
        self,
        user_request: str,
        schema_metadata: SchemaMetadata,
        business_context: BusinessContext,
        database_uri: str
    ) -> Optional[KPI]:
        """
        Create a single custom KPI based on user request
        
        Args:
            user_request: Natural language KPI request
            schema_metadata: Database schema
            business_context: Business context
            database_uri: Database URI for validation
            
        Returns:
            KPI if successful, None if failed
        """
        logger.info(f"Creating custom KPI: {user_request}")
        
        # Format schema
        from dashboard.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer(database_uri)
        schema_text = analyzer.format_schema_for_llm(schema_metadata)
        
        # Build prompt
        prompt = f"""You are a data analytics expert. Create a single KPI based on this request.

USER REQUEST:
{user_request}

DATABASE SCHEMA:
{schema_text}

BUSINESS CONTEXT:
{business_context.to_prompt_text()}

Generate ONE KPI that fulfills this request using the provided schema.

REQUIREMENTS:
- Use ACTUAL table and column names from schema
- For time-based KPIs, use placeholder: {{time_range_days}}
- Ensure SQL is valid and will execute successfully
- Choose appropriate chart type for the data

OUTPUT FORMAT (JSON only, no explanation):
{{
  "name": "KPI Name",
  "description": "What this measures",
  "sql": "SELECT ... FROM ...",
  "category": "financial|operational|customer|product",
  "chart_type": "metric|line|bar|pie|table",
  "has_time_parameter": true/false,
  "default_time_range_days": 30 or null
}}"""

        try:
            # Call LLM
            response_text = self._call_llm(prompt)
            
            # Parse response
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                if start != -1 and end > start:
                    cleaned = cleaned[start:end]
            
            kpi_data = json.loads(cleaned)
            
            # Create KPI
            kpi = KPI(
                name=kpi_data['name'],
                description=kpi_data['description'],
                sql_template=kpi_data['sql'].strip(),
                category=kpi_data.get('category', 'custom'),
                chart_type=kpi_data.get('chart_type', 'metric'),
                has_time_parameter=kpi_data.get('has_time_parameter', False),
                default_time_range_days=kpi_data.get('default_time_range_days')
            )
            
            # Validate structure
            is_valid, error = kpi.validate()
            if not is_valid:
                logger.error(f"Invalid KPI structure: {error}")
                return None
            
            # Validate SQL
            if not analyzer.connect():
                logger.error("Cannot validate - connection failed")
                return None
            
            try:
                self._validate_kpi_sql(kpi, analyzer)
                logger.info(f"Custom KPI created: {kpi.name}")
                return kpi
            finally:
                analyzer.close()
                
        except Exception as e:
            logger.error(f"Failed to create custom KPI: {e}")
            return None


class KPIGenerationError(Exception):
    """Raised when KPI generation fails"""
    pass


class ValidationError(Exception):
    """Raised when KPI validation fails"""
    pass