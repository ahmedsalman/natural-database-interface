"""
KPI Executor - Query Execution with Caching
Executes KPI queries and manages results with Redis caching
"""

import hashlib
import json
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from dashboard.models.kpi_models import KPI

logger = logging.getLogger(__name__)


class KPIExecutor:
    """
    Executes KPI queries with caching support
    
    Caching strategy (APPROVED):
    - TTL-based expiry
    - Manual refresh option
    - Cache key: kpi_id + parameters hash
    """
    
    # Default TTL values (seconds)
    DEFAULT_TTL = {
        'realtime': 300,      # 5 minutes
        'hourly': 3600,       # 1 hour
        'daily': 86400,       # 1 day
        'historical': 604800   # 7 days
    }
    
    # Query timeout
    QUERY_TIMEOUT = 30  # seconds
    
    def __init__(self, database_uri: str, use_cache: bool = True):
        """
        Initialize KPI executor
        
        Args:
            database_uri: Database connection URI
            use_cache: Whether to use caching (default: True)
        """
        self.database_uri = database_uri
        self.use_cache = use_cache
        self.engine = None
        self.cache = None
        
        # Initialize cache if enabled
        if use_cache:
            self._init_cache()
        
        logger.info(f"KPI executor initialized (cache: {use_cache})")
    
    def _init_cache(self):
        """Initialize Redis cache (optional)"""
        try:
            import redis
            self.cache = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.cache.ping()
            logger.info("Redis cache connected")
            
        except ImportError:
            import pdb; pdb.set_trace()
            logger.warning("Redis not installed - caching disabled")
            self.use_cache = False
            self.cache = None
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - caching disabled")
            self.use_cache = False
            self.cache = None
    
    def connect(self) -> bool:
        """
        Establish database connection
        
        Returns:
            True if successful
        """
        try:
            self.engine = create_engine(
                self.database_uri,
                pool_pre_ping=True,  # Verify connections
                pool_recycle=3600    # Recycle connections hourly
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def execute_kpi(
        self,
        kpi: KPI,
        time_range_days: Optional[int] = None,
        bypass_cache: bool = False,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute KPI query with caching
        
        Args:
            kpi: KPI to execute
            time_range_days: Override default time range
            bypass_cache: Skip cache (manual refresh)
            custom_params: Additional query parameters
            
        Returns:
            Query result as DataFrame
            
        Raises:
            ExecutionError: If query fails
        """
        # Connect if not already connected
        if not self.engine:
            if not self.connect():
                raise ExecutionError("Cannot connect to database")
        
        # Build parameters
        params = custom_params or {}
        if kpi.has_time_parameter:
            params['time_range_days'] = time_range_days or kpi.default_time_range_days or 30
        
        # Check cache (if enabled and not bypassed)
        if self.use_cache and not bypass_cache:
            cached_result = self._get_from_cache(kpi.id, params)
            if cached_result is not None:
                logger.debug(f"Cache hit: {kpi.name}")
                return cached_result
        
        # Execute query
        logger.debug(f"Executing query: {kpi.name}")
        start_time = time.time()
        
        try:
            # Get executable SQL
            sql = kpi.get_sql(params.get('time_range_days'))
            
            # Execute with timeout
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(sql).execution_options(timeout=self.QUERY_TIMEOUT)
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            execution_time = time.time() - start_time
            logger.info(
                f"Query executed: {kpi.name} "
                f"({len(df)} rows, {execution_time:.2f}s)"
            )
            
            # Cache result
            if self.use_cache:
                ttl = self._determine_ttl(kpi, params)
                self._save_to_cache(kpi.id, params, df, ttl)
            
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Query failed: {kpi.name} - {e}")
            raise ExecutionError(f"Query execution failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error: {kpi.name} - {e}")
            raise ExecutionError(f"Unexpected error: {str(e)}")
    
    def execute_multiple(
        self,
        kpis: list[KPI],
        time_range_days: Optional[int] = None,
        bypass_cache: bool = False,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute multiple KPIs
        
        Args:
            kpis: List of KPIs to execute
            time_range_days: Override time range for all
            bypass_cache: Skip cache for all
            parallel: Execute in parallel (future enhancement)
            
        Returns:
            Dictionary mapping kpi_id to result DataFrame
        """
        results = {}
        
        for kpi in kpis:
            try:
                result = self.execute_kpi(
                    kpi,
                    time_range_days=time_range_days,
                    bypass_cache=bypass_cache
                )
                results[kpi.id] = result
                
            except Exception as e:
                logger.error(f"Failed to execute {kpi.name}: {e}")
                # Store empty DataFrame as placeholder
                results[kpi.id] = pd.DataFrame()
        
        return results
    
    def _get_from_cache(
        self,
        kpi_id: str,
        params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Get cached result"""
        if not self.cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(kpi_id, params)
            cached = self.cache.get(cache_key)
            
            if cached:
                # Deserialize DataFrame
                data = json.loads(cached)
                df = pd.read_json(data, orient='records')
                return df
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _save_to_cache(
        self,
        kpi_id: str,
        params: Dict[str, Any],
        result: pd.DataFrame,
        ttl: int
    ):
        """Save result to cache"""
        if not self.cache:
            return
        
        try:
            cache_key = self._generate_cache_key(kpi_id, params)
            
            # Serialize DataFrame
            json_data = result.to_json(orient='records')
            
            # Save with TTL
            self.cache.setex(cache_key, ttl, json_data)
            logger.debug(f"Cached result: {kpi_id} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _generate_cache_key(
        self,
        kpi_id: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key from KPI ID and parameters"""
        # Create stable parameter representation
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"kpi:{kpi_id}:{param_hash}"
    
    def _determine_ttl(
        self,
        kpi: KPI,
        params: Dict[str, Any]
    ) -> int:
        """
        Determine appropriate TTL for KPI
        
        Args:
            kpi: KPI definition
            params: Query parameters
            
        Returns:
            TTL in seconds
        """
        # If KPI has time parameter, base TTL on time range
        if kpi.has_time_parameter:
            days = params.get('time_range_days', 30)
            
            if days <= 7:
                return self.DEFAULT_TTL['realtime']
            elif days <= 30:
                return self.DEFAULT_TTL['hourly']
            elif days <= 365:
                return self.DEFAULT_TTL['daily']
            else:
                return self.DEFAULT_TTL['historical']
        
        # Default: historical TTL
        return self.DEFAULT_TTL['historical']
    
    def invalidate_cache(self, kpi_id: Optional[str] = None):
        """
        Invalidate cache
        
        Args:
            kpi_id: Specific KPI to invalidate (None = all)
        """
        if not self.cache:
            return
        
        try:
            if kpi_id:
                # Invalidate specific KPI (all parameter variations)
                pattern = f"kpi:{kpi_id}:*"
                keys = self.cache.keys(pattern)
                if keys:
                    self.cache.delete(*keys)
                    logger.info(f"Cache invalidated: {kpi_id} ({len(keys)} keys)")
            else:
                # Invalidate all KPI cache
                pattern = "kpi:*"
                keys = self.cache.keys(pattern)
                if keys:
                    self.cache.delete(*keys)
                    logger.info(f"All cache invalidated ({len(keys)} keys)")
                    
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.cache:
            return {'enabled': False}
        
        try:
            pattern = "kpi:*"
            keys = self.cache.keys(pattern)
            
            return {
                'enabled': True,
                'total_keys': len(keys),
                'memory_usage': self.cache.info('memory')['used_memory_human']
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def close(self):
        """Close connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
        
        if self.cache:
            self.cache.close()
            logger.info("Cache connection closed")


class ExecutionError(Exception):
    """Raised when query execution fails"""
    pass


class InMemoryCache:
    """
    Simple in-memory cache fallback (if Redis not available)
    """
    
    def __init__(self):
        self._cache: Dict[str, tuple[str, float]] = {}  # key -> (value, expiry)
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                # Expired
                del self._cache[key]
        return None
    
    def setex(self, key: str, ttl: int, value: str):
        """Set value with TTL"""
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
    
    def delete(self, *keys):
        """Delete keys"""
        for key in keys:
            if key in self._cache:
                del self._cache[key]
    
    def keys(self, pattern: str) -> list:
        """Get keys matching pattern (simple glob)"""
        import fnmatch
        return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
    
    def ping(self):
        """Test connection (always succeeds)"""
        return True
    
    def close(self):
        """Close connection"""
        self._cache.clear()


# Convenience function
def execute_kpi(
    kpi: KPI,
    database_uri: str,
    time_range_days: Optional[int] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Quick function to execute single KPI
    
    Args:
        kpi: KPI to execute
        database_uri: Database connection URI
        time_range_days: Optional time range override
        use_cache: Whether to use caching
        
    Returns:
        Query result DataFrame
    """
    executor = KPIExecutor(database_uri, use_cache=use_cache)
    try:
        return executor.execute_kpi(kpi, time_range_days)
    finally:
        executor.close()
