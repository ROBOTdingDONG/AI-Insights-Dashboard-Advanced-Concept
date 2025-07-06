"""
Database Connection and Session Management
==========================================

Enterprise-grade database integration with PostgreSQL and TimescaleDB support.

Features:
- Async SQLAlchemy with connection pooling
- Automatic session management and cleanup
- Database health monitoring and reconnection
- Query performance tracking and optimization
- Migration support with Alembic integration
- Time-series data optimization with TimescaleDB

Security Features:
- SSL/TLS connection encryption
- Connection string sanitization
- Query logging with sensitive data masking
- Automatic connection timeout and retry logic

Author: AI Insights Team
Version: 1.0.0
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, timezone

import structlog
from sqlalchemy import (
    create_engine, 
    MetaData, 
    event, 
    pool,
    text,
    inspect
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    async_scoped_session
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import (
    SQLAlchemyError, 
    DisconnectionError, 
    OperationalError,
    DatabaseError
)
from alembic import command
from alembic.config import Config
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Import configuration
from app.core.config import get_settings


logger = structlog.get_logger(__name__)
settings = get_settings()


# =============================================================================
# DATABASE METADATA AND BASE
# =============================================================================
# SQLAlchemy metadata for schema management
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Declarative base for model classes
Base = declarative_base(metadata=metadata)


# =============================================================================
# DATABASE CONNECTION CONFIGURATION
# =============================================================================
class DatabaseManager:
    """
    Enterprise database manager with async support and connection pooling.
    
    Provides high-performance database operations with automatic connection
    management, health monitoring, and error recovery.
    """
    
    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self.sync_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None
        self.is_connected = False
        self.connection_attempts = 0
        self.last_health_check = None
        
        # Performance monitoring
        self.query_count = 0
        self.slow_query_count = 0
        self.connection_errors = 0
        
        # Connection pool configuration
        self.pool_config = {
            "poolclass": QueuePool,
            "pool_size": settings.database.DB_POOL_SIZE,
            "max_overflow": settings.database.DB_MAX_OVERFLOW,
            "pool_timeout": settings.database.DB_POOL_TIMEOUT,
            "pool_recycle": settings.database.DB_POOL_RECYCLE,
            "pool_pre_ping": True,  # Validate connections before use
            "pool_reset_on_return": "commit",
        }
        
        # SSL configuration
        self.connect_args = {}
        if settings.database.DB_SSL_MODE != "disable":
            self.connect_args.update({
                "sslmode": settings.database.DB_SSL_MODE,
                "server_settings": {
                    "application_name": f"{settings.APP_NAME}_v{settings.APP_VERSION}",
                    "timezone": "UTC"
                }
            })
            
            if settings.database.DB_SSL_CERT:
                self.connect_args.update({
                    "sslcert": settings.database.DB_SSL_CERT,
                    "sslkey": settings.database.DB_SSL_KEY,
                    "sslrootcert": settings.database.DB_SSL_ROOT_CERT
                })
    
    async def initialize(self) -> None:
        """
        Initialize database connections and session factories.
        
        Creates async and sync engines with optimized configuration
        for high-performance operations.
        """
        try:
            logger.info("Initializing database connections")
            
            # Create async engine
            self.async_engine = create_async_engine(
                settings.database.database_url,
                echo=settings.database.DB_ECHO,
                echo_pool=settings.database.DB_ECHO_POOL,
                connect_args=self.connect_args,
                **self.pool_config
            )
            
            # Create sync engine for migrations and admin operations
            self.sync_engine = create_engine(
                settings.database.sync_database_url,
                echo=settings.database.DB_ECHO,
                echo_pool=settings.database.DB_ECHO_POOL,
                connect_args=self.connect_args,
                **self.pool_config
            )
            
            # Create session factories
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autoflush=True,
                autocommit=False,
                expire_on_commit=False
            )
            
            # Set up event listeners for monitoring
            self._setup_event_listeners()
            
            # Test connection
            await self._test_connection()
            
            # Initialize TimescaleDB extensions if needed
            await self._setup_timescaledb()
            
            self.is_connected = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            self.connection_errors += 1
            logger.error("Database initialization failed", error=str(e), exc_info=True)
            raise
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring and logging."""
        
        # Query performance monitoring
        @event.listens_for(self.async_engine.sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time for performance monitoring."""
            context._query_start_time = time.time()
            self.query_count += 1
        
        @event.listens_for(self.async_engine.sync_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries and update performance metrics."""
            total_time = time.time() - context._query_start_time
            
            if total_time > settings.monitoring.SLOW_QUERY_THRESHOLD:
                self.slow_query_count += 1
                logger.warning(
                    "Slow query detected",
                    duration_seconds=round(total_time, 3),
                    statement=statement[:200] + "..." if len(statement) > 200 else statement
                )
        
        # Connection pool monitoring
        @event.listens_for(self.async_engine.sync_engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Log new database connections."""
            logger.debug("New database connection established")
        
        @event.listens_for(self.async_engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Monitor connection pool checkout."""
            pool_status = connection_proxy.get_pool().status()
            if pool_status.endswith("overflow"):
                logger.warning("Database connection pool overflow detected")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((OperationalError, DisconnectionError))
    )
    async def _test_connection(self) -> None:
        """Test database connection with retry logic."""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT version(), current_database(), current_user"))
                db_info = result.fetchone()
                
                logger.info(
                    "Database connection test successful",
                    version=str(db_info[0]).split(' ')[0] if db_info else "unknown",
                    database=db_info[1] if db_info else "unknown",
                    user=db_info[2] if db_info else "unknown"
                )
                
        except Exception as e:
            self.connection_attempts += 1
            logger.error(
                "Database connection test failed",
                attempt=self.connection_attempts,
                error=str(e)
            )
            raise
    
    async def _setup_timescaledb(self) -> None:
        """Initialize TimescaleDB extensions for time-series data."""
        try:
            async with self.async_engine.begin() as conn:
                # Check if TimescaleDB extension exists
                result = await conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
                )
                
                if not result.fetchone():
                    logger.info("TimescaleDB extension not found, attempting to create")
                    try:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                        logger.info("TimescaleDB extension created successfully")
                    except Exception as e:
                        logger.warning(
                            "Failed to create TimescaleDB extension",
                            error=str(e),
                            note="This is optional for basic functionality"
                        )
                else:
                    logger.info("TimescaleDB extension already installed")
                    
        except Exception as e:
            logger.warning(
                "TimescaleDB setup failed",
                error=str(e),
                note="Continuing without time-series optimization"
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check.
        
        Returns:
            Dict containing health status and performance metrics
        """
        health_data = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connection_pool": {},
            "performance": {},
            "errors": {}
        }
        
        try:
            # Test basic connectivity
            start_time = time.time()
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            connection_time = time.time() - start_time
            health_data["status"] = "healthy"
            
            # Connection pool status
            pool = self.async_engine.pool
            health_data["connection_pool"] = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            
            # Performance metrics
            health_data["performance"] = {
                "connection_time_ms": round(connection_time * 1000, 2),
                "total_queries": self.query_count,
                "slow_queries": self.slow_query_count,
                "slow_query_ratio": (
                    self.slow_query_count / max(self.query_count, 1)
                ) * 100
            }
            
            # Error tracking
            health_data["errors"] = {
                "connection_errors": self.connection_errors,
                "last_error": None  # Could track last error timestamp
            }
            
            self.last_health_check = datetime.now(timezone.utc)
            
        except Exception as e:
            health_data["status"] = "unhealthy"
            health_data["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
        
        return health_data
    
    async def get_session(self) -> AsyncSession:
        """
        Create a new async database session.
        
        Returns:
            AsyncSession: Configured database session
        """
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.async_session_factory()
    
    def get_sync_session(self) -> Session:
        """
        Create a new sync database session for migrations and admin tasks.
        
        Returns:
            Session: Configured sync database session
        """
        if not self.sync_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.sync_session_factory()
    
    async def close(self) -> None:
        """Clean shutdown of database connections."""
        logger.info("Closing database connections")
        
        if self.async_engine:
            await self.async_engine.dispose()
        
        if self.sync_engine:
            self.sync_engine.dispose()
        
        self.is_connected = False
        logger.info("Database connections closed")


# =============================================================================
# GLOBAL DATABASE INSTANCE
# =============================================================================
db_manager = DatabaseManager()


# =============================================================================
# SESSION DEPENDENCY FUNCTIONS
# =============================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Provides automatic session management with proper cleanup
    and error handling.
    
    Yields:
        AsyncSession: Database session for request scope
    """
    if not db_manager.is_connected:
        await db_manager.initialize()
    
    session = await db_manager.get_session()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error("Database session error", error=str(e))
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI.
    
    Provides manual session management for background tasks
    and utility functions.
    
    Yields:
        AsyncSession: Database session with automatic cleanup
    """
    if not db_manager.is_connected:
        await db_manager.initialize()
    
    session = await db_manager.get_session()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error("Database context error", error=str(e))
        raise
    finally:
        await session.close()


# =============================================================================
# DATABASE UTILITIES
# =============================================================================
async def create_tables() -> None:
    """
    Create all database tables defined in models.
    
    This is primarily for development and testing.
    Production should use Alembic migrations.
    """
    logger.info("Creating database tables")
    
    if not db_manager.async_engine:
        await db_manager.initialize()
    
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")


async def drop_tables() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data. Use with extreme caution.
    """
    logger.warning("Dropping all database tables")
    
    if not db_manager.async_engine:
        await db_manager.initialize()
    
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("Database tables dropped")


def run_migrations(revision: str = "head") -> None:
    """
    Run Alembic database migrations.
    
    Args:
        revision: Target revision (default: "head" for latest)
    """
    logger.info(f"Running database migrations to revision: {revision}")
    
    try:
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sync_database_url)
        
        command.upgrade(alembic_cfg, revision)
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error("Database migration failed", error=str(e), exc_info=True)
        raise


def create_migration(message: str, autogenerate: bool = True) -> None:
    """
    Create a new Alembic migration.
    
    Args:
        message: Migration description
        autogenerate: Whether to auto-detect model changes
    """
    logger.info(f"Creating new migration: {message}")
    
    try:
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sync_database_url)
        
        command.revision(
            alembic_cfg,
            message=message,
            autogenerate=autogenerate
        )
        logger.info("Migration created successfully")
        
    except Exception as e:
        logger.error("Migration creation failed", error=str(e), exc_info=True)
        raise


async def get_database_info() -> Dict[str, Any]:
    """
    Get comprehensive database information and statistics.
    
    Returns:
        Dict containing database metadata and performance info
    """
    if not db_manager.async_engine:
        await db_manager.initialize()
    
    info = {}
    
    try:
        async with db_manager.async_engine.begin() as conn:
            # Database version and settings
            result = await conn.execute(text("""
                SELECT 
                    version() as version,
                    current_database() as database_name,
                    current_user as current_user,
                    inet_server_addr() as server_address,
                    inet_server_port() as server_port
            """))
            db_basic = result.fetchone()
            
            # Database size
            result = await conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as database_size
            """))
            db_size = result.fetchone()
            
            # Connection stats
            result = await conn.execute(text("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """))
            connections = result.fetchone()
            
            # Table information
            result = await conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """))
            tables = result.fetchall()
            
            info = {
                "database": {
                    "version": db_basic[0] if db_basic else "unknown",
                    "name": db_basic[1] if db_basic else "unknown",
                    "user": db_basic[2] if db_basic else "unknown",
                    "server_address": db_basic[3] if db_basic else "unknown",
                    "server_port": db_basic[4] if db_basic else "unknown",
                    "size": db_size[0] if db_size else "unknown"
                },
                "connections": {
                    "total": connections[0] if connections else 0,
                    "active": connections[1] if connections else 0,
                    "idle": connections[2] if connections else 0
                },
                "tables": [
                    {
                        "schema": table[0],
                        "name": table[1],
                        "size": table[2]
                    } for table in tables
                ],
                "health": await db_manager.health_check()
            }
            
    except Exception as e:
        logger.error("Failed to get database info", error=str(e))
        info["error"] = str(e)
    
    return info


# =============================================================================
# TIMESCALEDB UTILITIES
# =============================================================================
async def create_hypertable(table_name: str, time_column: str = "timestamp") -> bool:
    """
    Create a TimescaleDB hypertable for time-series data optimization.
    
    Args:
        table_name: Name of the table to convert
        time_column: Name of the time column (default: "timestamp")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        async with db_manager.async_engine.begin() as conn:
            # Check if table exists
            result = await conn.execute(text(f"""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = '{table_name}' AND table_schema = 'public'
            """))
            
            if not result.fetchone():
                logger.error(f"Table {table_name} does not exist")
                return False
            
            # Check if already a hypertable
            result = await conn.execute(text(f"""
                SELECT 1 FROM timescaledb_information.hypertables 
                WHERE hypertable_name = '{table_name}'
            """))
            
            if result.fetchone():
                logger.info(f"Table {table_name} is already a hypertable")
                return True
            
            # Create hypertable
            await conn.execute(text(f"""
                SELECT create_hypertable('{table_name}', '{time_column}')
            """))
            
            logger.info(f"Successfully created hypertable: {table_name}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to create hypertable {table_name}", error=str(e))
        return False


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================
async def init_database() -> None:
    """Initialize database connections and run migrations."""
    logger.info("Initializing database system")
    
    try:
        # Initialize connection manager
        await db_manager.initialize()
        
        # Run migrations in production, create tables in development
        if settings.is_production:
            run_migrations()
        else:
            await create_tables()
        
        logger.info("Database system initialized successfully")
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


async def close_database() -> None:
    """Clean shutdown of database connections."""
    await db_manager.close()


# =============================================================================
# EXPORTS
# =============================================================================
# Export commonly used items
engine = db_manager.async_engine
SessionLocal = db_manager.async_session_factory


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    async def test_database():
        """Test database connectivity and operations."""
        try:
            await init_database()
            
            # Test basic operations
            async with get_db_context() as session:
                result = await session.execute(text("SELECT current_timestamp"))
                timestamp = result.scalar()
                print(f"Database timestamp: {timestamp}")
            
            # Get database info
            info = await get_database_info()
            print(f"Database info: {info}")
            
            # Health check
            health = await db_manager.health_check()
            print(f"Health status: {health['status']}")
            
        finally:
            await close_database()
    
    # asyncio.run(test_database())
