"""
Database connection and session management
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from config.settings import settings
from database.models import Base, create_tables, setup_indexes, initialize_default_data

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session manager"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return
        
        try:
            # Create database if it doesn't exist
            await self._ensure_database_exists()
            
            # Create synchronous engine
            self.engine = create_engine(
                settings.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG
            )
            
            # Create asynchronous engine
            async_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
            self.async_engine = create_async_engine(
                async_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG
            )
            
            # Create session factories
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False
            )
            
            # Create tables and initialize data
            await self._setup_database()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _ensure_database_exists(self):
        """Ensure the database exists, create if it doesn't"""
        try:
            # Extract database name from URL
            db_name = settings.DB_NAME
            
            # Check if database exists
            conn = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                database="postgres"
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database: {db_name}")
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database {db_name} created successfully")
            else:
                logger.info(f"Database {db_name} already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to ensure database exists: {e}")
            # Don't raise here - database might already exist and be accessible
    
    async def _setup_database(self):
        """Setup database tables and initial data"""
        try:
            # Create all tables
            create_tables(self.engine)
            logger.info("Database tables created successfully")
            
            # Create additional indexes
            setup_indexes(self.engine)
            logger.info("Database indexes created successfully")
            
            # Initialize default data
            with self.SessionLocal() as session:
                initialize_default_data(session)
            
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session with proper cleanup"""
        if not self._initialized:
            await self.initialize()
        
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_session(self):
        """Get synchronous database session"""
        if not self._initialized:
            # For sync operations, we need to initialize synchronously
            self._sync_initialize()
        
        return self.SessionLocal()
    
    def _sync_initialize(self):
        """Synchronous initialization (simplified)"""
        if self._initialized:
            return
        
        try:
            self.engine = create_engine(
                settings.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            create_tables(self.engine)
            setup_indexes(self.engine)
            
            # Initialize default data
            with self.SessionLocal() as session:
                initialize_default_data(session)
            
            self._initialized = True
            logger.info("Database synchronized initialization completed")
            
        except Exception as e:
            logger.error(f"Sync database initialization failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.get_async_session() as session:
                await session.execute("SELECT 1")
                return True
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        
        self._initialized = False
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
async def init_database():
    """Initialize database"""
    await db_manager.initialize()


async def get_db_session():
    """Get async database session"""
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_db_session():
    """Get synchronous database session"""
    return db_manager.get_session()


async def close_database():
    """Close database connections"""
    await db_manager.close()


# Transaction management utilities
@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions"""
    async with db_manager.get_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise


class DatabaseHealthMonitor:
    """Monitor database health and performance"""
    
    def __init__(self):
        self.connection_errors = 0
        self.last_error_time = None
        self.max_errors = 5
        self.error_reset_time = 300  # 5 minutes
    
    async def check_health(self) -> dict:
        """Comprehensive database health check"""
        health_status = {
            'status': 'unknown',
            'details': {},
            'timestamp': asyncio.get_event_loop().time()
        }
        
        try:
            # Basic connectivity test
            is_connected = await db_manager.health_check()
            health_status['details']['connectivity'] = is_connected
            
            if is_connected:
                # Connection pool status
                if db_manager.engine:
                    pool = db_manager.engine.pool
                    health_status['details']['pool'] = {
                        'size': pool.size(),
                        'checked_in': pool.checkedin(),
                        'checked_out': pool.checkedout(),
                        'overflow': pool.overflow(),
                        'invalid': pool.invalid()
                    }
                
                # Performance test (simple query timing)
                start_time = asyncio.get_event_loop().time()
                async with db_manager.get_async_session() as session:
                    await session.execute("SELECT COUNT(*) FROM scraping_jobs")
                end_time = asyncio.get_event_loop().time()
                
                health_status['details']['query_time'] = end_time - start_time
                health_status['status'] = 'healthy'
                
                # Reset error counter on successful check
                self.connection_errors = 0
                
            else:
                health_status['status'] = 'unhealthy'
                health_status['details']['error'] = 'Connection failed'
                self._record_error()
        
        except Exception as e:
            health_status['status'] = 'error'
            health_status['details']['error'] = str(e)
            self._record_error()
        
        return health_status
    
    def _record_error(self):
        """Record database error"""
        self.connection_errors += 1
        self.last_error_time = asyncio.get_event_loop().time()
        
        if self.connection_errors >= self.max_errors:
            logger.critical(f"Database has {self.connection_errors} consecutive errors")
    
    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should be triggered"""
        if self.connection_errors >= self.max_errors:
            if self.last_error_time:
                time_since_error = asyncio.get_event_loop().time() - self.last_error_time
                return time_since_error < self.error_reset_time
        return False


# Global health monitor
health_monitor = DatabaseHealthMonitor()


# Database utility functions
async def execute_raw_query(query: str, params: dict = None) -> list:
    """Execute raw SQL query"""
    async with db_manager.get_async_session() as session:
        result = await session.execute(query, params or {})
        return result.fetchall()


async def get_table_stats() -> dict:
    """Get database table statistics"""
    stats = {}
    
    try:
        async with db_manager.get_async_session() as session:
            # Get row counts for main tables
            tables = ['scraping_jobs', 'scraped_content', 'ml_models', 'content_categories']
            
            for table in tables:
                result = await session.execute(f"SELECT COUNT(*) FROM {table}")
                count = result.scalar()
                stats[table] = {'row_count': count}
            
            # Get database size
            result = await session.execute(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            stats['database_size'] = result.scalar()
            
    except Exception as e:
        logger.error(f"Failed to get table stats: {e}")
        stats['error'] = str(e)
    
    return stats


async def cleanup_old_data(days: int = 30):
    """Clean up old data from database"""
    try:
        async with database_transaction() as session:
            # Delete old completed jobs and their content
            from database.models import ScrapingJob
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Delete old jobs (cascades to scraped_content)
            result = await session.execute(
                "DELETE FROM scraping_jobs WHERE status = 'completed' AND end_time < :cutoff",
                {'cutoff': cutoff_date}
            )
            
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old records")
            
            return deleted_count
            
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        return 0