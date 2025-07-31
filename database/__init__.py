
# database/__init__.py
"""
Database models and connection management
"""

from .models import (
    ScrapingJob, ScrapedContent, MLModel, ContentCategory,
    UserSession, ExportLog, SystemMetrics, ContentDuplicate
)
from .connection import (
    init_database, get_db_session, get_sync_db_session,
    close_database, database_transaction, health_monitor
)

__all__ = [
    'ScrapingJob',
    'ScrapedContent', 
    'MLModel',
    'ContentCategory',
    'UserSession',
    'ExportLog',
    'SystemMetrics',
    'ContentDuplicate',
    'init_database',
    'get_db_session',
    'get_sync_db_session',
    'close_database',
    'database_transaction',
    'health_monitor'
]
