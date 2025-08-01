"""
Database models for ML-powered web scraper
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class ScrapingJob(Base):
    """Model for tracking scraping jobs"""
    
    __tablename__ = "scraping_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed, cancelled
    total_urls = Column(Integer, nullable=False, default=0)
    processed_urls = Column(Integer, nullable=False, default=0)
    failed_urls = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Configuration and metadata
    engine_type = Column(String(50), default="auto")
    ml_analysis_enabled = Column(Boolean, default=True)
    ollama_model = Column(String(100))
    
    # Results and errors
    errors = Column(Text)  # JSON string of errors
    job_metadata = Column(JSON)  # Additional job metadata
    
    # Relationships
    scraped_content = relationship("ScrapedContent", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ScrapingJob(job_id='{self.job_id}', status='{self.status}')>"


class ScrapedContent(Base):
    """Model for storing scraped content and analysis results"""
    
    __tablename__ = "scraped_content"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), ForeignKey("scraping_jobs.job_id"), nullable=False)
    
    # Content information
    url = Column(Text, nullable=False)
    title = Column(Text)
    content = Column(Text)
    word_count = Column(Integer, default=0)
    
    # Scraping metadata
    scraping_engine = Column(String(50))
    scraped_at = Column(DateTime, default=func.now())
    response_time = Column(Float)  # Response time in seconds
    status_code = Column(Integer)
    
    # ML Analysis results (stored as JSON)
    nlp_analysis = Column(JSON)
    classification = Column(JSON)
    ollama_analysis = Column(JSON)
    quality_score = Column(Float)
    
    # Additional metadata
    content_metadata = Column(JSON)  # Links, images, headers, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    job = relationship("ScrapingJob", back_populates="scraped_content")
    
    def __repr__(self):
        return f"<ScrapedContent(url='{self.url[:50]}...', job_id='{self.job_id}')>"


class MLModel(Base):
    """Model for tracking ML model versions and performance"""
    
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # classifier, quality_scorer, duplicate_detector
    version = Column(String(50), nullable=False)
    
    # Model metadata
    model_path = Column(String(255))
    model_size = Column(Integer)  # Size in bytes
    parameters_count = Column(Integer)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training information
    training_data_size = Column(Integer)
    training_duration = Column(Float)  # Duration in seconds
    hyperparameters = Column(JSON)
    
    # Status and timestamps
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    trained_at = Column(DateTime)
    
    def __repr__(self):
        return f"<MLModel(name='{self.model_name}', version='{self.version}')>"


class ContentCategory(Base):
    """Model for content categories and their definitions"""
    
    __tablename__ = "content_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    keywords = Column(JSON)  # List of keywords associated with this category
    
    # Statistics
    content_count = Column(Integer, default=0)
    avg_quality_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ContentCategory(name='{self.name}')>"


class UserSession(Base):
    """Model for tracking user sessions and preferences"""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # User preferences
    preferred_engine = Column(String(50), default="auto")
    preferred_ollama_model = Column(String(100))
    ml_analysis_enabled = Column(Boolean, default=True)
    
    # Session metadata
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    
    # Activity tracking
    jobs_count = Column(Integer, default=0)
    urls_scraped = Column(Integer, default=0)
    last_activity = Column(DateTime, default=func.now())
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}')>"


class ExportLog(Base):
    """Model for tracking data exports"""
    
    __tablename__ = "export_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    export_id = Column(String(255), unique=True, index=True, nullable=False)
    job_id = Column(String(255), ForeignKey("scraping_jobs.job_id"))
    
    # Export details
    export_type = Column(String(50), nullable=False)  # csv, json, pdf
    file_path = Column(String(500))
    s3_url = Column(String(500))
    file_size = Column(Integer)  # Size in bytes
    
    # Export configuration
    filters = Column(JSON)  # Export filters applied
    columns = Column(JSON)  # Columns included in export
    
    # Status and timestamps
    status = Column(String(50), default="pending")  # pending, completed, failed
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    # Error handling
    error_message = Column(Text)
    
    def __repr__(self):
        return f"<ExportLog(export_id='{self.export_id}', type='{self.export_type}')>"


class SystemMetrics(Base):
    """Model for storing system performance metrics"""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    
    # Context information
    context = Column(JSON)  # Additional context data
    
    # Timestamp
    recorded_at = Column(DateTime, default=func.now(), index=True)
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"


class ContentDuplicate(Base):
    """Model for tracking duplicate content detection results"""
    
    __tablename__ = "content_duplicates"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Content references
    content1_id = Column(Integer, ForeignKey("scraped_content.id"), nullable=False)
    content2_id = Column(Integer, ForeignKey("scraped_content.id"), nullable=False)
    
    # Similarity metrics
    neural_similarity = Column(Float, nullable=False)
    cosine_similarity = Column(Float)
    jaccard_similarity = Column(Float)
    
    # Detection result
    is_duplicate = Column(Boolean, nullable=False)
    confidence = Column(Float)
    threshold_used = Column(Float)
    
    # Timestamps
    detected_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<ContentDuplicate(content1_id={self.content1_id}, content2_id={self.content2_id}, is_duplicate={self.is_duplicate})>"


# Database utility functions
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


def get_model_by_name(model_name: str):
    """Get SQLAlchemy model by name"""
    models = {
        'scraping_jobs': ScrapingJob,
        'scraped_content': ScrapedContent,
        'ml_models': MLModel,
        'content_categories': ContentCategory,
        'user_sessions': UserSession,
        'export_logs': ExportLog,
        'system_metrics': SystemMetrics,
        'content_duplicates': ContentDuplicate
    }
    return models.get(model_name.lower())


# Model relationships and constraints
def setup_indexes(engine):
    """Create additional indexes for performance"""
    from sqlalchemy import Index
    
    # Create custom indexes
    indexes = [
        Index('idx_scraped_content_url_hash', ScrapedContent.url),
        Index('idx_scraped_content_job_created', ScrapedContent.job_id, ScrapedContent.created_at),
        Index('idx_scraping_jobs_status_created', ScrapingJob.status, ScrapingJob.created_at),
        Index('idx_system_metrics_name_time', SystemMetrics.metric_name, SystemMetrics.recorded_at),
        Index('idx_export_logs_status_created', ExportLog.status, ExportLog.created_at)
    ]
    
    # Create indexes
    for index in indexes:
        try:
            index.create(engine, checkfirst=True)
        except Exception as e:
            print(f"Warning: Could not create index {index.name}: {e}")


# Default data initialization
def initialize_default_data(session):
    """Initialize default data in the database"""
    
    # Default content categories
    default_categories = [
        {"name": "technology", "description": "Technology, software, and IT related content", 
         "keywords": ["software", "hardware", "programming", "computer", "tech", "ai", "ml"]},
        {"name": "business", "description": "Business, finance, and corporate content",
         "keywords": ["business", "finance", "market", "company", "revenue", "profit"]},
        {"name": "science", "description": "Scientific research and discoveries",
         "keywords": ["research", "study", "experiment", "scientific", "discovery", "analysis"]},
        {"name": "health", "description": "Health, medical, and wellness content",
         "keywords": ["health", "medical", "doctor", "treatment", "wellness", "medicine"]},
        {"name": "education", "description": "Educational and academic content",
         "keywords": ["education", "school", "university", "learning", "academic", "student"]},
        {"name": "entertainment", "description": "Entertainment and media content",
         "keywords": ["entertainment", "movie", "music", "celebrity", "show", "game"]},
        {"name": "sports", "description": "Sports and athletic content",
         "keywords": ["sport", "team", "player", "game", "tournament", "athletic"]},
        {"name": "politics", "description": "Political and government related content",
         "keywords": ["politics", "government", "election", "policy", "democracy", "vote"]},
        {"name": "travel", "description": "Travel and tourism content",
         "keywords": ["travel", "vacation", "tourism", "destination", "trip", "hotel"]},
        {"name": "general", "description": "General content that doesn't fit other categories",
         "keywords": ["general", "misc", "other", "various", "mixed"]}
    ]
    
    # Add categories if they don't exist
    for cat_data in default_categories:
        existing = session.query(ContentCategory).filter_by(name=cat_data["name"]).first()
        if not existing:
            category = ContentCategory(**cat_data)
            session.add(category)
    
    try:
        session.commit()
        print("Default data initialized successfully")
    except Exception as e:
        session.rollback()
        print(f"Warning: Could not initialize default data: {e}")