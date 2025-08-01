"""
Configuration settings for ML-Powered Web Scraper
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database Configuration
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(5432, env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field("us-east-1", env="AWS_REGION")
    S3_BUCKET_NAME: str = Field(..., env="S3_BUCKET_NAME")
    
    # Ollama Configuration
    OLLAMA_HOST: str = Field("localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, env="OLLAMA_PORT")
    DEFAULT_MODEL: str = Field("llama3.1:8b", env="DEFAULT_MODEL")
    BACKUP_MODEL: str = Field("llama2:latest", env="BACKUP_MODEL")
    
    # Application Settings
    APP_HOST: str = Field("localhost", env="APP_HOST")
    APP_PORT: int = Field(7860, env="APP_PORT")
    DEBUG: bool = Field(True, env="DEBUG")
    MAX_CONCURRENT_SCRAPES: int = Field(5, env="MAX_CONCURRENT_SCRAPES")
    MAX_URLS_PER_BATCH: int = Field(100, env="MAX_URLS_PER_BATCH")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Scraping Settings
    REQUEST_TIMEOUT: int = 30
    REQUEST_DELAY: float = 1.0
    MAX_RETRIES: int = 3
    USER_AGENT: str = "ML-Web-Scraper/1.0"
    
    # ML Settings
    MAX_TEXT_LENGTH: int = 10000
    MIN_TEXT_LENGTH: int = 10
    CONFIDENCE_THRESHOLD: float = 0.7
    BATCH_SIZE: int = 32
    
    # File Settings
    UPLOAD_DIR: str = "uploads"
    EXPORT_DIR: str = "exports"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    @property
    def DATABASE_URL(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def OLLAMA_URL(self) -> str:
        """Get Ollama API URL"""
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"
    
    @property
    def AVAILABLE_MODELS(self) -> List[str]:
        """Get list of available Ollama models"""
        return [self.DEFAULT_MODEL, self.BACKUP_MODEL]
    
    def get_upload_path(self, filename: str) -> Path:
        """Get upload file path"""
        upload_dir = Path(self.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        return upload_dir / filename
    
    def get_export_path(self, filename: str) -> Path:
        """Get export file path"""
        export_dir = Path(self.EXPORT_DIR)
        export_dir.mkdir(exist_ok=True)
        return export_dir / filename
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
Path(settings.EXPORT_DIR).mkdir(exist_ok=True)