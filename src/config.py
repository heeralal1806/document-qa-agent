"""
Document Q&A AI Agent - Configuration Management
=================================================
Centralized configuration using Pydantic settings.

This module provides type-safe configuration management
with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be configured via environment variables.
    Environment variables take precedence over defaults.
    """
    
    # =============================================
    # API Configuration
    # =============================================
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    
    # =============================================
    # LLM Configuration
    # =============================================
    LLM_PROVIDER: str = Field(
        default="openai",
        description="LLM provider: 'openai' or 'gemini'"
    )
    MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name"
    )
    GEMINI_MODEL_NAME: str = Field(
        default="gemini-1.5-flash",
        description="Google Gemini model name"
    )
    TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0-2.0)"
    )
    MAX_TOKENS: int = Field(
        default=2000,
        ge=1,
        le=128000,
        description="Maximum tokens for response"
    )
    TOP_P: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    
    # =============================================
    # Application Settings
    # =============================================
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    WORKERS: int = Field(default=1, description="Number of workers")
    
    # =============================================
    # Document Processing
    # =============================================
    UPLOAD_DIR: str = Field(
        default="uploads",
        description="Directory for uploaded files"
    )
    MAX_FILE_SIZE: int = Field(
        default=52428800,
        description="Maximum file size in bytes (50MB)"
    )
    ALLOWED_EXTENSIONS: str = Field(
        default="pdf",
        description="Allowed file extensions"
    )
    MAX_PAGES_PER_DOCUMENT: int = Field(
        default=100,
        description="Maximum pages per document"
    )
    
    # =============================================
    # Caching Configuration
    # =============================================
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    CACHE_TYPE: str = Field(
        default="disk",
        description="Cache type: 'memory', 'disk', or 'redis'"
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    DISK_CACHE_DIR: str = Field(
        default="data/cache",
        description="Directory for disk cache"
    )
    
    # =============================================
    # Rate Limiting
    # =============================================
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Maximum requests per window"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # =============================================
    # Logging
    # =============================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FILE: str = Field(
        default="logs/app.log",
        description="Log file path"
    )
    
    # =============================================
    # Security
    # =============================================
    SECRET_KEY: str = Field(
        default="change-this-in-production",
        description="Secret key for security"
    )
    CORS_ORIGINS: str = Field(
        default="*",
        description="CORS origins (comma-separated)"
    )
    
    # =============================================
    # ArXiv Configuration
    # =============================================
    ARXIV_MAX_RESULTS: int = Field(
        default=10,
        description="Maximum ArXiv results"
    )
    ARXIV_SORT_BY: str = Field(
        default="relevance",
        description="ArXiv sort order"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = ["openai", "gemini"]
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {v}. "
                f"Must be one of: {valid_providers}"
            )
        return v.lower()
    
    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def validate_extensions(cls, v: str) -> str:
        """Validate and parse allowed extensions."""
        ext_list = [e.strip().lower() for e in v.split(",")]
        return ",".join(ext_list)
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get list of allowed extensions."""
        return [e.strip() for e in self.ALLOWED_EXTENSIONS.split(",")]
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get list of CORS origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]
    
    def get_active_model(self) -> str:
        """Get the currently active LLM model name."""
        if self.LLM_PROVIDER == "openai":
            return self.MODEL_NAME
        elif self.LLM_PROVIDER == "gemini":
            return self.GEMINI_MODEL_NAME
        else:
            return self.MODEL_NAME


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function uses lru_cache to ensure settings
    are only loaded once and cached for subsequent calls.
    """
    return Settings()


# Global settings instance
settings = get_settings()

