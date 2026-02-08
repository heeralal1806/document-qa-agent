"""
Document Q&A AI Agent - Utilities
=================================
Helper functions and utilities.
"""

import os
import re
import logging
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import mimetypes

from src.config import settings

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "logs/app.log", 
                  log_level: str = "INFO") -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create application logger
    logger = logging.getLogger("document_qa_agent")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = Path(filename).name
    
    # Replace problematic characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:90] + ext
    
    return filename


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:6]
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_suffix}"
    return f"{timestamp}_{random_suffix}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up files older than specified age.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files deleted
    """
    count = 0
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for file_path in dir_path.glob("*"):
        try:
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime < cutoff_time:
                    file_path.unlink()
                    count += 1
        except Exception as e:
            logger.warning(f"Error cleaning up file {file_path}: {e}")
    
    logger.info(f"Cleaned up {count} old files from {directory}")
    return count


def validate_file_type(file_path: str) -> bool:
    """
    Validate that a file is an allowed type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file type is allowed
    """
    allowed_extensions = settings.allowed_extensions_list
    
    # Check by extension
    ext = Path(file_path).suffix.lower()
    if ext.replace('.', '') not in [e.replace('.', '') for e in allowed_extensions]:
        return False
    
    # Check by MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        allowed_mimes = {
            'application/pdf',
            'application/x-pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        if mime_type not in allowed_mimes:
            return False
    
    return True


def truncate_text(text: str, max_length: int = 1000, 
                  suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to analyze
        num_keywords: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stopwords
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
        'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
        'this', 'that', 'with', 'from', 'they', 'will', 'would', 'there',
        'their', 'what', 'about', 'which', 'when', 'make', 'just', 'over',
        'such', 'into', 'than', 'them', 'some', 'could', 'other', 'more'
    }
    
    words = [w for w in words if w not in stopwords]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(words)
    
    # Get top keywords
    keywords = [word for word, count in word_counts.most_common(num_keywords)]
    
    return keywords


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for text.
    
    Args:
        text: Text to analyze
        words_per_minute: Reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, word_count // words_per_minute)


def clean_json_string(json_str: str) -> str:
    """
    Clean and normalize a JSON string.
    
    Args:
        json_str: JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    # Remove comments
    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Remove trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    return json_str


def retry_on_failure(max_retries: int = 3, delay: int = 1):
    """
    Decorator for retrying failed functions.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries (seconds)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    import time
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.3f} seconds")
        return False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@lru_cache(maxsize=100)
def get_cached_keywords(text_hash: str, text: str) -> tuple:
    """
    Get cached keywords for text.
    
    Args:
        text_hash: Hash of the text
        text: Original text
        
    Returns:
        Tuple of (keywords, reading_time)
    """
    keywords = extract_keywords(text)
    reading_time = estimate_reading_time(text)
    return tuple(keywords), reading_time


def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def safe_get(data: Dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to access
        keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at path or default
    """
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def format_timestamp(timestamp: Optional[float] = None,
                    fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp.
    
    Args:
        timestamp: Unix timestamp (uses current time if None)
        fmt: Format string
        
    Returns:
        Formatted timestamp
    """
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    
    return dt.strftime(fmt)

