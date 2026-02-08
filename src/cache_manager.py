"""
Document Q&A AI Agent - Cache Manager
=====================================
Response caching for improved performance and reduced API calls.

Supports:
- In-memory caching
- Disk-based caching
- Redis caching (optional)
"""

import logging
import hashlib
import json
import os
import time
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod

from src.config import settings

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (seconds)."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        pass


class MemoryCache(BaseCache):
    """
    Simple in-memory cache with TTL support.
    
    Good for single-instance deployments.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if "expires_at" not in entry:
            return True
        return datetime.now() > entry["expires_at"]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                logger.debug(f"Cache hit: {key[:30]}...")
                return entry["value"]
            else:
                # Clean up expired entry
                del self._cache[key]
        logger.debug(f"Cache miss: {key[:30]}...")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            # Clean up old entries if at max capacity
            if len(self._cache) >= self.max_size:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].get("created_at", datetime.now())
                )[:10]
                for k in oldest_keys:
                    del self._cache[k]
            
            self._cache[key] = {
                "value": value,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cached values."""
        self._cache.clear()
        return True
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        keys_to_delete = [k for k in self._cache if pattern in k]
        for k in keys_to_delete:
            del self._cache[k]
        return len(keys_to_delete)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self.max_size
        }


class DiskCache(BaseCache):
    """
    Disk-based cache for persistence across restarts.
    
    Good for production deployments and caching large responses.
    """
    
    def __init__(self, cache_dir: str = "data/cache", max_size_mb: int = 100):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cleanup_old()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        # Create a safe filename from the key
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if "expires_at" not in metadata:
            return True
        return datetime.fromisoformat(metadata["expires_at"]) < datetime.now()
    
    def _cleanup_old(self):
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now()
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if "expires_at" in metadata:
                        if datetime.fromisoformat(metadata["expires_at"]) < current_time:
                            cache_file.unlink()
                except Exception:
                    # Corrupted file, remove it
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def _get_cache_size(self) -> int:
        """Get total size of cache in bytes."""
        total = 0
        for f in self.cache_dir.glob("*.cache"):
            total += f.stat().st_size
        return total
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while self._get_cache_size() > self.max_size_bytes:
            # Find oldest entry
            oldest_file = None
            oldest_time = None
            
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata.get("created_at", "2000-01-01"))
                    if oldest_time is None or created_at < oldest_time:
                        oldest_time = created_at
                        oldest_file = cache_file
                except Exception:
                    pass
            
            if oldest_file:
                oldest_file.unlink()
            else:
                break
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            logger.debug(f"Cache miss: {key[:30]}...")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if self._is_expired(data):
                file_path.unlink()
                logger.debug(f"Cache expired: {key[:30]}...")
                return None
            
            logger.debug(f"Cache hit: {key[:30]}...")
            return data.get("value")
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            file_path = self._get_file_path(key)
            
            data = {
                "value": value,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f)
            
            self._evict_if_needed()
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cached values."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                value = data.get("value", {})
                # Check if pattern is in sources
                if isinstance(value, dict) and "sources" in value:
                    sources = value.get("sources", [])
                    if any(pattern in str(s) for s in sources):
                        cache_file.unlink()
                        count += 1
            except Exception:
                pass
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "disk",
            "size_mb": self._get_cache_size() / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "entries": len(list(self.cache_dir.glob("*.cache")))
        }


class CacheManager:
    """
    Unified cache manager with fallback support.
    
    Provides a simple interface to different cache backends.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the cache manager."""
        if not settings.CACHE_ENABLED:
            logger.info("Caching is disabled")
            self._cache = None
            return
        
        cache_type = settings.CACHE_TYPE.lower()
        
        if cache_type == "memory":
            self._cache = MemoryCache()
            logger.info("Using in-memory cache")
        elif cache_type == "disk":
            self._cache = DiskCache(
                cache_dir=settings.DISK_CACHE_DIR,
                max_size_mb=100
            )
            logger.info("Using disk cache")
        elif cache_type == "redis":
            try:
                import redis
                self._redis_client = redis.Redis(
                    host=os.environ.get('REDIS_HOST', 'localhost'),
                    port=int(os.environ.get('REDIS_PORT', 6379)),
                    decode_responses=True
                )
                self._cache = RedisCache(self._redis_client)
                logger.info("Using Redis cache")
            except Exception as e:
                logger.warning(f"Redis not available, falling back to disk: {e}")
                self._cache = DiskCache()
        else:
            logger.warning(f"Unknown cache type: {cache_type}, using memory")
            self._cache = MemoryCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._cache is None:
            return None
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if self._cache is None:
            return False
        ttl = ttl or settings.CACHE_TTL
        return self._cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if self._cache is None:
            return False
        return self._cache.delete(key)
    
    def clear(self) -> bool:
        """Clear all cached values."""
        if self._cache is None:
            return False
        return self._cache.clear()
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if self._cache is None:
            return 0
        return self._cache.clear_pattern(pattern)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._cache.stats()
        }
    
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return settings.CACHE_ENABLED and self._cache is not None


# Optional Redis cache implementation
class RedisCache(BaseCache):
    """Redis-based cache for distributed deployments."""
    
    def __init__(self, redis_client):
        """
        Initialize Redis cache.
        
        Args:
            redis_client: Redis client instance
        """
        self.client = redis_client
        self.default_ttl = 3600
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            import json
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache."""
        try:
            import json
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached values (use with caution)."""
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = self.client.keys(f"*{pattern}*")
            if keys:
                self.client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Redis pattern clear error: {e}")
            return 0

