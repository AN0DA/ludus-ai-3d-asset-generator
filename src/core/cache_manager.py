"""
Cache management system for optimizing repeated operations.

This module provides intelligent caching with size limits,
TTL expiration, and automatic cleanup of cached data.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """Manages cached data with size limits and TTL."""
    
    def __init__(self, cache_dir: Path, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB default
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.cache_index_file = cache_dir / "cache_index.json"
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_cache_index()
    
    def _load_cache_index(self) -> None:
        """Load the cache index from disk."""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logger.info(f"Loaded cache index with {len(self.cache_index)} entries")
            else:
                self.cache_index = {}
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate a cache key from data."""
        if isinstance(data, dict):
            # Sort dict for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
        else:
            sorted_data = str(data)
        
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def get(self, key: str, data: Any, ttl_seconds: int = 3600) -> Optional[Any]:
        """Get cached data if it exists and is not expired."""
        cache_key = self._get_cache_key(data)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_info = self.cache_index[cache_key]
        
        # Check TTL
        if time.time() - cache_info["created_at"] > ttl_seconds:
            self.invalidate(key, data)
            return None
        
        # Load cached data
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Update access time
                cache_info["last_accessed"] = time.time()
                self._save_cache_index()
                
                logger.debug(f"Cache hit for key: {key}")
                return cached_data
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            self.invalidate(key, data)
        
        return None
    
    def set(self, key: str, data: Any, value: Any) -> None:
        """Cache data with automatic cleanup if needed."""
        cache_key = self._get_cache_key(data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Save cached data
            with open(cache_file, 'w') as f:
                json.dump(value, f, indent=2, default=str)
            
            # Update cache index
            file_size = cache_file.stat().st_size
            self.cache_index[cache_key] = {
                "key": key,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "file_size": file_size,
                "cache_file": str(cache_file)
            }
            
            self._save_cache_index()
            self._cleanup_if_needed()
            
            logger.debug(f"Cached data for key: {key}")
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def invalidate(self, key: str, data: Any) -> None:
        """Remove cached data."""
        cache_key = self._get_cache_key(data)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit."""
        total_size = sum(info["file_size"] for info in self.cache_index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time and remove oldest
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            for cache_key, info in sorted_entries:
                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[cache_key]
                total_size -= info["file_size"]
                
                if total_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                    break
            
            self._save_cache_index()
            logger.info("Cache cleanup completed")
    
    def clear_all(self) -> None:
        """Clear all cached data."""
        for cache_key in list(self.cache_index.keys()):
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
        
        self.cache_index.clear()
        self._save_cache_index()
        logger.info("All cached data cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(info["file_size"] for info in self.cache_index.values())
        return {
            "total_entries": len(self.cache_index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "usage_percentage": round((total_size / self.max_size_bytes) * 100, 2)
        }
