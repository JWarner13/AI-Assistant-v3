import time
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import threading
from datetime import datetime, timedelta

class CacheManager:
    """
    Enhanced cache manager with TTL support and intelligent eviction.
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 default_ttl: int = 3600,
                 max_size: int = 1000):
        """Initialize the cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size = max_size
        
        # In-memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check TTL
                if time.time() - entry["timestamp"] < entry["ttl"]:
                    self._access_times[key] = time.time()
                    self.stats["hits"] += 1
                    return entry["value"]
                else:
                    # Expired
                    del self._memory_cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        with self._lock:
            if len(self._memory_cache) >= self.max_size:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            self._memory_cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            self._access_times[key] = time.time()
            self.stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            self._access_times.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=self._access_times.get)
        del self._memory_cache[lru_key]
        del self._access_times[lru_key]
        self.stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "evictions": self.stats["evictions"],
            "hit_rate": round(hit_rate, 3),
            "current_size": len(self._memory_cache),
            "max_size": self.max_size
        }

class EmbeddingCache:
    """Specialized cache for embeddings."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._generate_key(text, model)
        return self.cache_manager.get(key)
    
    def set_embedding(self, text: str, model: str, embedding: List[float], ttl: Optional[int] = None) -> None:
        """Store embedding in cache."""
        key = self._generate_key(text, model)
        self.cache_manager.set(key, embedding, ttl)
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding."""
        content = f"embedding:{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

class QueryCache:
    """Specialized cache for query results."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def get_result(self, query: str, context_hash: str) -> Optional[str]:
        """Get query result from cache."""
        key = self._generate_key(query, context_hash)
        return self.cache_manager.get(key)
    
    def set_result(self, query: str, context_hash: str, result: str, ttl: Optional[int] = None) -> None:
        """Store query result in cache."""
        key = self._generate_key(query, context_hash)
        self.cache_manager.set(key, result, ttl)
    
    def _generate_key(self, query: str, context_hash: str) -> str:
        """Generate cache key for query result."""
        content = f"query:{query}:{context_hash}"
        return hashlib.md5(content.encode()).hexdigest()