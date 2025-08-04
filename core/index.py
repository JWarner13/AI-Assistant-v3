"""
Enhanced vector indexing system for the AI Document Assistant.
Provides intelligent document indexing, advanced search capabilities,
performance optimization, and comprehensive index management.
"""

import os
import pickle
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import threading
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from core.logger import get_logger, log_performance, timer
from core.cache import CacheManager

@dataclass
class IndexStats:
    """Statistics for vector index performance and usage."""
    total_documents: int = 0
    total_chunks: int = 0
    index_size_mb: float = 0.0
    last_updated: str = ""
    search_count: int = 0
    average_search_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update_search_stats(self, search_time: float, cache_hit: bool = False):
        """Update search statistics."""
        self.search_count += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # Update average search time (only for non-cached searches)
        if not cache_hit:
            if self.search_count == 1:
                self.average_search_time = search_time
            else:
                # Exclude cached searches from average
                non_cached_count = self.search_count - self.cache_hits
                self.average_search_time = (
                    (self.average_search_time * (non_cached_count - 1) + search_time) 
                    / non_cached_count
                )

class IndexManager:
    """
    Enhanced index manager with caching, persistence, and advanced search capabilities.
    """
    
    def __init__(self, 
                 index_dir: str = "indexes",
                 cache_manager: Optional[CacheManager] = None,
                 enable_persistence: bool = True):
        """Initialize the enhanced index manager."""
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.cache_manager = cache_manager
        self.enable_persistence = enable_persistence
        self.logger = get_logger()
        
        # Index storage
        self.indexes: Dict[str, FAISS] = {}
        self.index_metadata: Dict[str, Dict[str, Any]] = {}
        self.index_stats: Dict[str, IndexStats] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Search result caching
        self.search_cache: Dict[str, List[Document]] = {}
        self.search_cache_ttl = 3600  # 1 hour
        self.search_cache_timestamps: Dict[str, float] = {}
        
        self.logger.log("Enhanced index manager initialized", 
                       component="index", 
                       extra={"index_dir": str(self.index_dir)})
    
    def build_index(self, 
                    text_chunks: List[Dict[str, Any]], 
                    embeddings: Embeddings,
                    index_name: str = "default",
                    force_rebuild: bool = False,
                    save_to_disk: bool = True) -> FAISS:
        """
        Build or load a vector index with enhanced features.
        
        Args:
            text_chunks: List of text chunks with metadata
            embeddings: Embedding model to use
            index_name: Name for the index
            force_rebuild: Force rebuilding even if index exists
            save_to_disk: Whether to save index to disk
        
        Returns:
            FAISS vector index
        """
        with self._lock:
            start_time = time.time()
            
            # Check if index already exists and is current
            if not force_rebuild and self._is_index_current(index_name, text_chunks):
                self.logger.log(f"Loading existing index: {index_name}", component="index")
                return self._load_existing_index(index_name)
            
            self.logger.log(f"Building new index: {index_name}", 
                           component="index",
                           extra={"chunks": len(text_chunks), "force_rebuild": force_rebuild})
            
            # Prepare texts and metadata
            texts = []
            metadatas = []
            
            for chunk in text_chunks:
                texts.append(chunk["content"])
                
                # Enhanced metadata
                metadata = {
                    "source": chunk.get("source", "unknown"),
                    "chunk_id": chunk.get("chunk_metadata", {}).get("chunk_id", 0),
                    "word_count": chunk.get("chunk_metadata", {}).get("word_count", 0),
                    "character_count": chunk.get("chunk_metadata", {}).get("character_count", 0),
                    "content_hash": chunk.get("chunk_metadata", {}).get("content_hash", ""),
                    "global_position": chunk.get("chunk_metadata", {}).get("global_position", 0),
                    "file_type": chunk.get("file_type", "unknown"),
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add file metadata if available
                if "file_metadata" in chunk:
                    metadata["file_metadata"] = chunk["file_metadata"]
                
                metadatas.append(metadata)
            
            try:
                # Build the index with timing
                with timer(f"faiss_index_build_{index_name}", "index"):
                    index = FAISS.from_texts(
                        texts, 
                        embedding=embeddings, 
                        metadatas=metadatas
                    )
                
                build_time = time.time() - start_time
                
                # Store index and metadata
                self.indexes[index_name] = index
                self.index_metadata[index_name] = {
                    "created_at": datetime.now().isoformat(),
                    "document_count": len(set(chunk.get("source", "unknown") for chunk in text_chunks)),
                    "chunk_count": len(text_chunks),
                    "embedding_model": type(embeddings).__name__,
                    "build_time_seconds": build_time,
                    "content_hash": self._calculate_content_hash(text_chunks)
                }
                
                # Initialize statistics
                self.index_stats[index_name] = IndexStats(
                    total_documents=len(set(chunk.get("source", "unknown") for chunk in text_chunks)),
                    total_chunks=len(text_chunks),
                    last_updated=datetime.now().isoformat()
                )
                
                # Save to disk if requested
                if save_to_disk and self.enable_persistence:
                    self._save_index_to_disk(index_name, index)
                
                self.logger.log(f"Index built successfully: {index_name}", 
                               component="index",
                               extra={
                                   "documents": self.index_stats[index_name].total_documents,
                                   "chunks": self.index_stats[index_name].total_chunks,
                                   "build_time": build_time
                               })
                
                # Log performance
                self.logger.log_performance(
                    f"index_build_{index_name}",
                    build_time,
                    "index",
                    {
                        "documents": self.index_stats[index_name].total_documents,
                        "chunks": self.index_stats[index_name].total_chunks,
                        "chunks_per_second": len(text_chunks) / build_time if build_time > 0 else 0
                    }
                )
                
                return index
                
            except Exception as e:
                self.logger.log(f"Error building index {index_name}: {e}", 
                               level="ERROR", component="index")
                raise
    
    def search_index(self, 
                     index: Union[FAISS, str],
                     query: str,
                     k: int = 3,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     use_cache: bool = True,
                     similarity_threshold: float = 0.0) -> List[Document]:
        """
        Search the vector index with enhanced filtering and caching.
        
        Args:
            index: FAISS index or index name
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters to apply
            use_cache: Whether to use search result caching
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Get index if string name provided
        if isinstance(index, str):
            index_name = index
            if index_name not in self.indexes:
                raise ValueError(f"Index '{index_name}' not found")
            actual_index = self.indexes[index_name]
        else:
            index_name = "unknown"
            actual_index = index
        
        # Generate cache key
        cache_key = self._generate_search_cache_key(query, k, filter_metadata, similarity_threshold)
        
        # Check cache first
        if use_cache and self._is_search_cache_valid(cache_key):
            cached_results = self.search_cache[cache_key]
            search_time = time.time() - start_time
            
            # Update stats
            if index_name in self.index_stats:
                self.index_stats[index_name].update_search_stats(search_time, cache_hit=True)
            
            self.logger.log("Search cache hit", 
                           component="index",
                           extra={"query_preview": query[:50], "results": len(cached_results)})
            return cached_results
        
        try:
            with timer(f"vector_search_{index_name}", "index"):
                # Perform the search
                if filter_metadata:
                    # Advanced filtering (if supported by the index)
                    results = self._search_with_metadata_filter(
                        actual_index, query, k, filter_metadata, similarity_threshold
                    )
                else:
                    # Standard similarity search
                    if similarity_threshold > 0:
                        results = actual_index.similarity_search_with_score(query, k=k)
                        # Filter by similarity threshold
                        results = [doc for doc, score in results if score >= similarity_threshold]
                    else:
                        results = actual_index.similarity_search(query, k=k)
            
            search_time = time.time() - start_time
            
            # Cache the results
            if use_cache:
                self.search_cache[cache_key] = results
                self.search_cache_timestamps[cache_key] = time.time()
                
                # Clean old cache entries
                self._clean_search_cache()
            
            # Update statistics
            if index_name in self.index_stats:
                self.index_stats[index_name].update_search_stats(search_time, cache_hit=False)
            
            self.logger.log(f"Vector search completed", 
                           component="index",
                           extra={
                               "query_preview": query[:50],
                               "results": len(results),
                               "search_time": search_time,
                               "index": index_name
                           })
            
            # Log performance
            self.logger.log_performance(
                f"vector_search_{index_name}",
                search_time,
                "index",
                {
                    "query_length": len(query),
                    "k_value": k,
                    "results_found": len(results),
                    "cache_miss": True
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.log(f"Error in vector search: {e}", 
                           level="ERROR", component="index")
            raise
    
    def _search_with_metadata_filter(self, 
                                   index: FAISS, 
                                   query: str, 
                                   k: int,
                                   filter_metadata: Dict[str, Any],
                                   similarity_threshold: float) -> List[Document]:
        """Search with metadata filtering (simplified implementation)."""
        # Get more results than needed for filtering
        extended_k = min(k * 3, 50)  # Get 3x more for filtering
        
        if similarity_threshold > 0:
            results_with_scores = index.similarity_search_with_score(query, k=extended_k)
            filtered_results = []
            
            for doc, score in results_with_scores:
                if score >= similarity_threshold and self._matches_filter(doc.metadata, filter_metadata):
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break
        else:
            all_results = index.similarity_search(query, k=extended_k)
            filtered_results = []
            
            for doc in all_results:
                if self._matches_filter(doc.metadata, filter_metadata):
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break
        
        return filtered_results[:k]
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def _is_index_current(self, index_name: str, text_chunks: List[Dict[str, Any]]) -> bool:
        """Check if existing index is current with the provided chunks."""
        if index_name not in self.index_metadata:
            return False
        
        # Check if content has changed
        current_hash = self._calculate_content_hash(text_chunks)
        stored_hash = self.index_metadata[index_name].get("content_hash", "")
        
        return current_hash == stored_hash
    
    def _calculate_content_hash(self, text_chunks: List[Dict[str, Any]]) -> str:
        """Calculate a hash of the content for change detection."""
        import hashlib
        
        # Create a hash based on all content hashes
        content_hashes = []
        for chunk in text_chunks:
            chunk_hash = chunk.get("chunk_metadata", {}).get("content_hash", "")
            if chunk_hash:
                content_hashes.append(chunk_hash)
        
        combined_content = "|".join(sorted(content_hashes))
        return hashlib.md5(combined_content.encode()).hexdigest()
    
    def _load_existing_index(self, index_name: str) -> FAISS:
        """Load an existing index from memory or disk."""
        if index_name in self.indexes:
            return self.indexes[index_name]
        
        # Try to load from disk
        index_path = self.index_dir / f"{index_name}.faiss"
        metadata_path = self.index_dir / f"{index_name}_metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                index = FAISS.load_local(str(index_path.parent), index_name)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Store in memory
                self.indexes[index_name] = index
                self.index_metadata[index_name] = metadata
                
                # Initialize stats
                self.index_stats[index_name] = IndexStats(
                    total_documents=metadata.get("document_count", 0),
                    total_chunks=metadata.get("chunk_count", 0),
                    last_updated=metadata.get("created_at", "")
                )
                
                self.logger.log(f"Loaded index from disk: {index_name}", component="index")
                return index
                
            except Exception as e:
                self.logger.log(f"Error loading index from disk: {e}", 
                               level="WARNING", component="index")
        
        raise ValueError(f"Index '{index_name}' not found in memory or disk")
    
    def _save_index_to_disk(self, index_name: str, index: FAISS):
        """Save index and metadata to disk."""
        try:
            # Save FAISS index
            index.save_local(str(self.index_dir), index_name)
            
            # Save metadata
            metadata_path = self.index_dir / f"{index_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.index_metadata[index_name], f, indent=2)
            
            # Calculate and update index size
            index_path = self.index_dir / f"{index_name}.faiss"
            if index_path.exists():
                size_mb = index_path.stat().st_size / (1024 * 1024)
                if index_name in self.index_stats:
                    self.index_stats[index_name].index_size_mb = size_mb
            
            self.logger.log(f"Index saved to disk: {index_name}", 
                           component="index",
                           extra={"path": str(self.index_dir)})
            
        except Exception as e:
            self.logger.log(f"Error saving index to disk: {e}", 
                           level="ERROR", component="index")
    
    def _generate_search_cache_key(self, 
                                 query: str, 
                                 k: int, 
                                 filter_metadata: Optional[Dict[str, Any]],
                                 similarity_threshold: float) -> str:
        """Generate a cache key for search results."""
        import hashlib
        
        filter_str = json.dumps(filter_metadata or {}, sort_keys=True)
        content = f"{query}|{k}|{filter_str}|{similarity_threshold}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_search_cache_valid(self, cache_key: str) -> bool:
        """Check if cached search result is still valid."""
        if cache_key not in self.search_cache:
            return False
        
        timestamp = self.search_cache_timestamps.get(cache_key, 0)
        return time.time() - timestamp < self.search_cache_ttl
    
    def _clean_search_cache(self):
        """Clean expired entries from search cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.search_cache_timestamps.items():
            if current_time - timestamp > self.search_cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.search_cache[key]
            del self.search_cache_timestamps[key]
    
    def get_index_info(self, index_name: str = "default") -> Dict[str, Any]:
        """Get comprehensive information about an index."""
        if index_name not in self.indexes:
            return {"error": f"Index '{index_name}' not found"}
        
        metadata = self.index_metadata.get(index_name, {})
        stats = self.index_stats.get(index_name, IndexStats())
        
        # Calculate cache hit rate
        total_searches = stats.cache_hits + stats.cache_misses
        cache_hit_rate = stats.cache_hits / total_searches if total_searches > 0 else 0
        
        return {
            "index_name": index_name,
            "metadata": metadata,
            "statistics": {
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "index_size_mb": stats.index_size_mb,
                "last_updated": stats.last_updated,
                "search_count": stats.search_count,
                "average_search_time": round(stats.average_search_time, 4),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses
            },
            "status": "active" if index_name in self.indexes else "inactive"
        }
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all available indexes."""
        indexes = []
        
        for index_name in self.indexes.keys():
            info = self.get_index_info(index_name)
            indexes.append({
                "name": index_name,
                "documents": info["statistics"]["total_documents"],
                "chunks": info["statistics"]["total_chunks"],
                "size_mb": info["statistics"]["index_size_mb"],
                "searches": info["statistics"]["search_count"],
                "last_updated": info["statistics"]["last_updated"]
            })
        
        return indexes
    
    def delete_index(self, index_name: str, remove_from_disk: bool = True):
        """Delete an index from memory and optionally from disk."""
        with self._lock:
            # Remove from memory
            if index_name in self.indexes:
                del self.indexes[index_name]
            if index_name in self.index_metadata:
                del self.index_metadata[index_name]
            if index_name in self.index_stats:
                del self.index_stats[index_name]
            
            # Remove from disk
            if remove_from_disk:
                index_path = self.index_dir / f"{index_name}.faiss"
                metadata_path = self.index_dir / f"{index_name}_metadata.json"
                
                for path in [index_path, metadata_path]:
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception as e:
                            self.logger.log(f"Error deleting {path}: {e}", 
                                           level="WARNING", component="index")
            
            self.logger.log(f"Index deleted: {index_name}", component="index")
    
    def clear_search_cache(self):
        """Clear all cached search results."""
        with self._lock:
            self.search_cache.clear()
            self.search_cache_timestamps.clear()
            self.logger.log("Search cache cleared", component="index")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all indexes."""
        summary = {
            "total_indexes": len(self.indexes),
            "indexes": {},
            "overall_stats": {
                "total_searches": 0,
                "total_cache_hits": 0,
                "total_cache_misses": 0,
                "average_search_time": 0.0
            }
        }
        
        total_search_time = 0
        non_cached_searches = 0
        
        for index_name, stats in self.index_stats.items():
            summary["indexes"][index_name] = {
                "searches": stats.search_count,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "average_search_time": stats.average_search_time,
                "cache_hit_rate": stats.cache_hits / max(stats.search_count, 1)
            }
            
            # Aggregate overall stats
            summary["overall_stats"]["total_searches"] += stats.search_count
            summary["overall_stats"]["total_cache_hits"] += stats.cache_hits
            summary["overall_stats"]["total_cache_misses"] += stats.cache_misses
            
            if stats.cache_misses > 0:
                total_search_time += stats.average_search_time * stats.cache_misses
                non_cached_searches += stats.cache_misses
        
        # Calculate overall average search time (excluding cached searches)
        if non_cached_searches > 0:
            summary["overall_stats"]["average_search_time"] = total_search_time / non_cached_searches
        
        return summary

# Global index manager
_index_manager = None

def get_index_manager(index_dir: str = "indexes", 
                     cache_manager: Optional[CacheManager] = None) -> IndexManager:
    """Get the global index manager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager(index_dir, cache_manager)
    return _index_manager

# Backward compatibility functions
def build_index(text_chunks: List[Dict[str, Any]], embeddings: Embeddings) -> FAISS:
    """Build index using the enhanced index manager (backward compatibility)."""
    manager = get_index_manager()
    return manager.build_index(text_chunks, embeddings)

def search_index(index: FAISS, query: str, k: int = 3) -> List[Document]:
    """Search index using the enhanced manager (backward compatibility)."""
    manager = get_index_manager()
    return manager.search_index(index, query, k)

# Convenience functions
def build_and_search(text_chunks: List[Dict[str, Any]], 
                    embeddings: Embeddings,
                    query: str,
                    k: int = 3,
                    index_name: str = "default") -> List[Document]:
    """Build index and search in one operation."""
    manager = get_index_manager()
    index = manager.build_index(text_chunks, embeddings, index_name)
    return manager.search_index(index, query, k)

def get_index_stats(index_name: str = "default") -> Dict[str, Any]:
    """Get statistics for a specific index."""
    manager = get_index_manager()
    return manager.get_index_info(index_name)