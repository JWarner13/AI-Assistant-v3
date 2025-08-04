import os
import time
import hashlib
import json
import yaml
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
from pathlib import Path
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.logger import get_logger, log_performance, timer
from core.cache import CacheManager, EmbeddingCache

@dataclass
class EmbeddingStats:
    """Statistics for embedding generation and performance."""
    total_requests: int = 0
    total_texts_embedded: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_requests: int = 0
    average_batch_size: float = 0.0
    error_count: int = 0
    
    def add_request(self, text_count: int, processing_time: float, 
                   cached: bool = False, batch_size: int = 1):
        """Add a request to the embedding statistics."""
        self.total_requests += 1
        
        if cached:
            self.cache_hits += text_count
        else:
            self.cache_misses += text_count
            self.total_texts_embedded += text_count
            self.total_processing_time += processing_time
        
        if batch_size > 1:
            self.batch_requests += 1
            # Update average batch size
            self.average_batch_size = (
                (self.average_batch_size * (self.batch_requests - 1) + batch_size) 
                / self.batch_requests
            )
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per request."""
        return self.total_processing_time / max(self.total_requests - (self.cache_hits // max(self.average_batch_size, 1)), 1)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class EnhancedEmbeddingModel:
    """
    Enhanced HuggingFace embedding model with intelligent caching and batch processing.
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 cache_manager: Optional[CacheManager] = None):
        """Initialize the enhanced embedding model."""
        self.config_path = config_path
        self.config = self._load_embedding_config(config_path)
        self.logger = get_logger()
        
        # Initialize cache
        self.cache_manager = cache_manager
        self.embedding_cache = EmbeddingCache(cache_manager) if cache_manager else None
        
        # Statistics and monitoring
        self.stats = EmbeddingStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize the HuggingFace model
        self.model = self._create_huggingface_model()
        self.model_info = self._get_model_info()
        
        # Batch processing settings
        self.batch_size = self.config.get("batch_processing", {}).get("batch_size", 100)
        self.max_batch_size = self.config.get("batch_processing", {}).get("max_batch_size", 500)
        
        self.logger.log("Enhanced embedding model initialized", 
                       component="embedding",
                       extra={
                           "model": self.model_info.get("model_name", "unknown"),
                           "cache_enabled": self.embedding_cache is not None
                       })
    
    def _load_embedding_config(self, config_path: str) -> Dict[str, Any]:
        """Load embedding configuration from YAML file."""
        default_config = {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "chunk_size": 1000,
                "max_retries": 3
            },
            "batch_processing": {
                "batch_size": 100,
                "max_batch_size": 500,
                "parallel_processing": False
            },
            "caching": {
                "enable_embedding_cache": True,
                "cache_ttl": 86400,  # 24 hours
            },
            "optimization": {
                "enable_batch_optimization": True,
                "deduplicate_inputs": True,
                "normalize_whitespace": True
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    
                    # Extract HuggingFace config
                    hf_config = file_config.get("huggingface", {})
                    if "embedding" in hf_config:
                        default_config["embedding"].update(hf_config["embedding"])
                    
                    # Performance and cache configs
                    if "performance" in file_config:
                        perf_config = file_config["performance"]
                        if "batch" in perf_config:
                            default_config["batch_processing"].update(perf_config["batch"])
                    
                    if "cache" in file_config:
                        cache_config = file_config["cache"]
                        default_config["caching"].update(cache_config)
                        
        except Exception as e:
            self.logger.log(f"Error loading embedding config: {e}", 
                           level="WARNING", component="embedding")
        
        return default_config
    
    def _create_huggingface_model(self) -> HuggingFaceEmbeddings:
        """Create Hugging Face embeddings model with configuration."""
        model_name = self.config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            self.logger.log(f"Error creating HuggingFace model: {e}", 
                           level="ERROR", component="embedding")
            # Fallback to default model
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "huggingface",
            "model_name": self.config["embedding"]["model"],
            "dimensions": self.config["embedding"].get("dimensions", 384),
            "cost_per_1k_tokens": 0.0  # Free
        }
    
    def embed_documents(self, texts: List[str], 
                       use_cache: bool = True,
                       batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Embed documents with intelligent caching and batch processing.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            batch_size: Override default batch size
        
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        start_time = time.time()
        actual_batch_size = batch_size or self.batch_size
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Handle caching and deduplication
        embeddings_result = []
        texts_to_embed = []
        cache_map = {}  # Map from original index to cache result
        embed_map = {}  # Map from original index to embedding index
        
        for i, text in enumerate(processed_texts):
            if use_cache and self.embedding_cache:
                cached_embedding = self.embedding_cache.get_embedding(text, self.model_info["model_name"])
                if cached_embedding is not None:
                    embeddings_result.append(cached_embedding)
                    cache_map[i] = len(embeddings_result) - 1
                    continue
            
            # Check for duplicates in current batch
            if self.config.get("optimization", {}).get("deduplicate_inputs", True):
                try:
                    existing_idx = texts_to_embed.index(text)
                    embed_map[i] = existing_idx
                    embeddings_result.append(None)  # Placeholder
                    continue
                except ValueError:
                    pass
            
            embed_map[i] = len(texts_to_embed)
            texts_to_embed.append(text)
            embeddings_result.append(None)  # Placeholder
        
        # Process texts that need embedding
        new_embeddings = []
        if texts_to_embed:
            new_embeddings = self._embed_batch(texts_to_embed, actual_batch_size)
            
            # Store in cache
            if use_cache and self.embedding_cache:
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self.embedding_cache.set_embedding(
                        text, 
                        self.model_info["model_name"], 
                        embedding,
                        ttl=self.config.get("caching", {}).get("cache_ttl", 86400)
                    )
        
        # Reconstruct final embeddings list
        final_embeddings = []
        for i, placeholder in enumerate(embeddings_result):
            if i in cache_map:
                # Use cached result
                final_embeddings.append(embeddings_result[cache_map[i]])
            elif i in embed_map:
                # Use newly computed embedding
                embed_idx = embed_map[i]
                if embed_idx < len(new_embeddings):
                    final_embeddings.append(new_embeddings[embed_idx])
                else:
                    # Handle deduplication
                    original_text = processed_texts[i]
                    for j, text in enumerate(texts_to_embed):
                        if text == original_text:
                            final_embeddings.append(new_embeddings[j])
                            break
        
        processing_time = time.time() - start_time
        
        # Update statistics
        cache_hits = len([i for i in cache_map])
        
        with self._lock:
            self.stats.add_request(
                text_count=len(texts),
                processing_time=processing_time,
                cached=len(texts_to_embed) == 0,
                batch_size=len(texts_to_embed) if texts_to_embed else 1
            )
        
        # Log performance
        self.logger.log_performance(
            "embedding_documents",
            processing_time,
            "embedding",
            {
                "text_count": len(texts),
                "texts_embedded": len(texts_to_embed),
                "cache_hits": cache_hits,
                "cache_misses": len(texts_to_embed),
                "batch_size": actual_batch_size
            }
        )
        
        return final_embeddings
    
    def embed_query(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed a single query with caching.
        
        Args:
            text: Query text to embed
            use_cache: Whether to use caching
        
        Returns:
            Query embedding
        """
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Check cache first
        if use_cache and self.embedding_cache:
            cached_embedding = self.embedding_cache.get_embedding(
                processed_text, 
                self.model_info["model_name"]
            )
            if cached_embedding is not None:
                processing_time = time.time() - start_time
                
                with self._lock:
                    self.stats.add_request(1, processing_time, cached=True)
                
                self.logger.log("Query embedding cache hit", 
                               component="embedding",
                               extra={"text_preview": text[:50]})
                return cached_embedding
        
        # Generate embedding
        try:
            with timer(f"query_embedding_huggingface", "embedding"):
                embedding = self.model.embed_query(processed_text)
            
            # Cache the result
            if use_cache and self.embedding_cache:
                self.embedding_cache.set_embedding(
                    processed_text,
                    self.model_info["model_name"],
                    embedding,
                    ttl=self.config.get("caching", {}).get("cache_ttl", 86400)
                )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self.stats.add_request(1, processing_time, cached=False)
            
            self.logger.log("Query embedding generated", 
                           component="embedding",
                           extra={
                               "text_preview": text[:50],
                               "processing_time": processing_time
                           })
            
            return embedding
            
        except Exception as e:
            with self._lock:
                self.stats.error_count += 1
            
            self.logger.log(f"Error generating query embedding: {e}", 
                           level="ERROR", component="embedding")
            raise
    
    def _embed_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Embed texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for this batch
            try:
                with timer(f"batch_embedding_huggingface_{len(batch_texts)}", "embedding"):
                    batch_embeddings = self.model.embed_documents(batch_texts)
                
                all_embeddings.extend(batch_embeddings)
                
                self.logger.log(f"Batch embedding completed", 
                               component="embedding",
                               extra={
                                   "batch_size": len(batch_texts),
                                   "batch_number": i//batch_size + 1
                               })
                
            except Exception as e:
                with self._lock:
                    self.stats.error_count += 1
                
                self.logger.log(f"Error in batch embedding: {e}", 
                               level="ERROR", component="embedding")
                raise
        
        return all_embeddings
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for embedding generation."""
        if not self.config.get("optimization", {}).get("normalize_whitespace", True):
            return texts
        
        processed = []
        for text in texts:
            processed_text = self._preprocess_text(text)
            processed.append(processed_text)
        
        return processed
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess a single text for embedding generation."""
        if not text:
            return ""
        
        # Normalize whitespace
        if self.config.get("optimization", {}).get("normalize_whitespace", True):
            text = " ".join(text.split())
        
        return text.strip()
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get comprehensive embedding statistics."""
        with self._lock:
            return {
                "requests": {
                    "total": self.stats.total_requests,
                    "errors": self.stats.error_count,
                    "success_rate": (self.stats.total_requests - self.stats.error_count) / max(self.stats.total_requests, 1)
                },
                "performance": {
                    "total_texts_embedded": self.stats.total_texts_embedded,
                    "average_processing_time": self.stats.average_processing_time,
                    "cache_hit_rate": self.stats.cache_hit_rate
                },
                "caching": {
                    "cache_hits": self.stats.cache_hits,
                    "cache_misses": self.stats.cache_misses,
                    "cache_enabled": self.embedding_cache is not None
                },
                "batching": {
                    "batch_requests": self.stats.batch_requests,
                    "average_batch_size": self.stats.average_batch_size
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            **self.model_info,
            "config": {
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "cache_enabled": self.embedding_cache is not None
            }
        }
    
    def reset_stats(self):
        """Reset embedding statistics."""
        with self._lock:
            self.stats = EmbeddingStats()
            self.logger.log("Embedding statistics reset", component="embedding")
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate embedding quality and consistency."""
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        # Check dimensions consistency
        dimensions = set(len(emb) for emb in embeddings)
        if len(dimensions) > 1:
            return {
                "valid": False,
                "error": f"Inconsistent embedding dimensions: {dimensions}"
            }
        
        expected_dim = self.model_info.get("dimensions", 384)
        actual_dim = next(iter(dimensions))
        
        # Check for zero vectors (potentially problematic)
        zero_vectors = sum(1 for emb in embeddings if all(x == 0 for x in emb))
        
        # Check for NaN or infinite values
        invalid_vectors = 0
        for emb in embeddings:
            if any(np.isnan(x) or np.isinf(x) for x in emb):
                invalid_vectors += 1
        
        return {
            "valid": invalid_vectors == 0,
            "total_embeddings": len(embeddings),
            "embedding_dimension": actual_dim,
            "zero_vectors": zero_vectors,
            "invalid_vectors": invalid_vectors,
            "quality_score": 1.0 - (zero_vectors + invalid_vectors) / len(embeddings)
        }

# Global embedding model instance
_enhanced_embedding_model = None

def get_embedding_model(config_path: str = "config.yaml", 
                       cache_manager: Optional[CacheManager] = None) -> EnhancedEmbeddingModel:
    """Get the global enhanced embedding model instance."""
    global _enhanced_embedding_model
    if _enhanced_embedding_model is None:
        _enhanced_embedding_model = EnhancedEmbeddingModel(config_path, cache_manager)
    return _enhanced_embedding_model

def get_simple_embedding_model():
    """Get the underlying embedding model for backward compatibility."""
    enhanced_model = get_embedding_model()
    return enhanced_model.model

# Convenience functions
def embed_documents(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """Embed documents using the enhanced model."""
    model = get_embedding_model()
    return model.embed_documents(texts, use_cache=use_cache)

def embed_query(text: str, use_cache: bool = True) -> List[float]:
    """Embed a query using the enhanced model."""
    model = get_embedding_model()
    return model.embed_query(text, use_cache=use_cache)

def get_embedding_stats() -> Dict[str, Any]:
    """Get embedding statistics."""
    model = get_embedding_model()
    return model.get_embedding_stats()