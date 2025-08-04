import os
import time
import json
import yaml
import hashlib
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()  # Loads from `.env` into os.environ
from core.logger import get_logger, log_performance, timer
from core.cache import CacheManager


@dataclass
class LLMUsageStats:
    """Track LLM usage statistics for performance monitoring."""
    total_requests: int = 0
    average_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def add_request(self, response_time: float, cached: bool = False):
        """Add a request to the usage statistics."""
        self.total_requests += 1
        if not cached:
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        
        # Update average response time
        if self.total_requests > 0:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )

class EnhancedLLM:
    """
    Enhanced LLM interface using HuggingFace models with caching and monitoring.
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 cache_manager: Optional[CacheManager] = None):
        """Initialize the enhanced LLM interface."""
        self.config_path = config_path
        self.config = self._load_llm_config(config_path)
        self.logger = get_logger()
        self.cache_manager = cache_manager
        self.usage_stats = LLMUsageStats()
        
        # Initialize model
        self.primary_model = self._initialize_model()
        
        # Response templates for different query types
        self.response_templates = self._load_response_templates()
        
        self.logger.log("Enhanced LLM interface initialized", 
                       component="llm", 
                       extra={"model": self.config["llm"]["model"]})
    
    def _load_llm_config(self, config_path: str) -> Dict[str, Any]:
        """Load LLM configuration from YAML file."""
        default_config = {
            "llm": {
                "model": "google/flan-t5-base",  # More reliable model
                "temperature": 0.1,
                "max_tokens": 512,  # Reduced for free tier
                "max_retries": 3
            },
            "caching": {
                "enable_response_caching": True,
                "cache_ttl": 3600
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    
                    # Check for HuggingFace-specific config
                    hf_config = file_config.get("huggingface", {})
                    if "llm" in hf_config:
                        default_config["llm"].update(hf_config["llm"])
                    
                    # Also check for top-level cache config
                    if "cache" in file_config:
                        default_config["caching"].update(file_config["cache"])
                        
        except Exception as e:
            self.logger.log(f"Error loading LLM config: {e}", level="WARNING", component="llm")
        
        return default_config
    
    def _initialize_model(self):
        """Initialize the HuggingFace model with better error handling."""
        try:
            # Check for HuggingFace Hub token
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                self.logger.log("No HuggingFace API token found - using free tier with limitations", 
                               level="WARNING", component="llm")
            
            llm_config = self.config["llm"]
            
            # Try primary model first
            try:
                model = HuggingFaceEndpoint(
                    repo_id="tiiuae/falcon-7b-instruct",
                    task="text2text-generation",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.1,
                    max_new_tokens=512,
                    do_sample=True
                )
                
                # Test the model with a simple call
                test_response = model.invoke("Test")
                print(f"[DEBUG] HuggingFace test response: {test_response}")
                self.logger.log("Primary HuggingFace model initialized successfully", component="llm")
                return model
                
            except Exception as e:
                self.logger.log(f"Primary model failed: {e}, trying fallback", level="WARNING", component="llm")
                
                # Fallback to simpler model
                fallback_model = HuggingFaceEndpoint(
                    repo_id="tiiuae/falcon-7b-instruct",
                    task="text2text-generation",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.1,
                    max_new_tokens=256,
                    do_sample=False
                )

                self.logger.log("Fallback model initialized successfully", component="llm")
                return fallback_model
                
        except Exception as e:
            self.logger.log(f"All models failed: {e}", level="ERROR", component="llm")
            # Return a mock model that at least doesn't crash
            return MockLLM()
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for different query types."""
        return {
            "default": """Based on the following context, provide a clear and accurate answer to the question.

Context: {context}

Question: {query}

Answer:"""
        }
    
    def invoke(self, prompt: str, 
                query_type: str = "default",
                use_cache: bool = True,
                max_retries: Optional[int] = None) -> str:
        """
        Generate a response using the LLM with enhanced features.
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(prompt, query_type)
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.usage_stats.add_request(time.time() - start_time, cached=True)
                self.logger.log("LLM cache hit", component="llm")
                return cached_response
        
        # Generate response with simplified retry logic
        response = self._generate_response_with_retry(
            prompt, max_retries or self.config["llm"]["max_retries"]
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.usage_stats.add_request(processing_time)
        
        # Cache the response
        if use_cache and self.cache_manager:
            cache_ttl = self.config.get("caching", {}).get("cache_ttl", 3600)
            self.cache_manager.set(cache_key, response, cache_ttl)
        
        return response
    
    def _generate_response_with_retry(self, prompt: str, max_retries: int) -> str:
        """Generate response with simplified retry logic."""
        last_error = None
        
        # Truncate prompt if too long (free tier limitation)
        if len(prompt) > 1000:
            prompt = prompt[:1000] + "..."
        
        for attempt in range(max_retries):
            try:
                self.logger.log(
                    f"Attempting LLM invoke with prompt: {prompt}",
                    component="llm",
                    extra={"attempt": attempt + 1}
                )
                response = self.primary_model.invoke(prompt)
                
                if response and len(response.strip()) > 0:
                    self.logger.log(f"LLM response generated successfully", 
                                   component="llm", extra={"attempt": attempt + 1})
                    return response.strip()
                else:
                    raise Exception("Empty response from model")
                
            except Exception as e:
                last_error = e
                self.usage_stats.error_count += 1

                # Full traceback added here
                error_trace = traceback.format_exc()
                self.logger.log(
                    f"LLM call failed (attempt {attempt + 1}): {e}\n{error_trace}",
                    level="WARNING",
                    component="llm"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # Simple backoff
        
        # If all retries failed, return a fallback response
        fallback_response = f"I apologize, but I'm having trouble processing your request right now. The system processed your documents successfully, but the language model is experiencing issues. Please try again or check your API configuration."
        
        self.logger.log(f"LLM failed after {max_retries} retries, returning fallback", 
                       level="ERROR", component="llm")
        return fallback_response
    
    def _generate_cache_key(self, prompt: str, query_type: str) -> str:
        """Generate a cache key for the prompt."""
        content = f"{prompt}|{query_type}|{self.config['llm']['model']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_with_template(self, context: str, query: str, query_type: str = "default") -> str:
        """Generate response using appropriate template for query type."""
        template = self.response_templates.get(query_type, self.response_templates["default"])
        prompt = template.format(context=context, query=query)
        
        return self.invoke(prompt, query_type=query_type)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        cache_total = self.usage_stats.cache_hits + self.usage_stats.cache_misses
        cache_hit_rate = self.usage_stats.cache_hits / cache_total if cache_total > 0 else 0
        
        return {
            "requests": {
                "total": self.usage_stats.total_requests,
                "errors": self.usage_stats.error_count,
                "success_rate": (self.usage_stats.total_requests - self.usage_stats.error_count) / max(self.usage_stats.total_requests, 1)
            },
            "performance": {
                "average_response_time": self.usage_stats.average_response_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.usage_stats.cache_hits,
                "cache_misses": self.usage_stats.cache_misses
            }
        }

class MockLLM:
    """Mock LLM for fallback when HuggingFace fails."""
    
    def __call__(self, prompt: str) -> str:
        return "I'm a fallback response. The HuggingFace model is currently unavailable, but your document processing system is working correctly."

# Global LLM instance
_enhanced_llm = None

def get_llm(config_path: str = "config.yaml", cache_manager: Optional[CacheManager] = None) -> EnhancedLLM:
    """Get the global enhanced LLM instance."""
    global _enhanced_llm
    if _enhanced_llm is None:
        _enhanced_llm = EnhancedLLM(config_path, cache_manager)
    return _enhanced_llm

def get_simple_llm():
    """Get a simple LLM instance for backward compatibility."""
    enhanced_llm = get_llm()
    return enhanced_llm.primary_model

# Convenience functions
def generate_response(context: str, query: str, query_type: str = "default") -> str:
    """Generate a response using the enhanced LLM with templates."""
    llm = get_llm()
    return llm.generate_with_template(context, query, query_type)

def get_llm_stats() -> Dict[str, Any]:
    """Get LLM usage statistics."""
    llm = get_llm()
    return llm.get_usage_stats()