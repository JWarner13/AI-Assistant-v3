import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from core.cache import CacheManager
from core.rag_engine import _generate_cache_key

# Package metadata
__version__ = "2.0.0"
__author__ = "AI Document Assistant Team"
__description__ = "Enhanced RAG assistant with advanced reasoning and performance optimization"
__license__ = "MIT"

# Minimum Python version check
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later is required.")

# Core module imports with error handling
try:
    # Core RAG components
    from .rag_engine import (
        run_assistant,
        run_batch_queries,
        _generate_cache_key,
        _determine_optimal_k,
        _calculate_confidence_metrics
    )
    
    # Document processing
    from .document_loader import (
        load_documents,
        extract_text_from_pdf,
        extract_text_from_txt,
        split_into_chunks,
        get_document_summary,
        validate_documents
    )
    
    # Embedding system
    from .embedding import (
        get_embedding_model,
        embed_documents,
        embed_query,
        get_embedding_stats,
        estimate_embedding_cost,
        EnhancedEmbeddingModel
    )
    
    # Vector indexing
    from .index import (
        get_index_manager,
        build_index,
        search_index,
        build_and_search,
        get_index_stats,
        IndexManager
    )
    
    # LLM interface
    from .llm import (
        get_llm,
        generate_response,
        get_llm_stats,
        estimate_query_cost,
        EnhancedLLM
    )
    
    # Advanced reasoning
    from .reasoning import (
        explain_reasoning,
        detect_conflicts,
        perform_multi_hop_reasoning
    )
    
    # Logging and monitoring
    from .logger import (
        get_logger,
        log,
        log_performance,
        log_error,
        log_debug,
        get_performance_summary,
        timer,
        log_execution_time,
        log_method_calls,
        EnhancedLogger
    )
    
    # Caching system
    from .cache import (
        CacheManager,
        EmbeddingCache,
        QueryCache
    )
    
    # Output formatting
    from .output_formatter import (
        ResponseFormatter,
        BatchFormatter,
        format_output,
        format_for_api,
        format_for_human,
        format_for_report,
        format_for_business
    )

except ImportError as e:
    print(f"Warning: Could not import all core modules: {e}")
    print("Some functionality may not be available.")

# Package-level configuration
_package_config = {
    "initialized": False,
    "config_path": "config.yaml",
    "cache_manager": None,
    "logger": None,
    "performance_tracking": True,
    "debug_mode": False
}

def initialize_package(config_path: str = "config.yaml", 
                      enable_caching: bool = True,
                      enable_logging: bool = True,
                      debug_mode: bool = False) -> Dict[str, Any]:
    """
    Initialize the AI Document Assistant package with enhanced configuration.
    
    Args:
        config_path: Path to configuration file
        enable_caching: Whether to enable caching system
        enable_logging: Whether to enable enhanced logging
        debug_mode: Whether to enable debug mode
    
    Returns:
        Initialization status and component information
    """
    global _package_config
    
    init_status = {
        "success": True,
        "components": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        # Update package config
        _package_config.update({
            "config_path": config_path,
            "debug_mode": debug_mode
        })
        
        # Initialize logging system
        if enable_logging:
            try:
                logger = get_logger(config_path)
                _package_config["logger"] = logger
                init_status["components"]["logger"] = "initialized"
                logger.log("Package logging system initialized", component="core")
            except Exception as e:
                init_status["errors"].append(f"Logger initialization failed: {e}")
                init_status["components"]["logger"] = "failed"
        
        # Initialize cache manager
        if enable_caching:
            try:
                cache_manager = CacheManager()
                _package_config["cache_manager"] = cache_manager
                init_status["components"]["cache"] = "initialized"
                if _package_config["logger"]:
                    _package_config["logger"].log("Package cache system initialized", component="core")
            except Exception as e:
                init_status["errors"].append(f"Cache initialization failed: {e}")
                init_status["components"]["cache"] = "failed"
        
        # Validate configuration file
        if os.path.exists(config_path):
            init_status["components"]["config"] = "loaded"
            if _package_config["logger"]:
                _package_config["logger"].log(f"Configuration loaded from {config_path}", component="core")
        else:
            init_status["warnings"].append(f"Configuration file not found: {config_path}")
            init_status["components"]["config"] = "default"
        
        # Check environment variables
        env_status = _check_environment()
        init_status["components"].update(env_status)
        
        # Validate dependencies
        deps_status = _validate_dependencies()
        if not deps_status["all_available"]:
            init_status["warnings"].extend(deps_status["missing"])
        
        # Set initialization flag
        _package_config["initialized"] = True
        
        if _package_config["logger"]:
            _package_config["logger"].log("AI Document Assistant package initialized successfully", 
                                        component="core",
                                        extra={
                                            "version": __version__,
                                            "config_path": config_path,
                                            "caching_enabled": enable_caching,
                                            "debug_mode": debug_mode
                                        })
        
    except Exception as e:
        init_status["success"] = False
        init_status["errors"].append(f"Package initialization failed: {e}")
    
    return init_status

def _check_environment() -> Dict[str, str]:
    """Check environment variables and API keys."""
    env_status = {}
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your-api-key-here":
        env_status["openai_api"] = "configured"
    else:
        env_status["openai_api"] = "missing"
    
    # Check required directories
    required_dirs = ["data", "outputs", "cache"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            env_status[f"{dir_name}_dir"] = "exists"
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                env_status[f"{dir_name}_dir"] = "created"
            except Exception:
                env_status[f"{dir_name}_dir"] = "failed"
    
    return env_status

def _validate_dependencies() -> Dict[str, Any]:
    """Validate that all required dependencies are available."""
    required_packages = {
        "openai": "OpenAI API client",
        "langchain": "LangChain framework", 
        "faiss": "FAISS vector search",
        "tiktoken": "OpenAI tokenizer",
        "fitz": "PyMuPDF document processing",
        "pydantic": "Data validation",
        "yaml": "YAML configuration",
        "loguru": "Enhanced logging",
        "numpy": "Numerical computations"
    }
    
    available = []
    missing = []
    
    for package, description in required_packages.items():
        try:
            if package == "fitz":
                import fitz
            elif package == "yaml":
                import yaml
            else:
                __import__(package)
            available.append(package)
        except ImportError:
            missing.append(f"{package} ({description})")
    
    return {
        "all_available": len(missing) == 0,
        "available": available,
        "missing": missing
    }

def get_package_info() -> Dict[str, Any]:
    """Get comprehensive package information and status."""
    return {
        "package": {
            "name": "ai-document-assistant",
            "version": __version__,
            "description": __description__,
            "author": __author__,
            "license": __license__
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "initialized": _package_config["initialized"],
            "config_path": _package_config["config_path"],
            "debug_mode": _package_config["debug_mode"]
        },
        "components": {
            "cache_manager_available": _package_config["cache_manager"] is not None,
            "logger_available": _package_config["logger"] is not None,
            "performance_tracking": _package_config["performance_tracking"]
        },
        "environment": _check_environment(),
        "dependencies": _validate_dependencies()
    }

def get_cache_manager() -> Optional[CacheManager]:
    """Get the package-level cache manager instance."""
    return _package_config.get("cache_manager")

def get_package_logger() -> Optional['EnhancedLogger']:
    """Get the package-level logger instance."""
    return _package_config.get("logger")

def reset_package_state():
    """Reset package state (useful for testing)."""
    global _package_config
    _package_config.update({
        "initialized": False,
        "cache_manager": None,
        "logger": None
    })

def create_assistant(config_path: str = "config.yaml",
                    data_dir: str = "data",
                    enable_caching: bool = True,
                    enable_reasoning: bool = True) -> 'DocumentAssistant':
    """
    Create a pre-configured DocumentAssistant instance.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing documents
        enable_caching: Whether to enable caching
        enable_reasoning: Whether to enable reasoning traces
    
    Returns:
        Configured DocumentAssistant instance
    """
    # Initialize package if not already done
    if not _package_config["initialized"]:
        initialize_package(config_path, enable_caching)
    
    return DocumentAssistant(
        config_path=config_path,
        data_dir=data_dir,
        enable_caching=enable_caching,
        enable_reasoning=enable_reasoning
    )

class DocumentAssistant:
    """
    High-level interface for the AI Document Assistant with all enhanced features.
    """
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 data_dir: str = "data",
                 enable_caching: bool = True,
                 enable_reasoning: bool = True):
        """Initialize the Document Assistant with enhanced features."""
        self.config_path = config_path
        self.data_dir = data_dir
        self.enable_caching = enable_caching
        self.enable_reasoning = enable_reasoning
        
        # Get package-level components
        self.cache_manager = get_cache_manager()
        self.logger = get_package_logger()
        
        # Initialize core components
        self._initialize_components()
        
        if self.logger:
            self.logger.log("DocumentAssistant initialized", 
                           component="assistant",
                           extra={
                               "data_dir": data_dir,
                               "caching": enable_caching,
                               "reasoning": enable_reasoning
                           })
    
    def _initialize_components(self):
        """Initialize core components with package-level instances."""
        # Initialize with package cache manager
        self.embedding_model = get_embedding_model(
            config_path=self.config_path,
            cache_manager=self.cache_manager
        )
        
        self.index_manager = get_index_manager(
            cache_manager=self.cache_manager
        )
        
        self.llm = get_llm(
            config_path=self.config_path,
            cache_manager=self.cache_manager
        )
    
    def query(self, 
             question: str,
             format_type: str = "json",
             enable_reasoning: Optional[bool] = None) -> str:
        """
        Process a single query with enhanced features.
        
        Args:
            question: The question to answer
            format_type: Output format (json, text, markdown, executive)
            enable_reasoning: Override default reasoning setting
        
        Returns:
            Formatted response
        """
        reasoning_enabled = enable_reasoning if enable_reasoning is not None else self.enable_reasoning
        
        # Process the query
        result = run_assistant(
            question,
            enable_reasoning_trace=reasoning_enabled,
            data_dir=self.data_dir,
            use_cache=self.enable_caching
        )
        
        # Format output if not JSON
        if format_type != "json":
            import json
            data = json.loads(result)
            
            if format_type == "text":
                return format_for_human(data)
            elif format_type == "markdown":
                return format_for_report(data)
            elif format_type == "executive":
                return format_for_business(data)
            elif format_type == "api":
                return format_for_api(data)
        
        return result
    
    def batch_query(self, 
                   questions: List[str],
                   enable_reasoning: bool = False) -> Dict[str, Any]:
        """
        Process multiple queries efficiently.
        
        Args:
            questions: List of questions to process
            enable_reasoning: Whether to enable reasoning for all queries
        
        Returns:
            Batch processing results
        """
        return run_batch_queries(
            questions,
            data_dir=self.data_dir,
            enable_reasoning=enable_reasoning,
            use_cache=self.enable_caching
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {}
        
        # Embedding statistics
        try:
            stats["embedding"] = self.embedding_model.get_embedding_stats()
        except Exception as e:
            stats["embedding"] = {"error": str(e)}
        
        # Index statistics
        try:
            stats["indexing"] = self.index_manager.get_performance_summary()
        except Exception as e:
            stats["indexing"] = {"error": str(e)}
        
        # LLM statistics
        try:
            stats["llm"] = self.llm.get_usage_stats()
        except Exception as e:
            stats["llm"] = {"error": str(e)}
        
        # Cache statistics
        if self.cache_manager:
            try:
                stats["cache"] = self.cache_manager.get_stats()
            except Exception as e:
                stats["cache"] = {"error": str(e)}
        
        # Logger performance summary
        if self.logger:
            try:
                stats["performance"] = self.logger.get_performance_summary()
            except Exception as e:
                stats["performance"] = {"error": str(e)}
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear()
        
        if hasattr(self.index_manager, 'clear_search_cache'):
            self.index_manager.clear_search_cache()
        
        if self.logger:
            self.logger.log("All caches cleared", component="assistant")
    
    def reload_documents(self):
        """Reload documents from the data directory."""
        if self.logger:
            self.logger.log(f"Reloading documents from {self.data_dir}", component="assistant")
        
        # Clear relevant caches
        self.clear_cache()
        
        # This would trigger a rebuild on next query
        if self.logger:
            self.logger.log("Documents reload initiated", component="assistant")

# Convenience functions for quick access
def quick_query(question: str, 
               data_dir: str = "data",
               format_type: str = "text") -> str:
    """Quick query interface for simple use cases."""
    assistant = create_assistant(data_dir=data_dir)
    return assistant.query(question, format_type=format_type)

def batch_analyze(questions: List[str], 
                 data_dir: str = "data") -> Dict[str, Any]:
    """Quick batch analysis interface."""
    assistant = create_assistant(data_dir=data_dir)
    return assistant.batch_query(questions)

# Package-level exports
__all__ = [
    # Core functions
    "run_assistant",
    "run_batch_queries",
    
    # Document processing
    "load_documents",
    "get_document_summary",
    "validate_documents",
    
    # Embedding system
    "get_embedding_model",
    "embed_documents", 
    "embed_query",
    "get_embedding_stats",
    
    # Index management
    "get_index_manager",
    "build_index",
    "search_index",
    "get_index_stats",
    
    # LLM interface
    "get_llm",
    "generate_response",
    "get_llm_stats",
    
    # Reasoning system
    "explain_reasoning",
    "detect_conflicts",
    "perform_multi_hop_reasoning",
    
    # Logging and monitoring
    "get_logger",
    "log",
    "log_performance",
    "get_performance_summary",
    "timer",
    
    # Caching system
    "CacheManager",
    "EmbeddingCache",
    
    # Output formatting
    "ResponseFormatter",
    "format_for_api",
    "format_for_human",
    "format_for_report",
    "format_for_business",
    
    # Package management
    "initialize_package",
    "get_package_info",
    "create_assistant",
    "DocumentAssistant",
    
    # Convenience functions
    "quick_query",
    "batch_analyze",
    
    # Core classes
    "EnhancedEmbeddingModel",
    "IndexManager", 
    "EnhancedLLM",
    "EnhancedLogger",
    
    # Package metadata
    "__version__",
    "__author__",
    "__description__"
]

# Auto-initialization check
if not _package_config["initialized"]:
    # Try to auto-initialize with defaults
    try:
        initialize_package()
    except Exception:
        # Silent failure for auto-initialization
        pass