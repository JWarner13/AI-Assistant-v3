"""
Enhanced logging system for the AI Document Assistant.
Provides comprehensive logging with structured output, performance tracking,
and configurable log levels for different components.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import threading
from loguru import logger
import yaml

class EnhancedLogger:
    """
    Enhanced logging system with structured logging, performance tracking,
    and component-specific configuration.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the enhanced logger with configuration."""
        self.config = self._load_logging_config(config_path)
        self.performance_metrics = {}
        self.session_id = self._generate_session_id()
        self._setup_logger()
        self._lock = threading.Lock()
        
        # Component-specific loggers
        self.component_loggers = {}
        self._setup_component_loggers()
        
        # Performance tracking
        self.query_times = []
        self.cache_stats = {"hits": 0, "misses": 0}
        self.error_counts = {"total": 0, "by_component": {}}
        
        logger.info("Enhanced logging system initialized", 
                   extra={"session_id": self.session_id, "config": self.config})
    
    def _load_logging_config(self, config_path: str) -> Dict[str, Any]:
        """Load logging configuration from YAML file."""
        default_config = {
            "level": "INFO",
            "file": "outputs/activity.log",
            "rotation": "10 MB",
            "retention": "30 days",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message} | {extra}",
            "components": {
                "rag_engine": "INFO",
                "document_loader": "INFO",
                "embedding": "INFO",
                "cache": "INFO",
                "reasoning": "DEBUG",
                "cli": "INFO",
                "performance": "INFO"
            },
            "structured_logging": True,
            "performance_tracking": True,
            "error_aggregation": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    logging_config = file_config.get("logging", {})
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in logging_config:
                            logging_config[key] = value
                    return logging_config
        except Exception as e:
            print(f"Warning: Could not load logging config: {e}")
        
        return default_config
    
    def _setup_logger(self):
        """Setup the main logger with configuration."""
        # Remove default handler
        logger.remove()
        
        # Ensure output directory exists
        log_file = self.config["file"]
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        logger.add(
            sys.stderr,
            level=self.config["level"],
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
            colorize=True,
            filter=self._console_filter
        )
        
        # File handler
        logger.add(
            log_file,
            level=self.config["level"],
            format=self.config["format"],
            rotation=self.config["rotation"],
            retention=self.config["retention"],
            compression="zip",
            serialize=self.config.get("structured_logging", False),
            enqueue=True  # Thread-safe logging
        )
        
        # Performance log (separate file)
        if self.config.get("performance_tracking", True):
            perf_log_file = log_file.replace(".log", "_performance.log")
            logger.add(
                perf_log_file,
                level="INFO",
                format="{time} | PERF | {message}",
                filter=lambda record: record["extra"].get("log_type") == "performance",
                rotation="5 MB",
                retention="7 days"
            )
        
        # Error aggregation log
        if self.config.get("error_aggregation", True):
            error_log_file = log_file.replace(".log", "_errors.log")
            logger.add(
                error_log_file,
                level="ERROR",
                format="{time} | ERROR | {name} | {message} | {extra}",
                rotation="5 MB",
                retention="30 days"
            )
    
    def _setup_component_loggers(self):
        """Setup component-specific loggers."""
        components = self.config.get("components", {})
        
        for component, level in components.items():
            component_logger = logger.bind(component=component)
            self.component_loggers[component] = component_logger
    
    def _console_filter(self, record):
        """Filter console output to reduce noise."""
        # Only show INFO and above for console, unless in verbose mode
        if record["level"].no < logger.level("INFO").no:
            return False
        
        # Filter out performance logs from console
        if record["extra"].get("log_type") == "performance":
            return False
        
        return True
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def log(self, message: str, 
            level: str = "INFO", 
            component: str = "general",
            extra: Optional[Dict[str, Any]] = None,
            performance: bool = False):
        """
        Enhanced logging with component tracking and structured data.
        
        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            component: Component name for tracking
            extra: Additional structured data
            performance: Whether this is a performance-related log
        """
        with self._lock:
            # Prepare extra data
            log_extra = {
                "session_id": self.session_id,
                "component": component,
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident
            }
            
            if extra:
                log_extra.update(extra)
            
            if performance:
                log_extra["log_type"] = "performance"
            
            # Get component logger or use general logger
            component_logger = self.component_loggers.get(component, logger)
            
            # Log with appropriate level
            if level.upper() == "DEBUG":
                component_logger.debug(message, extra=log_extra)
            elif level.upper() == "INFO":
                component_logger.info(message, extra=log_extra)
            elif level.upper() == "WARNING":
                component_logger.warning(message, extra=log_extra)
            elif level.upper() == "ERROR":
                component_logger.opt(exception=True).error(message)
                self._track_error(component)
            elif level.upper() == "CRITICAL":
                component_logger.critical(message, extra=log_extra)
                self._track_error(component)
            else:
                component_logger.info(message, extra=log_extra)
    
    def log_performance(self, operation: str, 
                       duration: float, 
                       component: str = "performance",
                       additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics for operations.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            component: Component performing the operation
            additional_metrics: Additional performance data
        """
        perf_data = {
            "operation": operation,
            "duration_seconds": round(duration, 4),
            "component": component,
            "log_type": "performance"
        }
        
        if additional_metrics:
            perf_data.update(additional_metrics)
        
        # Track performance metrics
        with self._lock:
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0
                }
            
            metrics = self.performance_metrics[operation]
            metrics["count"] += 1
            metrics["total_time"] += duration
            metrics["avg_time"] = metrics["total_time"] / metrics["count"]
            metrics["min_time"] = min(metrics["min_time"], duration)
            metrics["max_time"] = max(metrics["max_time"], duration)
        
        self.log(
            f"Performance: {operation} completed in {duration:.4f}s",
            level="INFO",
            component=component,
            extra=perf_data,
            performance=True
        )
    
    def log_query_performance(self, query: str, duration: float, 
                            cache_hit: bool = False,
                            document_count: int = 0,
                            reasoning_enabled: bool = False):
        """
        Log query-specific performance metrics.
        
        Args:
            query: The query string
            duration: Processing duration
            cache_hit: Whether this was a cache hit
            document_count: Number of documents processed
            reasoning_enabled: Whether reasoning was enabled
        """
        # Update cache stats
        with self._lock:
            if cache_hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            
            self.query_times.append(duration)
            
            # Keep only last 100 query times for rolling average
            if len(self.query_times) > 100:
                self.query_times = self.query_times[-100:]
        
        query_data = {
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "query_length": len(query),
            "cache_hit": cache_hit,
            "document_count": document_count,
            "reasoning_enabled": reasoning_enabled,
            "log_type": "performance"
        }
        
        self.log_performance("query_processing", duration, "rag_engine", query_data)
    
    def log_cache_operation(self, operation: str, key: str, hit: bool = None):
        """
        Log cache operations.
        
        Args:
            operation: Cache operation (get, set, delete, clear)
            key: Cache key (truncated for privacy)
            hit: Whether operation was a hit (for get operations)
        """
        cache_data = {
            "operation": operation,
            "key_hash": key[:8] + "..." if len(key) > 8 else key,
            "cache_hit": hit
        }
        
        if hit is not None:
            with self._lock:
                if hit:
                    self.cache_stats["hits"] += 1
                else:
                    self.cache_stats["misses"] += 1
        
        self.log(
            f"Cache {operation}: {cache_data['key_hash']}",
            level="DEBUG",
            component="cache",
            extra=cache_data
        )
    
    def log_document_processing(self, filename: str, 
                              chunks_created: int,
                              processing_time: float,
                              file_size: int,
                              success: bool = True,
                              error: str = None):
        """
        Log document processing results.
        
        Args:
            filename: Name of processed file
            chunks_created: Number of chunks created
            processing_time: Time taken to process
            file_size: Size of file in bytes
            success: Whether processing was successful
            error: Error message if processing failed
        """
        doc_data = {
            "filename": filename,
            "chunks_created": chunks_created,
            "file_size_bytes": file_size,
            "processing_success": success,
            "chunks_per_second": chunks_created / processing_time if processing_time > 0 else 0
        }
        
        if error:
            doc_data["error"] = error
        
        level = "INFO" if success else "ERROR"
        message = f"Document processed: {filename} -> {chunks_created} chunks"
        if not success:
            message = f"Document processing failed: {filename} - {error}"
        
        self.log(message, level=level, component="document_loader", extra=doc_data)
        
        if success:
            self.log_performance("document_processing", processing_time, 
                                "document_loader", doc_data)
    
    def log_reasoning_analysis(self, query: str, reasoning_type: str,
                             complexity: str, conflicts_found: int,
                             analysis_time: float):
        """
        Log reasoning analysis results.
        
        Args:
            query: The query being analyzed
            reasoning_type: Type of reasoning detected
            complexity: Complexity level
            conflicts_found: Number of conflicts detected
            analysis_time: Time taken for analysis
        """
        reasoning_data = {
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "reasoning_type": reasoning_type,
            "complexity_level": complexity,
            "conflicts_detected": conflicts_found,
            "multi_hop_reasoning": complexity in ["high", "very_high"]
        }
        
        self.log(
            f"Reasoning analysis: {reasoning_type} ({complexity}) - {conflicts_found} conflicts",
            level="DEBUG",
            component="reasoning",
            extra=reasoning_data
        )
        
        self.log_performance("reasoning_analysis", analysis_time, 
                           "reasoning", reasoning_data)
    
    def _track_error(self, component: str):
        """Track errors by component."""
        with self._lock:
            self.error_counts["total"] += 1
            if component not in self.error_counts["by_component"]:
                self.error_counts["by_component"][component] = 0
            self.error_counts["by_component"][component] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self._lock:
            cache_total = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = self.cache_stats["hits"] / cache_total if cache_total > 0 else 0
            
            avg_query_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
            
            summary = {
                "session_id": self.session_id,
                "query_performance": {
                    "total_queries": len(self.query_times),
                    "average_time": round(avg_query_time, 4),
                    "fastest_query": round(min(self.query_times), 4) if self.query_times else 0,
                    "slowest_query": round(max(self.query_times), 4) if self.query_times else 0
                },
                "cache_performance": {
                    "hit_rate": round(cache_hit_rate, 3),
                    "total_hits": self.cache_stats["hits"],
                    "total_misses": self.cache_stats["misses"]
                },
                "error_summary": self.error_counts.copy(),
                "operation_metrics": {}
            }
            
            # Add operation-specific metrics
            for operation, metrics in self.performance_metrics.items():
                summary["operation_metrics"][operation] = {
                    "count": metrics["count"],
                    "avg_time": round(metrics["avg_time"], 4),
                    "min_time": round(metrics["min_time"], 4),
                    "max_time": round(metrics["max_time"], 4)
                }
        
        return summary
    
    def log_session_summary(self):
        """Log a summary of the session performance."""
        summary = self.get_performance_summary()
        
        self.log(
            "Session performance summary",
            level="INFO",
            component="performance",
            extra=summary,
            performance=True
        )
    
    def export_logs(self, export_path: str, format_type: str = "json"):
        """
        Export logs in different formats for analysis.
        
        Args:
            export_path: Path to export file
            format_type: Export format (json, csv, yaml)
        """
        summary = self.get_performance_summary()
        
        try:
            if format_type.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump(summary, f, indent=4)
            elif format_type.lower() == "yaml":
                with open(export_path, 'w') as f:
                    yaml.dump(summary, f, default_flow_style=False)
            elif format_type.lower() == "csv":
                self._export_csv(export_path, summary)
            
            self.log(f"Performance logs exported to {export_path}", 
                    level="INFO", component="logger")
                    
        except Exception as e:
            self.log(f"Failed to export logs: {e}", 
                    level="ERROR", component="logger")
    
    def _export_csv(self, export_path: str, summary: Dict[str, Any]):
        """Export performance summary as CSV."""
        import csv
        
        with open(export_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Query performance
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Queries", summary["query_performance"]["total_queries"]])
            writer.writerow(["Average Time", summary["query_performance"]["average_time"]])
            writer.writerow(["Cache Hit Rate", summary["cache_performance"]["hit_rate"]])
            writer.writerow(["Total Errors", summary["error_summary"]["total"]])
            
            # Operation metrics
            writer.writerow([])
            writer.writerow(["Operation", "Count", "Avg Time", "Min Time", "Max Time"])
            for op, metrics in summary["operation_metrics"].items():
                writer.writerow([
                    op, metrics["count"], metrics["avg_time"], 
                    metrics["min_time"], metrics["max_time"]
                ])


# Global logger instance
_enhanced_logger = None

def get_logger(config_path: str = "config.yaml") -> EnhancedLogger:
    """Get the global enhanced logger instance."""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(config_path)
    return _enhanced_logger

# Convenience functions for backward compatibility
def log(message: str, extra: Optional[Dict[str, Any]] = None):
    """Simple logging function for backward compatibility."""
    logger_instance = get_logger()
    logger_instance.log(message, level="INFO", extra=extra or {})

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    logger_instance = get_logger()
    logger_instance.log_performance(operation, duration, **kwargs)

def log_error(message: str, component: str = "general", extra: Optional[Dict[str, Any]] = None):
    """Log error messages."""
    logger_instance = get_logger()
    logger_instance.log(message, level="ERROR", component=component, extra=extra or {})

def log_debug(message: str, component: str = "general", extra: Optional[Dict[str, Any]] = None):
    """Log debug messages."""
    logger_instance = get_logger()
    logger_instance.log(message, level="DEBUG", component=component, extra=extra or {})

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    logger_instance = get_logger()
    return logger_instance.get_performance_summary()

# Context manager for timing operations
class timer:
    """Context manager for timing operations with automatic logging."""
    
    def __init__(self, operation_name: str, component: str = "general", 
                 log_result: bool = True):
        self.operation_name = operation_name
        self.component = component
        self.log_result = log_result
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        if self.log_result:
            logger_instance = get_logger()
            logger_instance.log_performance(
                self.operation_name, 
                self.duration, 
                self.component
            )

# Decorators for automatic performance logging
def log_execution_time(operation_name: str = None, component: str = "general"):
    """Decorator to automatically log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with timer(op_name, component):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def log_method_calls(component: str = None):
    """Decorator to log method calls and their performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            comp = component or args[0].__class__.__name__ if args else "unknown"
            method_name = f"{comp}.{func.__name__}"
            
            logger_instance = get_logger()
            logger_instance.log(
                f"Calling {method_name}",
                level="DEBUG",
                component=comp
            )
            
            with timer(method_name, comp):
                result = func(*args, **kwargs)
            
            logger_instance.log(
                f"Completed {method_name}",
                level="DEBUG", 
                component=comp
            )
            
            return result
        
        return wrapper
    return decorator
