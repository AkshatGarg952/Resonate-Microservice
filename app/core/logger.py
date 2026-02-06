"""
Structured logging for Resonate Microservice.
"""
import logging
import sys
from datetime import datetime


def setup_logger(name: str = "resonate") -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name for identification
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def log_request(endpoint: str, method: str = "POST") -> None:
    """Log incoming API request."""
    logger.info(f"Request: {method} {endpoint}")


def log_response(endpoint: str, status: str, duration_ms: float = None) -> None:
    """Log API response with optional duration."""
    msg = f"Response: {endpoint} -> {status}"
    if duration_ms:
        msg += f" ({duration_ms:.0f}ms)"
    logger.info(msg)


def log_error(context: str, error: Exception) -> None:
    """Log error with context."""
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")


def log_ai_call(operation: str, model: str) -> None:
    """Log OpenAI API call."""
    logger.info(f"AI Call: {operation} using {model}")
