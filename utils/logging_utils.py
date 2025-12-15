# logging_utils.py - Complete file with all required functions
import logging
import time
import os
import sys
import threading
from datetime import datetime
from typing import Optional, Dict

def setup_logging(log_file: str = "travel_planner.log", 
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        console_level: Log level for console output
        file_level: Log level for file output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("TravelPlanner")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Could not setup file logging: {e}")
    
    return logger

def log_step(step_name: str, message: str, level: str = "info"):
    """
    Log a step in the pipeline.
    
    Args:
        step_name: Name of the step/agent
        message: Message to log
        level: Log level (info, warning, error, debug)
    """
    logger = logging.getLogger("TravelPlanner")
    
    formatted_message = f"[{step_name.upper()}] {message}"
    
    if level.lower() == "info":
        logger.info(formatted_message)
    elif level.lower() == "warning":
        logger.warning(formatted_message)
    elif level.lower() == "error":
        logger.error(formatted_message)
    elif level.lower() == "debug":
        logger.debug(formatted_message)
    else:
        logger.info(formatted_message)

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        log_step("TIMER", f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        log_step("TIMER", f"Completed {self.name} in {elapsed:.2f} seconds")
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

def log_performance(agent_name: str, operation: str, 
                   duration: float, success: bool = True,
                   additional_info: Optional[Dict] = None):
    """
    Log performance metrics for an agent operation.
    
    Args:
        agent_name: Name of the agent
        operation: Operation performed
        duration: Duration in seconds
        success: Whether operation succeeded
        additional_info: Additional information to log
    """
    logger = logging.getLogger("TravelPlanner")
    
    status = "SUCCESS" if success else "FAILED"
    message = f"Performance - {agent_name}.{operation}: {status} in {duration:.3f}s"
    
    if additional_info:
        message += f" | {additional_info}"
    
    logger.info(message)

def log_error(agent_name: str, operation: str, error: Exception,
             context: Optional[Dict] = None):
    """
    Log an error with context.
    
    Args:
        agent_name: Name of the agent where error occurred
        operation: Operation being performed
        error: Exception that occurred
        context: Additional context information
    """
    logger = logging.getLogger("TravelPlanner")
    
    error_message = f"Error in {agent_name}.{operation}: {str(error)}"
    
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        error_message += f" | Context: {context_str}"
    
    logger.error(error_message, exc_info=True)

# NEW FUNCTIONS FOR USER-FRIENDLY UI

class ConsoleFormatter:
    """Formats console output for better UX."""
    
    @staticmethod
    def loading(message: str = "Processing"):
        """Show a loading message with spinner."""
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        index = datetime.now().microsecond // 100000 % len(spinner)
        return f"{spinner[index]} {message}..."
    
    @staticmethod
    def success(message: str):
        """Show success message."""
        return f"✅ {message}"
    
    @staticmethod
    def info(message: str):
        """Show info message."""
        return f"ℹ️  {message}"
    
    @staticmethod
    def warning(message: str):
        """Show warning message."""
        return f"⚠️  {message}"
    
    @staticmethod
    def error(message: str):
        """Show error message."""
        return f"❌ {message}"
    
    @staticmethod
    def question(message: str):
        """Show question."""
        return f"🤔 {message}"
    
    @staticmethod
    def step(message: str):
        """Show a step in process."""
        return f"📝 {message}"
    
    @staticmethod
    def travel(message: str):
        """Show travel-related info."""
        return f"✈️  {message}"

class LoadingSpinner:
    """Animated loading spinner for long operations."""
    
    def __init__(self, message="Loading", delay=0.1):
        self.message = message
        self.delay = delay
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the spinner animation."""
        if self.running:
            return
        
        self.running = True
        
        def spin():
            i = 0
            while self.running:
                char = self.spinner_chars[i % len(self.spinner_chars)]
                sys.stdout.write(f"\r{char} {self.message}...")
                sys.stdout.flush()
                time.sleep(self.delay)
                i += 1
            sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
            sys.stdout.flush()
        
        self.thread = threading.Thread(target=spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def log_to_file_only(message: str, level: str = "info", logger_name: str = "TravelPlanner"):
    """Log message only to file, not console."""
    logger = logging.getLogger(logger_name)
    
    # Temporarily disable console handler
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    console_levels = {}
    
    for handler in console_handlers:
        console_levels[handler] = handler.level
        handler.setLevel(logging.CRITICAL + 1)  # Higher than any log level
    
    # Log the message
    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    elif level.lower() == "debug":
        logger.debug(message)
    else:
        logger.info(message)
    
    # Restore console handler levels
    for handler, level_val in console_levels.items():
        handler.setLevel(level_val)

def log_performance_with_threshold(agent_name: str, operation: str, 
                                  duration: float, threshold: float = 5.0):
    """Log performance and warn if too slow."""
    logger = logging.getLogger("TravelPlanner")
    
    if duration > threshold:
        logger.warning(f"⏱️  SLOW: {agent_name}.{operation} took {duration:.1f}s (> {threshold}s)")
    
    log_performance(agent_name, operation, duration)

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')