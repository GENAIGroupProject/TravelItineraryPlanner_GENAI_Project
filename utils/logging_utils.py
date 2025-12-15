# logging_utils.py - Complete file with unified logging
import logging
import time
import os
import sys
import threading
import json
from datetime import datetime
from typing import Optional, Dict

# Global logger instance
_logger_instance = None

def get_logger():
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logging()
    return _logger_instance

def setup_logging(log_file: str = "travel_planner.log", 
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Setup unified logging configuration.
    
    Args:
        log_file: Path to log file
        console_level: Log level for console output
        file_level: Log level for file output
        
    Returns:
        Configured logger
    """
    global _logger_instance
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("TravelPlanner")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Unified file formatter that includes agent communications
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    
    
    # Unified file handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception:
        # Silent fallback
        pass
    
    # Prevent propagation to root logger
    logger.propagate = False
    _logger_instance = logger
    
    return logger

def log_step(step_name: str, message: str, level: str = "info"):
    """
    Log a step in the pipeline.
    
    Args:
        step_name: Name of the step/agent
        message: Message to log
        level: Log level (info, warning, error, debug)
    """
    logger = get_logger()
    
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

def log_agent_communication(from_agent: str, to_agent: str, 
                           message_type: str, data: Dict, 
                           city: str = None, level: str = "info"):
    """
    Log communication between agents to the unified log file.
    
    Args:
        from_agent: Sending agent
        to_agent: Receiving agent
        message_type: Type of message/communication
        data: Data being communicated
        city: City context (if any)
        level: Log level
    """
    logger = get_logger()
    
    # Create a structured log entry
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_entry = {
        "timestamp": timestamp,
        "type": "agent_communication",
        "from": from_agent,
        "to": to_agent,
        "message_type": message_type,
        "city": city,
        "data_type": type(data).__name__,
        "data_length": len(data) if isinstance(data, (list, dict)) else 1,
        "data_preview": str(data)[:500] if isinstance(data, dict) else str(data)[:200]
    }
    
    # Convert to string format for logging
    log_message = f"AGENT_COMM: {from_agent} -> {to_agent} | TYPE: {message_type} | CITY: {city or 'N/A'} | DATA: {json.dumps(log_entry, ensure_ascii=False)}"
    
    if level.lower() == "info":
        logger.info(log_message)
    elif level.lower() == "warning":
        logger.warning(log_message)
    elif level.lower() == "debug":
        logger.debug(log_message)
    else:
        logger.info(log_message)

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
    logger = get_logger()
    
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
    logger = get_logger()
    
    error_message = f"Error in {agent_name}.{operation}: {str(error)}"
    
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        error_message += f" | Context: {context_str}"
    
    logger.error(error_message, exc_info=True)

# USER-FRIENDLY UI FUNCTIONS

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
    
    @staticmethod
    def clear_screen():
        """Clear screen method."""
        clear_screen()

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
    
    if not logger.handlers:
        # If no handlers, use the unified logger
        logger = get_logger()
    
    # Temporarily disable console handlers
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
    logger = get_logger()
    
    if duration > threshold:
        logger.warning(f"⏱️  SLOW: {agent_name}.{operation} took {duration:.1f}s (> {threshold}s)")
    
    log_performance(agent_name, operation, duration)

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def log_user_input(category: str, message: str):
    """Log user input to unified log file."""
    log_step("USER_INPUT", f"[{category}] {message}")

def log_agent_output(agent_name: str, output_data: Dict, context: str = ""):
    """Log agent output to unified log file."""
    logger = get_logger()
    
    log_entry = {
        "agent": agent_name,
        "context": context,
        "output_preview": str(output_data)[:300],
        "output_type": type(output_data).__name__
    }
    
    logger.info(f"AGENT_OUTPUT: {agent_name} | CONTEXT: {context} | DATA: {json.dumps(log_entry, ensure_ascii=False)}")