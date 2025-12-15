import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the travel planner."""
    
    # LLM Configuration
    LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "llama3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Google Places API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyANxjWJzD0BetncqmnBp069mfnawH9xO6g")
    
    # Default trip parameters
    DEFAULT_BUDGET = float(os.getenv("DEFAULT_BUDGET", 600.0))
    DEFAULT_DAYS = int(os.getenv("DEFAULT_DAYS", 3))
    DEFAULT_PEOPLE = int(os.getenv("DEFAULT_PEOPLE", 1))
    
    # Semantic Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # OPTIMIZATION SETTINGS
    LLM_TEMPERATURE = 0.5  # Lower temperature = faster, more deterministic responses
    LLM_MAX_TOKENS = 150   # Limit response length
    ENABLE_STREAMING = True  # If supported by your Ollama setup
    
    # Cache settings
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600  # 1 hour cache
    
    # Parallel processing
    ENABLE_PARALLEL_PROCESSING = False  # Set based on your CPU cores
    
    # Performance thresholds
    MAX_RESPONSE_TIME = 40  # Seconds before timeout
    MIN_SIMILARITY_FOR_CACHE = 0.85  # Cache similar queries
    # Thresholds
    SIM_UPDATE_THRESHOLD = 0.75
    MAX_DIALOGUE_TURNS = 3
    
    # Performance Settings
    REQUEST_TIMEOUT = 600
    GOOGLE_REQUEST_DELAY = 0.1
    
    # Paths
    PROMPTS_DIR = "prompts"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and print warnings."""
        warnings = []
        
        if cls.GOOGLE_API_KEY == "AIzaSyANxjWJzD0BetncqmnBp069mfnawH9xO6g":
            warnings.append("⚠️ Using default Google API key. Replace with your own key.")
        
        if cls.DEFAULT_BUDGET < 100:
            warnings.append("⚠️ Default budget seems low for a 3-day trip.")
        
        for warning in warnings:
            print(warning)
        
        return len(warnings) == 0
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Return all configuration as dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }