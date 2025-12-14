"""
Utilities module for Travel Itinerary Planner.
"""

from .llm_client import LLMClient
from .data_structures import (
    PreferenceSnippet, PreferenceState, TripConstraints,
    TravelProfile, Attraction, DayItinerary, CompleteItinerary,
    EvaluationScores
)
from .json_parser import JSONParser
from .logging_utils import setup_logging, log_step
from .pydantic_compat import to_dict

__all__ = [
    'LLMClient',
    'PreferenceSnippet', 'PreferenceState', 'TripConstraints',
    'TravelProfile', 'Attraction', 'DayItinerary', 'CompleteItinerary',
    'EvaluationScores',
    'JSONParser',
    'setup_logging', 'log_step',
    'to_dict'
]
