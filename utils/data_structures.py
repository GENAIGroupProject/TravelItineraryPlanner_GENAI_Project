from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

@dataclass
class PreferenceSnippet:
    """Represents a snippet of user preference with semantic embedding."""
    text: str
    embedding: np.ndarray
    slot: str

@dataclass
class PreferenceState:
    """State tracking user preferences across dialogue turns."""
    snippets: List[PreferenceSnippet] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=lambda: {
        "activities": "",
        "pace": "",
        "food": "",
        "constraints": "",
        "budget": "",
        "other": ""
    })
    global_embedding: Optional[np.ndarray] = None
    turns: int = 0

class TripConstraints(BaseModel):
    """Constraints for the trip planning."""
    with_children: Optional[bool] = None
    with_disabled: Optional[bool] = None
    budget: float = Field(..., gt=0, description="Total budget in EUR")
    people: int = Field(..., gt=0, description="Number of people")

class TravelProfile(BaseModel):
    """Complete travel profile for a user."""
    refined_profile: str
    chosen_city: str
    constraints: TripConstraints
    travel_style: Optional[str] = None
    semantic_profile_slots: Dict[str, str] = {}
    interest_embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True

class Attraction(BaseModel):
    """Attraction data model."""
    name: str
    short_description: str
    approx_price_per_person: float = Field(..., ge=0)
    tags: List[str]
    reason_for_user: str
    google_place_id: Optional[str] = None
    opening_hours: Optional[Dict[str, Any]] = None
    google_price_level: Optional[int] = Field(None, ge=0, le=4)
    location: Optional[Dict[str, float]] = None
    google_rating: Optional[float] = Field(None, ge=0, le=5)
    google_user_ratings_total: Optional[int] = Field(None, ge=0)
    final_price_estimate: Optional[float] = Field(None, ge=0)
    
    class Config:
        arbitrary_types_allowed = True

class DayItinerary(BaseModel):
    """Daily itinerary schedule."""
    morning: List[Attraction] = []
    afternoon: List[Attraction] = []
    evening: List[Attraction] = []

class CompleteItinerary(BaseModel):
    """Complete multi-day itinerary."""
    day1: DayItinerary = Field(default_factory=DayItinerary)
    day2: DayItinerary = Field(default_factory=DayItinerary)
    day3: DayItinerary = Field(default_factory=DayItinerary)
    
    def get_day(self, day_number: int) -> DayItinerary:
        """Get itinerary for specific day."""
        if day_number == 1:
            return self.day1
        elif day_number == 2:
            return self.day2
        elif day_number == 3:
            return self.day3
        else:
            return DayItinerary()

class EvaluationScores(BaseModel):
    """Evaluation scores for itinerary."""
    interest_match: int = Field(..., ge=1, le=5)
    budget_realism: int = Field(..., ge=1, le=5)
    logistics: int = Field(..., ge=1, le=5)
    suitability_for_constraints: int = Field(..., ge=1, le=5)
    comment: str
