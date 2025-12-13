from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TestCase:
    """Represents a test case for evaluation."""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    category: str  # performance, relevance, usability, etc.

def get_test_cases(category: str = None) -> List[TestCase]:
    """Get test cases for evaluation."""
    all_cases = [
        # Interest Refinement Agent Tests
        TestCase(
            name="museum_enthusiast",
            description="User interested in museums and historical sites",
            input_data={
                "user_input": "I love museums and historical sites",
                "budget": 500,
                "people": 2,
                "days": 3
            },
            expected_output={
                "should_recommend_city": True,
                "expected_cities": ["Rome", "Paris", "London", "Berlin"],
                "max_questions": 5
            },
            category="relevance"
        ),
        
        TestCase(
            name="budget_traveler",
            description="User with limited budget",
            input_data={
                "user_input": "I want to travel cheaply",
                "budget": 300,
                "people": 1,
                "days": 3
            },
            expected_output={
                "should_recommend_affordable_cities": True,
                "max_budget_per_day": 100
            },
            category="budget"
        ),
        
        TestCase(
            name="family_travel",
            description="Family with children",
            input_data={
                "user_input": "Traveling with 2 kids",
                "budget": 800,
                "people": 4,
                "days": 4
            },
            expected_output={
                "should_ask_about_children": True,
                "should_recommend_family_friendly": True
            },
            category="constraints"
        ),
        
        # Location Scout Agent Tests
        TestCase(
            name="rome_attractions",
            description="Generate attractions for Rome",
            input_data={
                "city": "Rome",
                "interests": "ancient buildings, museums",
                "budget": 600,
                "people": 2
            },
            expected_output={
                "min_attractions": 8,
                "max_attractions": 12,
                "expected_themes": ["historical", "ancient", "museum"]
            },
            category="relevance"
        ),
        
        TestCase(
            name="paris_attractions",
            description="Generate attractions for Paris",
            input_data={
                "city": "Paris",
                "interests": "art, food, romance",
                "budget": 1000,
                "people": 2
            },
            expected_output={
                "min_attractions": 8,
                "max_attractions": 12,
                "expected_themes": ["art", "food", "romantic"]
            },
            category="relevance"
        ),
        
        # Budget Agent Tests
        TestCase(
            name="budget_filtering",
            description="Filter attractions by budget",
            input_data={
                "attractions": [
                    {"name": "Expensive Attraction", "approx_price_per_person": 50},
                    {"name": "Medium Attraction", "approx_price_per_person": 20},
                    {"name": "Free Attraction", "approx_price_per_person": 0},
                ],
                "total_budget": 100,
                "people": 2,
                "days": 2
            },
            expected_output={
                "max_selected": 6,  # 3 per day * 2 days
                "should_include_free": True,
                "total_cost_under_budget": True
            },
            category="budget"
        ),
        
        # Scheduler Agent Tests
        TestCase(
            name="schedule_creation",
            description="Create schedule from attractions",
            input_data={
                "attractions": [{"name": f"Attraction {i}"} for i in range(10)],
                "days": 3
            },
            expected_output={
                "total_scheduled": 9,  # 3 per day
                "days_with_attractions": 3,
                "balanced_distribution": True
            },
            category="logistics"
        ),
        
        # Performance Tests
        TestCase(
            name="quick_response",
            description="Test response time for simple query",
            input_data={
                "user_input": "museums",
                "budget": 500,
                "people": 1,
                "days": 2
            },
            expected_output={
                "max_response_time": 30.0,  # seconds
                "should_complete": True
            },
            category="performance"
        ),
        
        TestCase(
            name="complex_query",
            description="Test with complex preferences",
            input_data={
                "user_input": "I want to see ancient buildings, eat local food, and experience nightlife",
                "budget": 1000,
                "people": 3,
                "days": 4
            },
            expected_output={
                "max_response_time": 60.0,
                "should_handle_complexity": True
            },
            category="performance"
        ),
        
        # Error Handling Tests
        TestCase(
            name="invalid_input",
            description="Test with invalid input",
            input_data={
                "user_input": "",  # Empty input
                "budget": -100,  # Invalid budget
                "people": 0,  # Invalid people count
                "days": 0  # Invalid days
            },
            expected_output={
                "should_handle_gracefully": True,
                "should_provide_defaults": True,
                "should_not_crash": True
            },
            category="error_handling"
        ),
        
        TestCase(
            name="api_failure",
            description="Test when external API fails",
            input_data={
                "city": "UnknownCity123",  # Non-existent city
                "interests": "anything",
                "budget": 500,
                "people": 2
            },
            expected_output={
                "should_use_fallback": True,
                "should_not_crash": True
            },
            category="error_handling"
        )
    ]
    
    if category:
        return [case for case in all_cases if case.category == category]
    return all_cases

def get_stress_test_cases() -> List[TestCase]:
    """Get stress test cases for performance evaluation."""
    return [
        TestCase(
            name="many_attractions",
            description="Test with many attractions",
            input_data={
                "attractions": [{"name": f"Attraction {i}"} for i in range(50)],
                "days": 3
            },
            expected_output={
                "max_processing_time": 5.0,
                "should_complete": True
            },
            category="performance"
        ),
        
        TestCase(
            name="concurrent_requests",
            description="Simulate concurrent user requests",
            input_data={
                "concurrent_users": 5,
                "requests_per_user": 3,
                "delay_between_requests": 0.1
            },
            expected_output={
                "system_should_remain_stable": True,
                "max_response_time_per_request": 60.0
            },
            category="performance"
        )
    ]

def get_usability_test_cases() -> List[TestCase]:
    """Get usability test cases."""
    return [
        TestCase(
            name="dialogue_naturalness",
            description="Test dialogue flow and naturalness",
            input_data={
                "dialogue_scenario": "User gradually reveals preferences"
            },
            expected_output={
                "questions_should_be_relevant": True,
                "should_not_repeat_questions": True,
                "should_progress_logically": True
            },
            category="usability"
        ),
        
        TestCase(
            name="result_clarity",
            description="Test clarity of final results",
            input_data={
                "check_metrics": ["format", "completeness", "readability"]
            },
            expected_output={
                "results_should_be_clear": True,
                "should_include_all_sections": True,
                "should_be_well_formatted": True
            },
            category="usability"
        )
    ]
