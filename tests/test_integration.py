import unittest
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from agents.semantic_agent import SemanticAgent
from agents.interest_refinement_agent import InterestRefinementAgent
from agents.location_scout_agent import LocationScoutAgent
from agents.budget_agent import BudgetAgent
from agents.scheduler_agent import SchedulerAgent
from agents.evaluation_agent import EvaluationAgent
from agents.google_places_agent import GooglePlacesAgent
from utils.data_structures import (
    PreferenceState, Attraction, DayItinerary, 
    CompleteItinerary, TravelProfile, TripConstraints
)

class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        # Mock LLM responses for the entire pipeline
        self.mock_llm = Mock()
        
        # Mock responses for interest refinement
        self.mock_llm.generate.side_effect = [
            # First call: city recommendation
            json.dumps({
                "action": "finalize",
                "question": "",
                "refined_profile": "User interested in ancient buildings and museums",
                "chosen_city": "Rome",
                "constraints": {
                    "with_children": False,
                    "with_disabled": False,
                    "budget": 500,
                    "people": 1
                },
                "travel_style": "medium"
            }),
            # Second call: attractions generation
            json.dumps([
                {
                    "name": "Colosseum",
                    "short_description": "Ancient Roman amphitheater",
                    "approx_price_per_person": 16.0,
                    "tags": ["historical", "ancient", "architecture"],
                    "reason_for_user": "Perfect for ancient building enthusiasts"
                },
                {
                    "name": "Roman Forum",
                    "short_description": "Ancient ruins",
                    "approx_price_per_person": 12.0,
                    "tags": ["historical", "archaeological"],
                    "reason_for_user": "Historical site for history lovers"
                }
            ]),
            # Third call: evaluation
            json.dumps({
                "interest_match": 5,
                "budget_realism": 4,
                "logistics": 4,
                "suitability_for_constraints": 5,
                "comment": "Excellent itinerary for a history enthusiast"
            })
        ]
        
        # Initialize agents with mocked LLM
        self.semantic_agent = SemanticAgent()
        self.interest_agent = InterestRefinementAgent(llm_client=self.mock_llm)
        self.location_agent = LocationScoutAgent(llm_client=self.mock_llm)
        self.budget_agent = BudgetAgent()
        self.scheduler_agent = SchedulerAgent()
        self.evaluation_agent = EvaluationAgent(llm_client=self.mock_llm)
        self.places_agent = GooglePlacesAgent(api_key="disabled")  # Disabled for tests
        
        # Create initial state
        self.state = PreferenceState()
    
    def test_complete_pipeline(self):
        """Test the complete planning pipeline."""
        
        # Step 1: Update state with user preferences
        user_input = "I want to see ancient buildings and museums. Budget is 500 EUR for 1 person for 3 days."
        self.state = self.semantic_agent.update_state(self.state, user_input)
        
        # Step 2: Get city recommendation
        response = self.interest_agent.process_turn(
            self.state, user_input, 500, 1, 3
        )
        
        self.assertEqual(response["action"], "finalize")
        self.assertEqual(response["chosen_city"], "Rome")
        
        # Create profile
        profile = self.interest_agent.create_final_profile(self.state, response)
        self.assertEqual(profile.chosen_city, "Rome")
        self.assertEqual(profile.constraints.budget, 500)
        self.assertEqual(profile.constraints.people, 1)
        
        # Step 3: Generate attractions
        attractions = self.location_agent.generate_attractions(
            profile.chosen_city, 
            profile.refined_profile, 
            profile.constraints.dict()
        )
        
        self.assertGreater(len(attractions), 0)
        self.assertTrue(any("Colosseum" in attr.name for attr in attractions))
        
        # Step 4: Budget filtering
        affordable = self.budget_agent.filter_by_budget(
            attractions, 500, 3, 1
        )
        
        self.assertLessEqual(len(affordable), len(attractions))
        
        # Step 5: Create itinerary
        itinerary = self.scheduler_agent.create_itinerary(affordable, 3)
        
        self.assertIsInstance(itinerary, CompleteItinerary)
        
        # Step 6: Evaluate itinerary
        evaluation = self.evaluation_agent.evaluate_itinerary(
            profile.dict(), 
            itinerary.dict()
        )
        
        self.assertEqual(evaluation.interest_match, 5)
        self.assertEqual(evaluation.budget_realism, 4)
        self.assertIn("Excellent", evaluation.comment)
        
        # Verify all LLM calls were made
        self.assertEqual(self.mock_llm.generate.call_count, 3)
    
    def test_pipeline_with_fallback(self):
        """Test pipeline when some components fail."""
        
        # Mock LLM to fail on attraction generation
        failing_mock_llm = Mock()
        failing_mock_llm.generate.side_effect = [
            # First call succeeds
            json.dumps({
                "action": "finalize",
                "question": "",
                "refined_profile": "Test profile",
                "chosen_city": "Athens",
                "constraints": {
                    "with_children": False,
                    "with_disabled": False,
                    "budget": 400,
                    "people": 2
                },
                "travel_style": "medium"
            }),
            # Second call fails (attraction generation)
            Exception("LLM failed"),
            # Third call for evaluation
            json.dumps({
                "interest_match": 3,
                "budget_realism": 3,
                "logistics": 3,
                "suitability_for_constraints": 3,
                "comment": "Basic itinerary"
            })
        ]
        
        # Create agents with failing LLM
        location_agent = LocationScoutAgent(llm_client=failing_mock_llm)
        evaluation_agent = EvaluationAgent(llm_client=failing_mock_llm)
        
        # Create profile
        profile = TravelProfile(
            refined_profile="Test profile",
            chosen_city="Athens",
            constraints=TripConstraints(
                with_children=False,
                with_disabled=False,
                budget=400,
                people=2
            ),
            travel_style="medium"
        )
        
        # Generate attractions (should use fallback)
        attractions = location_agent.generate_attractions(
            profile.chosen_city,
            profile.refined_profile,
            profile.constraints.dict()
        )
        
        # Should still get attractions from fallback
        self.assertGreater(len(attractions), 0)
        
        # Continue with pipeline
        affordable = self.budget_agent.filter_by_budget(
            attractions, 400, 3, 2
        )
        
        itinerary = self.scheduler_agent.create_itinerary(affordable, 3)
        
        evaluation = evaluation_agent.evaluate_itinerary(
            profile.dict(),
            itinerary.dict()
        )
        
        # Should get evaluation even with basic itinerary
        self.assertIsInstance(evaluation.interest_match, int)
        self.assertGreaterEqual(evaluation.interest_match, 1)
        self.assertLessEqual(evaluation.interest_match, 5)
    
    def test_error_handling_integration(self):
        """Test error handling throughout the pipeline."""
        
        # Test with empty/invalid inputs
        empty_state = PreferenceState()
        
        # Should handle empty input gracefully
        updated_state = self.semantic_agent.update_state(empty_state, "")
        self.assertEqual(updated_state.turns, 0)  # No sentences to process
        
        # Test with minimal valid input
        minimal_state = PreferenceState()
        updated_state = self.semantic_agent.update_state(minimal_state, "museums")
        self.assertEqual(updated_state.turns, 1)
    
    @patch('agents.google_places_agent.requests.get')
    def test_pipeline_with_google_places(self, mock_get):
        """Test pipeline with Google Places integration."""
        
        # Mock Google Places API responses
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"place_id": "test_place_id"}],
            "result": {
                "place_id": "test_place_id",
                "name": "Colosseum",
                "geometry": {"location": {"lat": 41.8902, "lng": 12.4922}},
                "rating": 4.7,
                "user_ratings_total": 150000,
                "price_level": 2,
                "types": ["tourist_attraction", "historical_landmark"]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Create enabled Google Places agent
        places_agent = GooglePlacesAgent(api_key="real_key")
        
        # Create test attractions
        attractions = [
            Attraction(
                name="Colosseum",
                short_description="Ancient Roman amphitheater",
                approx_price_per_person=16.0,
                tags=["historical"],
                reason_for_user="Test"
            )
        ]
        
        # Enrich attractions
        enriched = places_agent.enrich_attractions(attractions, "Rome")
        
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].google_place_id, "test_place_id")
        self.assertEqual(enriched[0].google_rating, 4.7)
        self.assertEqual(enriched[0].google_price_level, 2)
        
        # Continue with rest of pipeline
        affordable = self.budget_agent.filter_by_budget(enriched, 500, 3, 2)
        itinerary = self.scheduler_agent.create_itinerary(affordable, 3)
        
        self.assertIsInstance(itinerary, CompleteItinerary)

class TestAgentInteractions(unittest.TestCase):
    """Test interactions between different agents."""
    
    def test_semantic_to_interest_flow(self):
        """Test flow from semantic understanding to interest refinement."""
        semantic_agent = SemanticAgent()
        interest_agent = InterestRefinementAgent()
        
        # Process user input
        state = PreferenceState()
        state = semantic_agent.update_state(
            state, 
            "I love ancient history and museums. Budget 600 EUR for 2 people."
        )
        
        # Build profile summary
        summary = semantic_agent.build_profile_summary(state)
        
        # Summary should contain the extracted information
        self.assertIn("ancient history", summary.lower())
        self.assertIn("museums", summary.lower())
        self.assertIn("600", summary)
        
        # Interest agent should be able to use this summary
        # (Note: we're not calling the actual LLM in this test)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_location_to_budget_flow(self):
        """Test flow from location scouting to budget filtering."""
        # Create test attractions with different prices
        attractions = [
            Attraction(
                name="Expensive Museum",
                short_description="Luxury museum",
                approx_price_per_person=50.0,
                tags=["museum", "luxury"],
                reason_for_user="For luxury travelers"
            ),
            Attraction(
                name="Free Park",
                short_description="Public park",
                approx_price_per_person=0.0,
                tags=["park", "free", "outdoor"],
                reason_for_user="Budget-friendly option"
            ),
            Attraction(
                name="Mid-range Gallery",
                short_description="Art gallery",
                approx_price_per_person=20.0,
                tags=["gallery", "art"],
                reason_for_user="Cultural experience"
            )
        ]
        
        budget_agent = BudgetAgent()
        
        # Test with limited budget
        affordable = budget_agent.filter_by_budget(attractions, 100, 3, 2)
        
        # Should include free and mid-range, but not expensive
        self.assertLess(len(affordable), len(attractions))
        
        # Check that expensive attraction is filtered out
        expensive_names = [attr.name for attr in affordable if "Expensive" in attr.name]
        self.assertEqual(len(expensive_names), 0)
        
        # Free attraction should be included
        free_names = [attr.name for attr in affordable if "Free" in attr.name]
        self.assertGreater(len(free_names), 0)
    
    def test_budget_to_scheduler_flow(self):
        """Test flow from budget filtering to scheduling."""
        # Create affordable attractions
        affordable = [
            Attraction(
                name=f"Attraction {i}",
                short_description=f"Description {i}",
                approx_price_per_person=10.0 * (i + 1),
                tags=["test"],
                reason_for_user=f"Reason {i}"
            ) for i in range(6)
        ]
        
        scheduler_agent = SchedulerAgent()
        
        # Create itinerary
        itinerary = scheduler_agent.create_itinerary(affordable, 2)
        
        # Should distribute across 2 days
        day1_total = (
            len(itinerary.day1.morning) + 
            len(itinerary.day1.afternoon) + 
            len(itinerary.day1.evening)
        )
        day2_total = (
            len(itinerary.day2.morning) + 
            len(itinerary.day2.afternoon) + 
            len(itinerary.day2.evening)
        )
        
        # Should have some attractions each day
        self.assertGreater(day1_total, 0)
        self.assertGreater(day2_total, 0)
        
        # Should not exceed 3 per day
        self.assertLessEqual(day1_total, 3)
        self.assertLessEqual(day2_total, 3)
    
    def test_scheduler_to_evaluation_flow(self):
        """Test flow from scheduling to evaluation."""
        # Create a simple itinerary
        itinerary = CompleteItinerary(
            day1=DayItinerary(
                morning=[
                    Attraction(
                        name="Morning Museum",
                        short_description="Museum visit",
                        approx_price_per_person=15.0,
                        tags=["museum"],
                        reason_for_user="Cultural experience"
                    )
                ],
                afternoon=[],
                evening=[]
            ),
            day2=DayItinerary(),
            day3=DayItinerary()
        )
        
        # Create a profile
        profile_dict = {
            "refined_profile": "Museum enthusiast",
            "chosen_city": "Rome",
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": 500,
                "people": 1
            },
            "travel_style": "medium"
        }
        
        # Mock evaluation agent
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "interest_match": 4,
            "budget_realism": 5,
            "logistics": 3,
            "suitability_for_constraints": 5,
            "comment": "Good match for interests"
        })
        
        evaluation_agent = EvaluationAgent(llm_client=mock_llm)
        
        # Evaluate itinerary
        evaluation = evaluation_agent.evaluate_itinerary(
            profile_dict,
            itinerary.dict()
        )
        
        # Should get valid scores
        self.assertGreaterEqual(evaluation.interest_match, 1)
        self.assertLessEqual(evaluation.interest_match, 5)
        self.assertIsInstance(evaluation.comment, str)
        self.assertGreater(len(evaluation.comment), 0)

if __name__ == '__main__':
    unittest.main()