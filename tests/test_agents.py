import unittest
import json
from unittest.mock import Mock, patch
import numpy as np

from agents.semantic_agent import SemanticAgent
from agents.interest_refinement_agent import InterestRefinementAgent
from agents.location_scout_agent import LocationScoutAgent
from agents.budget_agent import BudgetAgent
from agents.scheduler_agent import SchedulerAgent
from agents.evaluation_agent import EvaluationAgent
from agents.google_places_agent import GooglePlacesAgent
from utils.data_structures import (
    PreferenceState, PreferenceSnippet, Attraction,
    DayItinerary, CompleteItinerary, TripConstraints, TravelProfile
)

class TestSemanticAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SemanticAgent()
    
    def test_split_into_sentences(self):
        text = "Hello world. How are you? I'm fine!"
        sentences = self.agent.split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Hello world")
        self.assertEqual(sentences[1], "How are you")
        self.assertEqual(sentences[2], "I'm fine")
    
    def test_update_state(self):
        state = PreferenceState()
        user_msg = "I love museums and historical sites. Budget is 500 EUR."
        
        updated_state = self.agent.update_state(state, user_msg)
        
        self.assertGreater(len(updated_state.snippets), 0)
        self.assertIsNotNone(updated_state.global_embedding)
        self.assertEqual(updated_state.turns, 1)
    
    def test_build_profile_summary(self):
        state = PreferenceState()
        state.slots["activities"] = "museums, historical sites"
        state.slots["budget"] = "500 EUR"
        
        summary = self.agent.build_profile_summary(state)
        
        self.assertIn("Activities: museums, historical sites", summary)
        self.assertIn("Budget: 500 EUR", summary)

class TestInterestRefinementAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.generate.return_value = json.dumps({
            "action": "finalize",
            "question": "",
            "refined_profile": "User interested in museums",
            "chosen_city": "Rome",
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": 500,
                "people": 2
            },
            "travel_style": "medium"
        })
        
        self.agent = InterestRefinementAgent(llm_client=self.mock_llm)
    
    def test_process_turn(self):
        state = PreferenceState()
        state.slots["activities"] = "museums"
        
        response = self.agent.process_turn(
            state, "I like museums", 500, 2, 3
        )
        
        self.assertEqual(response["action"], "finalize")
        self.assertEqual(response["chosen_city"], "Rome")
        self.assertEqual(response["constraints"]["budget"], 500)
        self.assertEqual(response["constraints"]["people"], 2)
    
    def test_create_final_profile(self):
        state = PreferenceState()
        state.slots["activities"] = "museums"
        state.global_embedding = np.array([0.1, 0.2, 0.3])
        
        llm_output = {
            "action": "finalize",
            "question": "",
            "refined_profile": "Museum enthusiast",
            "chosen_city": "Rome",
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": 500,
                "people": 2
            },
            "travel_style": "medium"
        }
        
        profile = self.agent.create_final_profile(state, llm_output)
        
        self.assertEqual(profile.chosen_city, "Rome")
        self.assertEqual(profile.constraints.budget, 500)
        self.assertEqual(profile.constraints.people, 2)
        self.assertIsNotNone(profile.interest_embedding)

class TestLocationScoutAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.generate.return_value = json.dumps([
            {
                "name": "Test Museum",
                "short_description": "A test museum",
                "approx_price_per_person": 15.0,
                "tags": ["museum", "historical"],
                "reason_for_user": "Perfect for museum lovers"
            }
        ])
        
        self.agent = LocationScoutAgent(llm_client=self.mock_llm)
    
    def test_generate_attractions(self):
        constraints = {
            "with_children": False,
            "with_disabled": False,
            "budget": 500,
            "people": 2
        }
        
        attractions = self.agent.generate_attractions(
            "Rome", "User likes museums", constraints
        )
        
        self.assertGreater(len(attractions), 0)
        self.assertEqual(attractions[0].name, "Test Museum")
        self.assertEqual(attractions[0].approx_price_per_person, 15.0)
    
    def test_get_fallback_attractions(self):
        attractions = self.agent.get_fallback_attractions("Rome")
        
        self.assertGreater(len(attractions), 0)
        self.assertTrue(any("Colosseum" in attr.name for attr in attractions))

class TestBudgetAgent(unittest.TestCase):
    def setUp(self):
        self.agent = BudgetAgent()
    
    def test_filter_by_budget(self):
        attractions = [
            Attraction(
                name="Expensive",
                short_description="Test",
                approx_price_per_person=50.0,
                tags=["test"],
                reason_for_user="Test"
            ),
            Attraction(
                name="Cheap",
                short_description="Test",
                approx_price_per_person=10.0,
                tags=["test"],
                reason_for_user="Test"
            ),
            Attraction(
                name="Free",
                short_description="Test",
                approx_price_per_person=0.0,
                tags=["test"],
                reason_for_user="Test"
            )
        ]
        
        filtered = self.agent.filter_by_budget(attractions, 100, 2, 2)
        
        # Should include cheap and free, but not expensive
        self.assertLess(len(filtered), len(attractions))
        self.assertTrue(any("Free" in attr.name for attr in filtered))
    
    def test_calculate_budget_summary(self):
        attractions = [
            Attraction(
                name="Test",
                short_description="Test",
                approx_price_per_person=20.0,
                tags=["test"],
                reason_for_user="Test",
                final_price_estimate=40.0  # For 2 people
            )
        ]
        
        summary = self.agent.calculate_budget_summary(attractions, 100, 2)
        
        self.assertEqual(summary["total_cost"], 40.0)
        self.assertEqual(summary["remaining_budget"], 60.0)
        self.assertEqual(summary["cost_per_person"], 20.0)

class TestSchedulerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SchedulerAgent()
    
    def test_create_itinerary(self):
        attractions = [
            Attraction(
                name=f"Attraction {i}",
                short_description=f"Test {i}",
                approx_price_per_person=10.0,
                tags=["test"],
                reason_for_user=f"Test {i}"
            ) for i in range(9)
        ]
        
        itinerary = self.agent.create_itinerary(attractions, 3)
        
        self.assertIsInstance(itinerary, CompleteItinerary)
        
        # Should have attractions distributed across days
        total_attractions = (
            len(itinerary.day1.morning) + len(itinerary.day1.afternoon) + len(itinerary.day1.evening) +
            len(itinerary.day2.morning) + len(itinerary.day2.afternoon) + len(itinerary.day2.evening) +
            len(itinerary.day3.morning) + len(itinerary.day3.afternoon) + len(itinerary.day3.evening)
        )
        
        # Should schedule up to 9 attractions (3 per day)
        self.assertLessEqual(total_attractions, 9)
    
    def test_calculate_itinerary_metrics(self):
        itinerary = CompleteItinerary(
            day1=DayItinerary(
                morning=[Attraction(name="AM", short_description="", approx_price_per_person=0, tags=[], reason_for_user="")],
                afternoon=[Attraction(name="PM", short_description="", approx_price_per_person=0, tags=[], reason_for_user="")],
                evening=[]
            )
        )
        
        metrics = self.agent.calculate_itinerary_metrics(itinerary)
        
        self.assertEqual(metrics["total_attractions"], 2)
        self.assertEqual(metrics["morning_attractions"], 1)
        self.assertEqual(metrics["afternoon_attractions"], 1)
        self.assertEqual(metrics["evening_attractions"], 0)

class TestEvaluationAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.generate.return_value = json.dumps({
            "interest_match": 4,
            "budget_realism": 5,
            "logistics": 4,
            "suitability_for_constraints": 5,
            "comment": "Good itinerary"
        })
        
        self.agent = EvaluationAgent(llm_client=self.mock_llm)
    
    def test_evaluate_itinerary(self):
        profile = {
            "refined_profile": "Museum lover",
            "chosen_city": "Rome",
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": 500,
                "people": 2
            }
        }
        
        itinerary = {
            "day1": {"morning": [], "afternoon": [], "evening": []},
            "day2": {"morning": [], "afternoon": [], "evening": []},
            "day3": {"morning": [], "afternoon": [], "evening": []}
        }
        
        evaluation = self.agent.evaluate_itinerary(profile, itinerary)
        
        self.assertEqual(evaluation.interest_match, 4)
        self.assertEqual(evaluation.budget_realism, 5)
        self.assertEqual(evaluation.logistics, 4)
        self.assertEqual(evaluation.suitability_for_constraints, 5)
        self.assertIn("Good", evaluation.comment)
    
    def test_calculate_overall_score(self):
        evaluation = self.agent._create_fallback_evaluation()
        overall_score = self.agent.calculate_overall_score(evaluation)
        
        # Default scores are all 3, so average should be 3.0
        self.assertEqual(overall_score, 3.0)

class TestGooglePlacesAgent(unittest.TestCase):
    def setUp(self):
        self.agent = GooglePlacesAgent(api_key="test_key")
    
    @patch('requests.get')
    def test_find_place_id(self, mock_get):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"place_id": "test_place_id"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        place_id = self.agent._find_place_id("Test Museum", "Rome")
        
        self.assertEqual(place_id, "test_place_id")
    
    def test_is_enabled(self):
        # With test key, should be enabled
        self.assertTrue(self.agent.is_enabled())

if __name__ == '__main__':
    unittest.main()