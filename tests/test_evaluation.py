import unittest
import json
from unittest.mock import Mock, patch

from evaluation.evaluator import ProjectEvaluator, EvaluationCriteria
from evaluation.metrics import calculate_metrics, generate_report
from evaluation.test_cases import get_test_cases, TestCase

class TestProjectEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.generate.return_value = json.dumps({
            "expected_action": "finalize",
            "expected_question": "",
            "expected_city": "Rome",
            "reasoning": "User clearly wants museums"
        })
        
        self.evaluator = ProjectEvaluator(llm_client=self.mock_llm)
    
    def test_evaluation_criteria(self):
        criteria = EvaluationCriteria()
        
        self.assertGreater(len(criteria.performance), 0)
        self.assertGreater(len(criteria.relevance), 0)
        self.assertGreater(len(criteria.completeness), 0)
        self.assertGreater(len(criteria.usability), 0)
    
    def test_validate_response(self):
        # Test valid response
        valid_response = {
            "expected_action": "finalize",
            "expected_question": "",
            "expected_city": "Rome",
            "reasoning": "Test"
        }
        
        is_valid = self.evaluator._validate_response("interest_refinement", valid_response)
        self.assertTrue(is_valid)
        
        # Test invalid response (missing field)
        invalid_response = {
            "expected_action": "finalize",
            "expected_city": "Rome"
            # Missing reasoning and expected_question
        }
        
        is_valid = self.evaluator._validate_response("interest_refinement", invalid_response)
        self.assertFalse(is_valid)
    
    def test_assign_grade(self):
        self.assertEqual(self.evaluator._assign_grade(95), "A")
        self.assertEqual(self.evaluator._assign_grade(85), "B")
        self.assertEqual(self.evaluator._assign_grade(75), "C")
        self.assertEqual(self.evaluator._assign_grade(65), "D")
        self.assertEqual(self.evaluator._assign_grade(55), "F")

class TestMetrics(unittest.TestCase):
    def test_calculate_metrics(self):
        test_results = [
            {"response_time": 10.5, "success": True, "relevance_score": 4},
            {"response_time": 8.2, "success": True, "relevance_score": 5},
            {"response_time": 15.0, "success": False, "relevance_score": 2}
        ]
        
        metrics = calculate_metrics(test_results)
        
        self.assertEqual(metrics["total_tests"], 3)
        self.assertEqual(metrics["successful_tests"], 2)
        self.assertEqual(metrics["failed_tests"], 1)
        
        # Check response time metrics
        self.assertAlmostEqual(metrics["response_time"]["average"], 11.233, places=2)
        self.assertEqual(metrics["response_time"]["min"], 8.2)
        self.assertEqual(metrics["response_time"]["max"], 15.0)
    
    def test_calculate_overall_score(self):
        metrics = {
            "success_rate": {"average": 0.8},
            "response_time": {"average": 10.0},
            "quality_metrics": {
                "relevance_score": 4.0,
                "completeness_score": 3.5,
                "accuracy_score": 4.2,
                "consistency_score": 3.8
            }
        }
        
        overall_score = calculate_metrics.calculate_overall_score(metrics)
        
        # Score should be between 0-100
        self.assertGreaterEqual(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
    
    def test_assign_grades(self):
        metrics = {
            "success_rate": {"average": 0.85},
            "response_time": {"average": 15.0},
            "overall_score": 78.5
        }
        
        grades = generate_report.assign_grades(metrics)
        
        self.assertIn("overall", grades)
        self.assertIn(grades["overall"], ["A", "B", "C", "D", "F"])

class TestTestCases(unittest.TestCase):
    def test_get_test_cases(self):
        test_cases = get_test_cases()
        
        self.assertGreater(len(test_cases), 0)
        
        # Check that all test cases have required fields
        for test_case in test_cases:
            self.assertIsInstance(test_case, TestCase)
            self.assertIsInstance(test_case.name, str)
            self.assertIsInstance(test_case.description, str)
            self.assertIsInstance(test_case.input_data, dict)
            self.assertIsInstance(test_case.expected_output, dict)
            self.assertIsInstance(test_case.category, str)
    
    def test_get_test_cases_by_category(self):
        relevance_cases = get_test_cases("relevance")
        
        self.assertGreater(len(relevance_cases), 0)
        
        # All should be relevance category
        for case in relevance_cases:
            self.assertEqual(case.category, "relevance")
    
    def test_get_stress_test_cases(self):
        from evaluation.test_cases import get_stress_test_cases
        
        stress_cases = get_stress_test_cases()
        
        self.assertGreater(len(stress_cases), 0)
        
        for case in stress_cases:
            self.assertEqual(case.category, "performance")
    
    def test_get_usability_test_cases(self):
        from evaluation.test_cases import get_usability_test_cases
        
        usability_cases = get_usability_test_cases()
        
        self.assertGreater(len(usability_cases), 0)
        
        for case in usability_cases:
            self.assertEqual(case.category, "usability")

if __name__ == '__main__':
    unittest.main()