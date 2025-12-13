"""
Evaluation module for Travel Itinerary Planner.
"""

from .evaluator import ProjectEvaluator, run_evaluation_for_team
from .metrics import calculate_metrics, generate_report
from .test_cases import get_test_cases, TestCase

__all__ = [
    'ProjectEvaluator',
    'run_evaluation_for_team',
    'calculate_metrics',
    'generate_report',
    'get_test_cases',
    'TestCase'
]
