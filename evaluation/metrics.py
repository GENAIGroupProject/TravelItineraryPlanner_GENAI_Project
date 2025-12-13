from typing import Dict, List, Any, Tuple
import statistics
from datetime import datetime

def calculate_metrics(test_results: List[Dict]) -> Dict[str, Any]:
    """Calculate metrics from test results."""
    if not test_results:
        return {}
    
    # Extract performance metrics
    response_times = [r.get("response_time", 0) for r in test_results if "response_time" in r]
    success_rates = [r.get("success", 0) for r in test_results if "success" in r]
    
    metrics = {
        "total_tests": len(test_results),
        "successful_tests": sum(1 for r in test_results if r.get("success", False)),
        "failed_tests": sum(1 for r in test_results if not r.get("success", True)),
        
        "response_time": {
            "average": statistics.mean(response_times) if response_times else 0,
            "median": statistics.median(response_times) if response_times else 0,
            "min": min(response_times) if response_times else 0,
            "max": max(response_times) if response_times else 0,
            "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
        } if response_times else {},
        
        "success_rate": {
            "average": statistics.mean(success_rates) if success_rates else 0,
            "min": min(success_rates) if success_rates else 0,
            "max": max(success_rates) if success_rates else 0
        } if success_rates else {},
        
        "quality_metrics": calculate_quality_metrics(test_results)
    }
    
    # Calculate overall score
    metrics["overall_score"] = calculate_overall_score(metrics)
    
    return metrics

def calculate_quality_metrics(test_results: List[Dict]) -> Dict[str, float]:
    """Calculate quality metrics."""
    quality_metrics = {
        "relevance_score": 0,
        "completeness_score": 0,
        "accuracy_score": 0,
        "consistency_score": 0
    }
    
    relevance_scores = []
    completeness_scores = []
    accuracy_scores = []
    consistency_scores = []
    
    for result in test_results:
        if "relevance_score" in result:
            relevance_scores.append(result["relevance_score"])
        if "completeness_score" in result:
            completeness_scores.append(result["completeness_score"])
        if "accuracy_score" in result:
            accuracy_scores.append(result["accuracy_score"])
        if "consistency_score" in result:
            consistency_scores.append(result["consistency_score"])
    
    if relevance_scores:
        quality_metrics["relevance_score"] = statistics.mean(relevance_scores)
    if completeness_scores:
        quality_metrics["completeness_score"] = statistics.mean(completeness_scores)
    if accuracy_scores:
        quality_metrics["accuracy_score"] = statistics.mean(accuracy_scores)
    if consistency_scores:
        quality_metrics["consistency_score"] = statistics.mean(consistency_scores)
    
    return quality_metrics

def calculate_overall_score(metrics: Dict) -> float:
    """Calculate overall score from metrics."""
    weights = {
        "success_rate": 0.4,
        "response_time": 0.3,
        "quality": 0.3
    }
    
    # Success rate component (0-100)
    success_rate = metrics.get("success_rate", {}).get("average", 0) * 100
    
    # Response time component (inverted, lower is better)
    avg_response_time = metrics.get("response_time", {}).get("average", 0)
    # Assume ideal response time is 5 seconds, worse as it increases
    response_time_score = max(0, 100 - (avg_response_time * 10))
    
    # Quality component (average of quality metrics)
    quality_metrics = metrics.get("quality_metrics", {})
    quality_scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
    quality_score = statistics.mean(quality_scores) * 20 if quality_scores else 50  # Convert 0-5 to 0-100
    
    # Calculate weighted sum
    overall_score = (
        success_rate * weights["success_rate"] +
        response_time_score * weights["response_time"] +
        quality_score * weights["quality"]
    )
    
    return min(100, max(0, overall_score))

def generate_report(metrics: Dict, test_results: List[Dict]) -> Dict:
    """Generate comprehensive evaluation report."""
    report = {
        "report_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "summary": generate_summary(metrics),
        "detailed_metrics": metrics,
        "test_results_summary": summarize_test_results(test_results),
        "recommendations": generate_recommendations(metrics),
        "grades": assign_grades(metrics)
    }
    
    return report

def generate_summary(metrics: Dict) -> Dict:
    """Generate executive summary."""
    overall_score = metrics.get("overall_score", 0)
    
    summary = {
        "overall_performance": overall_score,
        "performance_level": get_performance_level(overall_score),
        "key_strengths": identify_strengths(metrics),
        "key_weaknesses": identify_weaknesses(metrics),
        "verdict": get_verdict(overall_score)
    }
    
    return summary

def get_performance_level(score: float) -> str:
    """Get performance level based on score."""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Satisfactory"
    elif score >= 60:
        return "Needs Improvement"
    else:
        return "Poor"

def identify_strengths(metrics: Dict) -> List[str]:
    """Identify system strengths."""
    strengths = []
    
    success_rate = metrics.get("success_rate", {}).get("average", 0)
    if success_rate >= 0.9:
        strengths.append("High success rate in operations")
    
    response_time = metrics.get("response_time", {}).get("average", 0)
    if response_time < 10:
        strengths.append("Fast response times")
    
    quality = metrics.get("quality_metrics", {})
    if quality.get("relevance_score", 0) >= 4:
        strengths.append("Highly relevant recommendations")
    
    return strengths

def identify_weaknesses(metrics: Dict) -> List[str]:
    """Identify system weaknesses."""
    weaknesses = []
    
    success_rate = metrics.get("success_rate", {}).get("average", 0)
    if success_rate < 0.7:
        weaknesses.append("Low success rate needs improvement")
    
    response_time = metrics.get("response_time", {}).get("average", 0)
    if response_time > 30:
        weaknesses.append("Slow response times affecting user experience")
    
    quality = metrics.get("quality_metrics", {})
    if quality.get("accuracy_score", 0) < 3:
        weaknesses.append("Accuracy could be improved")
    
    return weaknesses

def get_verdict(score: float) -> str:
    """Get final verdict."""
    if score >= 80:
        return "System is performing well and ready for production use."
    elif score >= 60:
        return "System is functional but needs some improvements."
    else:
        return "System needs significant improvements before production use."

def summarize_test_results(test_results: List[Dict]) -> Dict:
    """Summarize test results."""
    categories = {}
    
    for result in test_results:
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = {"total": 0, "passed": 0, "failed": 0}
        
        categories[category]["total"] += 1
        if result.get("success", False):
            categories[category]["passed"] += 1
        else:
            categories[category]["failed"] += 1
    
    # Calculate pass rates
    for category in categories:
        total = categories[category]["total"]
        passed = categories[category]["passed"]
        categories[category]["pass_rate"] = (passed / total * 100) if total > 0 else 0
    
    return categories

def generate_recommendations(metrics: Dict) -> List[str]:
    """Generate improvement recommendations."""
    recommendations = []
    
    # Performance recommendations
    response_time = metrics.get("response_time", {}).get("average", 0)
    if response_time > 20:
        recommendations.append("Optimize LLM response times through caching or model optimization")
    
    # Success rate recommendations
    success_rate = metrics.get("success_rate", {}).get("average", 0)
    if success_rate < 0.8:
        recommendations.append("Improve error handling and fallback mechanisms")
    
    # Quality recommendations
    quality = metrics.get("quality_metrics", {})
    if quality.get("relevance_score", 0) < 3:
        recommendations.append("Enhance semantic understanding for better relevance")
    
    if quality.get("completeness_score", 0) < 3:
        recommendations.append("Improve completeness of generated itineraries")
    
    return recommendations

def assign_grades(metrics: Dict) -> Dict[str, str]:
    """Assign letter grades to different aspects."""
    grades = {}
    
    # Success rate grade
    success_rate = metrics.get("success_rate", {}).get("average", 0) * 100
    grades["success_rate"] = _assign_letter_grade(success_rate)
    
    # Response time grade (inverted)
    response_time = metrics.get("response_time", {}).get("average", 0)
    response_time_score = max(0, 100 - (response_time * 2))  # 50 seconds = 0, 0 seconds = 100
    grades["response_time"] = _assign_letter_grade(response_time_score)
    
    # Quality grade
    quality = metrics.get("quality_metrics", {})
    quality_scores = [v for v in quality.values() if isinstance(v, (int, float))]
    quality_score = statistics.mean(quality_scores) * 20 if quality_scores else 50
    grades["quality"] = _assign_letter_grade(quality_score)
    
    # Overall grade
    overall_score = metrics.get("overall_score", 0)
    grades["overall"] = _assign_letter_grade(overall_score)
    
    return grades

def _assign_letter_grade(score: float) -> str:
    """Assign letter grade A-F."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
