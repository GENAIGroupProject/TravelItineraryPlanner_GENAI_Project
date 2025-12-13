import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from config import Config

@dataclass
class EvaluationCriteria:
    """Evaluation criteria for different aspects."""
    performance: List[str] = None
    relevance: List[str] = None
    completeness: List[str] = None
    usability: List[str] = None
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = [
                "Response time",
                "System efficiency",
                "Resource usage",
                "Processing speed"
            ]
        if self.relevance is None:
            self.relevance = [
                "City recommendation relevance",
                "Attraction relevance to interests",
                "Budget appropriateness",
                "Constraint handling"
            ]
        if self.completeness is None:
            self.completeness = [
                "Information completeness",
                "Detail level",
                "Coverage of preferences",
                "Comprehensive planning"
            ]
        if self.usability is None:
            self.usability = [
                "Ease of use",
                "Dialogue naturalness",
                "Clarity of questions",
                "Result presentation"
            ]

class ProjectEvaluator:
    """Evaluator for assessing project performance."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
        self.criteria = EvaluationCriteria()
    
    def evaluate_agent_performance(self, agent_name: str, 
                                  test_cases: List[Dict]) -> Dict:
        """Evaluate specific agent performance."""
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_name}")
        print(f"{'='*60}")
        
        results = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(test_cases),
            "performance_metrics": [],
            "success_rate": 0,
            "average_response_time": 0,
            "issues_found": []
        }
        
        total_time = 0
        successful = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}/{len(test_cases)}")
            
            try:
                start_time = time.time()
                
                # This would call the actual agent based on agent_name
                # For now, we'll simulate with prompts
                prompt = self._create_evaluation_prompt(agent_name, test_case)
                response = self.llm_client.generate(prompt)
                
                end_time = time.time()
                response_time = end_time - start_time
                total_time += response_time
                
                # Parse and validate response
                parsed = self.json_parser.parse_response(response)
                is_valid = self._validate_response(agent_name, parsed)
                
                if is_valid:
                    successful += 1
                    print(f"✅ Success - Response time: {response_time:.2f}s")
                else:
                    results["issues_found"].append({
                        "test_case": i,
                        "issue": "Invalid response format",
                        "response": parsed
                    })
                    print(f"❌ Failed - Invalid response")
                    
            except Exception as e:
                results["issues_found"].append({
                    "test_case": i,
                    "issue": str(e),
                    "response": None
                })
                print(f"❌ Error: {e}")
        
        # Calculate metrics
        results["success_rate"] = (successful / len(test_cases)) * 100
        results["average_response_time"] = total_time / len(test_cases) if test_cases else 0
        
        return results
    
    def _create_evaluation_prompt(self, agent_name: str, test_case: Dict) -> str:
        """Create evaluation prompt for different agents."""
        
        prompts = {
            "interest_refinement": f"""
You are testing the Interest Refinement Agent.

TEST INPUT:
{json.dumps(test_case, indent=2)}

Evaluate if the agent should:
1. Ask a relevant question
2. Finalize with a city recommendation
3. Handle the input appropriately

Return ONLY JSON:
{{
  "expected_action": "ask_question" or "finalize",
  "expected_question": "string if asking",
  "expected_city": "string if finalizing",
  "reasoning": "brief explanation"
}}
""",
            "location_scout": f"""
You are testing the Location Scout Agent.

TEST INPUT:
City: {test_case.get('city', 'Unknown')}
Interests: {test_case.get('interests', 'Not specified')}

Evaluate if the agent would generate appropriate attractions.

Return ONLY JSON:
{{
  "expected_attraction_count": "number between 8-12",
  "expected_themes": ["list", "of", "expected", "themes"],
  "budget_appropriateness": "low/medium/high",
  "reasoning": "brief explanation"
}}
""",
            "scheduler": f"""
You are testing the Scheduler Agent.

TEST INPUT:
Attractions: {len(test_case.get('attractions', []))}
Days: {test_case.get('days', 3)}

Evaluate if the scheduling would be appropriate.

Return ONLY JSON:
{{
  "distribution_quality": "poor/fair/good/excellent",
  "daily_balance": "poor/fair/good/excellent",
  "time_utilization": "poor/fair/good/excellent",
  "reasoning": "brief explanation"
}}
"""
        }
        
        return prompts.get(agent_name, "")
    
    def _validate_response(self, agent_name: str, response: Dict) -> bool:
        """Validate agent response."""
        validation_rules = {
            "interest_refinement": lambda r: all(
                k in r for k in ["expected_action", "expected_question", 
                                "expected_city", "reasoning"]
            ),
            "location_scout": lambda r: all(
                k in r for k in ["expected_attraction_count", "expected_themes",
                                "budget_appropriateness", "reasoning"]
            ),
            "scheduler": lambda r: all(
                k in r for k in ["distribution_quality", "daily_balance",
                                "time_utilization", "reasoning"]
            )
        }
        
        validator = validation_rules.get(agent_name)
        return validator(response) if validator else False
    
    def run_comprehensive_evaluation(self, evaluators: List[str] = None) -> Dict:
        """Run comprehensive evaluation for multiple evaluators."""
        if evaluators is None:
            evaluators = ["evaluator1", "evaluator2", "evaluator3"]
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PROJECT EVALUATION")
        print("="*60)
        
        results = {
            "project": "Travel Itinerary Planner",
            "evaluation_date": datetime.now().isoformat(),
            "evaluators": evaluators,
            "agent_evaluations": {},
            "overall_scores": {},
            "recommendations": []
        }
        
        # Define test cases for each agent
        test_cases = self._get_test_cases()
        
        # Evaluate each agent
        for agent_name in ["interest_refinement", "location_scout", 
                          "budget", "scheduler", "evaluation"]:
            if agent_name in test_cases:
                agent_results = self.evaluate_agent_performance(
                    agent_name, test_cases[agent_name]
                )
                results["agent_evaluations"][agent_name] = agent_results
        
        # Calculate overall scores
        results["overall_scores"] = self._calculate_overall_scores(
            results["agent_evaluations"]
        )
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(
            results["agent_evaluations"]
        )
        
        return results
    
    def _get_test_cases(self) -> Dict[str, List[Dict]]:
        """Get test cases for each agent."""
        return {
            "interest_refinement": [
                {
                    "user_input": "I love museums and historical sites",
                    "budget": 500,
                    "people": 2,
                    "days": 3
                },
                {
                    "user_input": "I want to see ancient buildings and eat local food",
                    "budget": 800,
                    "people": 1,
                    "days": 4
                }
            ],
            "location_scout": [
                {
                    "city": "Rome",
                    "interests": "ancient buildings, museums",
                    "budget": 600,
                    "people": 2
                },
                {
                    "city": "Paris",
                    "interests": "art, food, architecture",
                    "budget": 1000,
                    "people": 4
                }
            ],
            "scheduler": [
                {
                    "attractions": [{"name": f"Attraction {i}"} for i in range(10)],
                    "days": 3
                }
            ]
        }
    
    def _calculate_overall_scores(self, agent_results: Dict) -> Dict:
        """Calculate overall project scores."""
        total_success = 0
        total_tests = 0
        total_response_time = 0
        
        for agent, results in agent_results.items():
            total_success += results.get("success_rate", 0) * results.get("test_cases", 0)
            total_tests += results.get("test_cases", 0)
            total_response_time += results.get("average_response_time", 0)
        
        avg_success = total_success / total_tests if total_tests > 0 else 0
        avg_response_time = total_response_time / len(agent_results) if agent_results else 0
        
        return {
            "overall_success_rate": avg_success,
            "average_response_time": avg_response_time,
            "total_test_cases": total_tests,
            "performance_grade": self._assign_grade(avg_success)
        }
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score."""
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
    
    def _generate_recommendations(self, agent_results: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for agent, results in agent_results.items():
            success_rate = results.get("success_rate", 0)
            
            if success_rate < 80:
                recommendations.append(
                    f"Improve {agent} agent: Success rate is {success_rate:.1f}%"
                )
            
            issues = results.get("issues_found", [])
            if issues:
                recommendations.append(
                    f"Fix {len(issues)} issues in {agent} agent"
                )
        
        if not recommendations:
            recommendations.append("All agents performing well. Maintain current standards.")
        
        return recommendations
    
    def save_evaluation_report(self, results: Dict, filename: str = None):
        """Save evaluation report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Evaluation report saved to: {filename}")
        return filename

def run_evaluation_for_team():
    """Run evaluation for 3-person team with different focuses."""
    
    evaluator = ProjectEvaluator()
    
    # Define evaluation focuses for 3 team members
    team_focuses = {
        "Team Member 1 (Performance)": {
            "focus": "performance",
            "test_cases": "stress_tests",
            "metrics": ["response_time", "throughput", "resource_usage"]
        },
        "Team Member 2 (Relevance)": {
            "focus": "relevance",
            "test_cases": "relevance_tests",
            "metrics": ["city_match", "attraction_relevance", "budget_appropriateness"]
        },
        "Team Member 3 (Usability)": {
            "focus": "usability",
            "test_cases": "user_experience_tests",
            "metrics": ["dialogue_quality", "result_clarity", "ease_of_use"]
        }
    }
    
    print("="*70)
    print("3-PERSON TEAM EVALUATION - TRAVEL ITINERARY PLANNER")
    print("="*70)
    
    all_results = {}
    
    for member_name, focus_info in team_focuses.items():
        print(f"\n\n{member_name}")
        print(f"Focus: {focus_info['focus'].upper()}")
        print("-" * 50)
        
        # Run evaluation with specific focus
        results = evaluator.run_comprehensive_evaluation()
        
        # Add focus-specific analysis
        results["evaluation_focus"] = focus_info
        
        all_results[member_name] = results
        
        # Save individual report
        filename = f"{member_name.replace(' ', '_').replace('(', '').replace(')', '')}_report.json"
        evaluator.save_evaluation_report(results, filename)
    
    # Create consolidated team report
    consolidated = {
        "project": "Travel Itinerary Planner",
        "evaluation_date": datetime.now().isoformat(),
        "team_evaluations": all_results,
        "consolidated_findings": consolidate_team_findings(all_results),
        "final_recommendations": generate_final_recommendations(all_results)
    }
    
    evaluator.save_evaluation_report(consolidated, "team_consolidated_report.json")
    
    return consolidated

def consolidate_team_findings(team_results: Dict) -> Dict:
    """Consolidate findings from all team members."""
    consolidated = {
        "overall_performance_score": 0,
        "key_strengths": [],
        "key_weaknesses": [],
        "consensus_issues": []
    }
    
    performance_scores = []
    strengths = set()
    weaknesses = set()
    
    for member, results in team_results.items():
        perf_score = results.get("overall_scores", {}).get("overall_success_rate", 0)
        performance_scores.append(perf_score)
        
        # Collect issues
        for agent_name, agent_results in results.get("agent_evaluations", {}).items():
            for issue in agent_results.get("issues_found", []):
                issue_desc = f"{agent_name}: {issue.get('issue', 'Unknown')}"
                weaknesses.add(issue_desc)
    
    if performance_scores:
        consolidated["overall_performance_score"] = sum(performance_scores) / len(performance_scores)
    
    # Convert sets to lists
    consolidated["key_weaknesses"] = list(weaknesses)[:10]  # Top 10 weaknesses
    
    # Identify strengths (agents with >90% success)
    for member, results in team_results.items():
        for agent_name, agent_results in results.get("agent_evaluations", {}).items():
            if agent_results.get("success_rate", 0) > 90:
                strengths.add(f"{agent_name} agent (high reliability)")
    
    consolidated["key_strengths"] = list(strengths)
    
    return consolidated

def generate_final_recommendations(team_results: Dict) -> List[str]:
    """Generate final recommendations from team evaluation."""
    recommendations = [
        "Based on 3-person team evaluation:",
        "="*50
    ]
    
    # Common recommendations
    recommendations.append("\n1. PRIORITY FIXES:")
    
    # Collect common issues
    all_issues = {}
    for member, results in team_results.items():
        for agent_name, agent_results in results.get("agent_evaluations", {}).items():
            for issue in agent_results.get("issues_found", []):
                key = f"{agent_name}: {issue.get('issue', 'Unknown')}"
                all_issues[key] = all_issues.get(key, 0) + 1
    
    # Sort by frequency
    sorted_issues = sorted(all_issues.items(), key=lambda x: x[1], reverse=True)
    
    for i, (issue, count) in enumerate(sorted_issues[:5], 1):
        recommendations.append(f"   {i}. {issue} (reported by {count} evaluators)")
    
    recommendations.append("\n2. PERFORMANCE IMPROVEMENTS:")
    recommendations.append("   • Optimize LLM response times")
    recommendations.append("   • Implement response caching")
    recommendations.append("   • Add parallel processing for Google Places API")
    
    recommendations.append("\n3. ENHANCEMENTS:")
    recommendations.append("   • Add weather considerations")
    recommendations.append("   • Include transportation planning")
    recommendations.append("   • Add restaurant recommendations")
    
    return recommendations

if __name__ == "__main__":
    # Run team evaluation
    print("Starting 3-person team evaluation...")
    final_report = run_evaluation_for_team()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    print(f"\nOverall Performance Score: {final_report['consolidated_findings']['overall_performance_score']:.1f}%")
    
    print("\nKey Strengths:")
    for strength in final_report['consolidated_findings']['key_strengths'][:3]:
        print(f"  • {strength}")
    
    print("\nKey Weaknesses:")
    for weakness in final_report['consolidated_findings']['key_weaknesses'][:3]:
        print(f"  • {weakness}")
    
    print("\nFinal Recommendations:")
    for rec in final_report['final_recommendations']:
        print(rec)
