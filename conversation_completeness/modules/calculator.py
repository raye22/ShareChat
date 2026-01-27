import statistics
from typing import List, Dict
from collections import defaultdict

class ScoreCalculator:
    @staticmethod
    def calculate(verdicts: List[Dict]):
        """Calculate completeness score with three-tier system
        
        Scoring:
        - 'yes': 1.0 (fully satisfied)
        - 'partial': 0.5 (partially satisfied)
        - 'no': 0.0 (not satisfied)
        - 'skipped' or 'unknown': excluded from calculation
        
        Returns:
            float: Score between 0.0 and 1.0
            None: If no valid verdicts (all skipped/unknown)
        """
        if not verdicts:
            return None  # No intentions extracted at all
        
        total_score = 0.0
        counted_verdicts = 0
        
        for v in verdicts:
            verdict = v.get('verdict', '').lower()
            if verdict == 'yes':
                total_score += 1.0
                counted_verdicts += 1
            elif verdict == 'partial':
                total_score += 0.5
                counted_verdicts += 1
            elif verdict == 'no':
                total_score += 0.0
                counted_verdicts += 1
            # Skip 'skipped' and 'unknown' verdicts
        
        # Return None if all verdicts were skipped/unknown
        return total_score / counted_verdicts if counted_verdicts > 0 else None
    
    @staticmethod
    def aggregate_by_platform(results: List[Dict]) -> Dict:
        """Aggregate results by platform"""
        platform_scores = defaultdict(list)
        
        for result in results:
            platform = result.get('platform', 'unknown')
            score = result.get('completeness_score', 0)
            platform_scores[platform].append(score)
        
        aggregated = {}
        for platform, scores in platform_scores.items():
            aggregated[platform] = {
                'total': len(scores),
                'mean': statistics.mean(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores)
            }
        
        return aggregated
