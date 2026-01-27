import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.logger = logging.getLogger(__name__)
    
    def load(self) -> Tuple[int, Dict]:
        """Load checkpoint
        
        Returns:
            Tuple of (total_count, platform_data)
            platform_data format: {platform: {'completed': bool, 'processed_ids': [id1, id2, ...]}}
        """
        if not self.checkpoint_file.exists():
            return 0, {}
        
        try:
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            return data.get('count', 0), data.get('platforms', {})
        except:
            return 0, {}
    
    def save(self, count: int, platform_data: Dict):
        """Save checkpoint with platform-level tracking
        
        Args:
            count: Total conversations processed
            platform_data: {platform: {'completed': bool, 'processed_ids': [id1, id2, ...]}}
        """
        checkpoint = {
            'count': count,
            'platforms': platform_data
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
                f.flush()  # Force write to disk immediately
        except Exception as e:
            self.logger.error(f"✗ Checkpoint save FAILED at {count}: {e}")
            raise
    
    def save_result(self, result: Dict, results_dir: str):
        """Save individual result to platform-specific JSONL file"""
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save to platform-specific file
        platform = result.get('platform', 'unknown')
        platform_file = results_path / f"{platform}_completeness.jsonl"
        
        with open(platform_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Also save to combined file for backward compatibility
        with open(results_path / "all_platforms_completeness.jsonl", 'a') as f:
            f.write(json.dumps(result) + '\n')
