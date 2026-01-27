import pandas as pd
from pathlib import Path
from typing import Iterator, List, Dict
from modules.logger import get_logger

class DataLoader:
    """Load conversations from CSV files for 5 platforms"""
    FILE_MAPPING = {
        'claude': 'claude_results_turn_final.csv',
        'grok': 'grok_results_turn_final.csv',
        'gemini': 'gemini_results_turn_final.csv',
        'perplexity': 'perplexity_turn_final_with_languages.csv',
        'chatgpt': 'chatgpt_results_turn_final_grouped.csv',
    }
    
    def __init__(self, data_dir: str = 'data', batch_size: int = 16, platform_data: Dict = None):
        """Initialize data loader"""
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.platform_data = platform_data or {}
        self.logger = get_logger()
    
    def load_platform_data(self, platform: str) -> pd.DataFrame:
        """Load CSV for a specific platform"""
        filepath = self.data_dir / self.FILE_MAPPING[platform]
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        required_cols = ['file_name', 'role', 'turn_index', 'plain_text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    
    def convert_to_conversations(self, df: pd.DataFrame, platform: str) -> List[Dict]:
        """Convert CSV dataframe to conversation format"""
        conversations = []
        grouped = df.groupby('file_name', sort=False)
        
        for file_name, group in grouped:
            group = group.sort_values('turn_index')
            turns = []
            for _, row in group.iterrows():
                turns.append({
                    'role': row['role'].strip().lower(),
                    'content': str(row['plain_text']).strip()
                })
            
            if len(turns) >= 2:
                conv = {
                    'conv_id': str(file_name),
                    'platform': platform,
                    'turns': turns,
                    'num_turns': len(turns)
                }
                conversations.append(conv)
        
        return conversations
    
    def __iter__(self) -> Iterator[tuple[str, List[Dict]]]:
        """Yield (platform, batches) tuples, processing one platform at a time"""
        for platform in self.FILE_MAPPING.keys():
            # Check if platform already completed (check ONCE before loading data)
            if platform in self.platform_data and self.platform_data[platform].get('completed', False):
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Platform: {platform.upper()} - ALREADY COMPLETED, skipping")
                self.logger.info(f"{'='*80}")
                continue  # Skip to next platform without loading any data
            
            try:
                df = self.load_platform_data(platform)
                conversations = self.convert_to_conversations(df, platform)
                print(f"✓ Loaded {platform}: {len(conversations)} conversations")
                
                # Yield platform and all its conversations in batches
                batch = []
                for conv in conversations:
                    batch.append(conv)
                    if len(batch) == self.batch_size:
                        yield (platform, batch)
                        batch = []
                
                # Yield remaining conversations for this platform
                if batch:
                    yield (platform, batch)
                    
            except FileNotFoundError as e:
                print(f"⚠ Skipping {platform}: {e}")
                continue
    
    @staticmethod
    def validate(conv: Dict) -> bool:
        """Check if conversation has required fields"""
        required = ['conv_id', 'platform', 'turns']
        if not all(field in conv for field in required):
            return False
        if not conv['turns'] or len(conv['turns']) < 2:
            return False
        for turn in conv['turns']:
            if 'role' not in turn or 'content' not in turn:
                return False
        return True
