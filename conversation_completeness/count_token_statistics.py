#!/usr/bin/env python3
"""
Standalone script to analyze token statistics across all conversations.

This script counts how many conversations exceed various token thresholds
WITHOUT truncation, giving you the true distribution of conversation lengths.

Usage:
    python count_token_statistics.py [--data-dir DATA_DIR]

Output:
    - token_statistics.json: Detailed statistics per platform
    - token_statistics.csv: Summary table
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer

# Same file mapping as data_loader
FILE_MAPPING = {
    'claude': 'claude_results_turn_final.csv',
    'grok': 'grok_results_turn_final.csv',
    'gemini': 'gemini_results_turn_final.csv',
    'perplexity': 'perplexity_turn_final_with_languages.csv',
    'chatgpt': 'chatgpt_results_turn_final_grouped.csv',
}

# Token thresholds to check
THRESHOLDS = [8000, 16000, 32000, 64000, 131072]
data_dir='./data/input'

def load_conversations(data_dir: Path, platform: str) -> List[Dict]:
    """Load conversations from CSV for a platform"""
    filepath = data_dir / FILE_MAPPING[platform]
    if not filepath.exists():
        print(f"⚠ File not found: {filepath}")
        return []
    
    df = pd.read_csv(filepath)
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
            conv_text = "\n".join([f"{t['role']}: {t['content']}" for t in turns])
            conversations.append({
                'conv_id': str(file_name),
                'text': conv_text,
                'num_turns': len(turns)
            })
    
    return conversations

def count_tokens_chunked(text: str, tokenizer, chunk_size: int = 100000) -> int:
    """Count tokens by processing text in chunks to avoid overflow"""
    if len(text) <= chunk_size:
        try:
            # Try direct encoding without truncation
            return len(tokenizer.encode(text, add_special_tokens=False))
        except:
            # If it fails, estimate based on characters
            return len(text) // 4
    
    # Process in chunks
    total_tokens = 0
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            total_tokens += len(tokenizer.encode(chunk, add_special_tokens=False))
        except:
            total_tokens += len(chunk) // 4
    
    return total_tokens


def analyze_platform(data_dir: Path, platform: str, tokenizer) -> Dict:
    """Analyze token statistics for a single platform"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {platform.upper()}")
    print(f"{'='*60}")
    
    conversations = load_conversations(data_dir, platform)
    if not conversations:
        return None
    
    print(f"Total conversations: {len(conversations)}")
    print("Counting tokens...")
    
    stats = {
        'platform': platform,
        'total_conversations': len(conversations),
        'token_counts': [],
        'exceeds': {threshold: 0 for threshold in THRESHOLDS}
    }
    
    for conv in tqdm(conversations, desc=f"Tokenizing {platform}", unit="conv"):
        token_count = count_tokens_chunked(conv['text'], tokenizer)
        stats['token_counts'].append({
            'conv_id': conv['conv_id'],
            'token_count': token_count,
            'num_turns': conv['num_turns']
        })
        
        # Count exceeding thresholds
        for threshold in THRESHOLDS:
            if token_count > threshold:
                stats['exceeds'][threshold] += 1
    
    # Calculate statistics
    all_counts = [c['token_count'] for c in stats['token_counts']]
    stats['min_tokens'] = min(all_counts)
    stats['max_tokens'] = max(all_counts)
    stats['mean_tokens'] = sum(all_counts) / len(all_counts)
    stats['median_tokens'] = sorted(all_counts)[len(all_counts) // 2]
    
    # Print summary
    print(f"\nToken Statistics:")
    print(f"  Min: {stats['min_tokens']:,}")
    print(f"  Max: {stats['max_tokens']:,}")
    print(f"  Mean: {stats['mean_tokens']:,.1f}")
    print(f"  Median: {stats['median_tokens']:,}")
    print(f"\nExceeding thresholds:")
    for threshold in THRESHOLDS:
        count = stats['exceeds'][threshold]
        percentage = (count / len(conversations)) * 100
        print(f"  > {threshold:,} tokens: {count} ({percentage:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Count token statistics for conversations")
    parser.add_argument('--data-dir', default='./data/input', help="Directory containing CSV files")
    parser.add_argument('--output-dir', default='results', help="Directory for output files")
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help="Model name for tokenizer")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Analyze each platform
    all_stats = []
    for platform in FILE_MAPPING.keys():
        stats = analyze_platform(data_dir, platform, tokenizer)
        if stats:
            all_stats.append(stats)
    
    # Save detailed results
    output_json = output_dir / "token_statistics.json"
    with open(output_json, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Detailed statistics saved to: {output_json}")
    
    # Create summary table
    summary_data = []
    for stats in all_stats:
        row = {
            'Platform': stats['platform'],
            'Total Conversations': stats['total_conversations'],
            'Min Tokens': stats['min_tokens'],
            'Max Tokens': stats['max_tokens'],
            'Mean Tokens': f"{stats['mean_tokens']:.1f}",
            'Median Tokens': stats['median_tokens'],
        }
        for threshold in THRESHOLDS:
            count = stats['exceeds'][threshold]
            pct = (count / stats['total_conversations']) * 100
            row[f'> {threshold//1000}k'] = f"{count} ({pct:.1f}%)"
        summary_data.append(row)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    output_csv = output_dir / "token_statistics.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"✓ Summary table saved to: {output_csv}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    # Overall statistics
    total_convs = sum(s['total_conversations'] for s in all_stats)
    print(f"\n{'='*80}")
    print(f"OVERALL: {total_convs:,} total conversations across {len(all_stats)} platforms")
    for threshold in THRESHOLDS:
        total_exceeds = sum(s['exceeds'][threshold] for s in all_stats)
        pct = (total_exceeds / total_convs) * 100
        print(f"  > {threshold:,} tokens: {total_exceeds} ({pct:.1f}%)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
