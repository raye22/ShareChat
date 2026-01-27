#!/usr/bin/env python3
import asyncio
import argparse
import json
import yaml
from pathlib import Path

from modules.logger import setup_logging, get_logger
from modules.model_wrapper import ModelWrapper
from modules.extractor import IntentExtractor
from modules.evaluator import VerdictEvaluator
from modules.calculator import ScoreCalculator
from pipeline import Pipeline

# Will be set up after parsing args
logger = None
base_dir='./data/input'

async def test_quick(data_dir: str = 'data', num_samples: int = 2, debug: bool = False):
    """Quick test with random samples from real CSV files"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info(f"QUICK TEST MODE (Sampling {num_samples} conversations from real data)")
    logger.info(f"DETAILED LOGGING ENABLED for debugging")
    logger.info("=" * 80 + "\n")
    
    try:
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        
        # Try vLLM first for speed, fall back to Transformers if it fails
        config['model']['use_vllm'] = True
        
        # Load real data from CSVs
        from modules.data_loader import DataLoader
        loader = DataLoader(data_dir, batch_size=100)
        
        # Collect samples from real data
        samples = []
        logger.info(f"Loading conversations from {data_dir}/...")
        
        for platform, batch in loader:
            for conv in batch:
                if DataLoader.validate(conv):
                    samples.append(conv)
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break
        
        if not samples:
            logger.error(f"✗ No valid conversations found in {data_dir}/")
            logger.error("Make sure your CSV files are in correct format:")
            logger.error("  - claude_results_turn_final.csv")
            logger.error("  - grok_results_turn_final.csv")
            logger.error("  - gemini_results_turn_final.csv")
            logger.error("  - perplexity_turn_final_with_languages.csv")
            logger.error("  - chatgpt_results_turn_final_grouped.csv")
            return False
        
        logger.info(f"✓ Loaded {len(samples)} sample conversations from real data\n")
        
        # Create and run pipeline
        pipeline = Pipeline(config)
        
        # Get batch_size from config
        batch_size = config['data']['batch_size']
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(samples)} test conversations...\n")
        logger.info("=" * 80)
        logger.info(f"USING OPTIMIZED BATCH PROCESSING")
        logger.info(f"  - Processing {len(samples)} conversations in {num_batches} batch(es) of {batch_size}")
        logger.info(f"  - Each batch: Step 1 (1 vLLM call) → Step 2 (1+ vLLM calls)")
        logger.info("=" * 80 + "\n")
        
        # Create test output directory
        test_output_dir = Path("output/test_results")
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log individual conversation info before processing
        for i, sample in enumerate(samples, 1):
            logger.info(f"[Conv {i}/{len(samples)}] {sample['conv_id']} ({sample['platform']}) - {sample['num_turns']} turns")
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING BATCH PROCESSING")
        logger.info("=" * 80 + "\n")
        
        # Process samples in batches respecting batch_size
        from tqdm import tqdm
        pbar = tqdm(total=len(samples), desc="Testing", unit="conv")
        
        results = []
        for batch_idx in range(0, len(samples), batch_size):
            batch = samples[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING BATCH {batch_num}/{num_batches} ({len(batch)} conversations)")
            logger.info(f"{'='*80}\n")
            
            batch_results = await pipeline.process_batch_optimized(batch, pbar)
            results.extend(batch_results)
        
        pbar.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 80 + "\n")
        
        # Collect successful results
        success_count = 0
        test_results = []
        
        for i, (sample, result) in enumerate(zip(samples, results), 1):
            if isinstance(result, Exception):
                logger.error(f"\n✗ [Test {i}/{len(samples)}] {sample['conv_id']}: {result}")
                continue
            
            if not result:
                logger.warning(f"\n✗ [Test {i}/{len(samples)}] {sample['conv_id']}: No result")
                continue
            
            success_count += 1
            test_results.append(result)
            
            logger.info("\n" + "-" * 80)
            logger.info(f"[Test {i}/{len(samples)}] {sample['conv_id']} ({sample['platform']})")
            logger.info(f"  Turns: {sample['num_turns']}")
            logger.info(f"\n  [Results]:")
            logger.info(f"    ✓ Score: {result['completeness_score']:.3f}")
            logger.info(f"    ✓ Intentions: {len(result['intentions'])} found")
            logger.info(f"      {result['intentions'][:2]}{'...' if len(result['intentions']) > 2 else ''}")
            
            # Log verdict summary
            verdict_counts = {}
            for v in result['verdicts']:
                verdict = v.get('verdict', 'unknown')
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            logger.info(f"    ✓ Verdicts: {verdict_counts}")
            
            # Save individual test result
            test_file = test_output_dir / f"test_{sample['platform']}_{sample['conv_id']}.json"
            with open(test_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"    ✓ Saved to: {test_file}")
            logger.info("-" * 80)
        
        # Save summary of all test results
        if test_results:
            summary_file = test_output_dir / "test_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'total_tested': len(samples),
                    'successful': success_count,
                    'failed': len(samples) - success_count,
                    'results': test_results
                }, f, indent=2)
            logger.info(f"\n✓ Test summary saved to: {summary_file}")
        
        logger.info("=" * 80)
        logger.info(f"✓ Test completed: {success_count}/{len(samples)} conversations processed successfully!")
        logger.info("=" * 80)
        
        if success_count == len(samples):
            logger.info("\n✓ All tests passed! Pipeline is ready for full run.")
            return True
        else:
            logger.warning(f"\n⚠ {len(samples) - success_count} conversations failed. Check logs above.")
            return False
    
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        return False

async def run_pipeline(data_dir: str = 'data', debug: bool = False):
    """Run full pipeline on CSV files"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING PIPELINE: Reading from {data_dir}/")
    if debug:
        logger.info("DEBUG MODE ENABLED - Detailed logging active")
    logger.info("=" * 80 + "\n")
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Run pipeline
    pipeline = Pipeline(config)
    await pipeline.run(data_dir)

def main():
    parser = argparse.ArgumentParser(description="Conversation Completeness Evaluation Pipeline")
    parser.add_argument('--test', action='store_true', help='Run quick test with real data samples')
    parser.add_argument('--test-samples', type=int, default=2, help='Number of samples for test (default: 2)')
    parser.add_argument('--data-dir', type=str, default='./data/input', help='Path to directory with CSV files')
    parser.add_argument('--debug', action='store_true', help='Enable detailed logging (like test mode)')
    
    args = parser.parse_args()
    
    # Setup logging with debug mode
    setup_logging(debug=args.debug or args.test)  # Auto-enable debug in test mode
    
    if args.test:
        asyncio.run(test_quick(data_dir=args.data_dir, num_samples=args.test_samples, debug=args.debug))
    else:
        asyncio.run(run_pipeline(args.data_dir, debug=args.debug))

if __name__ == "__main__":
    main()
