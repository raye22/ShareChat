import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

from modules.logger import setup_logging, get_logger
from modules.data_loader import DataLoader
from modules.model_wrapper import ModelWrapper
from modules.extractor import IntentExtractor
from modules.evaluator import VerdictEvaluator
from modules.calculator import ScoreCalculator
from modules.checkpoint import CheckpointManager

class Pipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger()
        
        # Log model initialization
        model_backend = "vLLM" if config['model']['use_vllm'] else "Transformers"
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"INITIALIZING MODEL: {config['model']['name']}")
        self.logger.info(f"Backend: {model_backend}")
        self.logger.info(f"{'='*80}")
        
        self.model = ModelWrapper(
            config['model']['name'],
            use_vllm=config['model']['use_vllm']
        )
        
        # Log which backend actually loaded
        actual_backend = "vLLM" if self.model.use_vllm else "Transformers (fallback)"
        self.logger.info(f"✓ Model ready: {actual_backend}")
        self.logger.info(f"{'='*80}\n")
        
        # Pass config parameters for truncation
        max_llm_tokens = config.get('processing', {}).get('llm_response_truncate_tokens', 1024)
        min_content_ratio = config.get('processing', {}).get('min_content_ratio', 0.8)
        include_reason = config.get('processing', {}).get('include_reason', True)
        
        self.extractor = IntentExtractor(self.model, max_llm_tokens=max_llm_tokens)
        self.evaluator = VerdictEvaluator(self.model, min_content_ratio=min_content_ratio, include_reason=include_reason)
        self.calculator = ScoreCalculator()
        self.checkpoint = CheckpointManager(config['checkpoint']['output_dir'])
    
    async def process_batch_optimized(self, conversations: List[Dict], pbar=None) -> List:
        """Process batch of conversations with optimized vLLM batching (2 calls total instead of 2*N)"""
        try:
            # Get max_tokens from config
            max_tokens_step1 = self.config.get('model', {}).get('max_tokens_step1', 256)
            max_tokens_step2 = self.config.get('model', {}).get('max_tokens_step2', 128)
            
            # Step 1: Extract ALL intentions in ONE vLLM call
            self.logger.info("=" * 80)
            self.logger.info(f"[BATCH STEP 1] EXTRACTING INTENTIONS")
            self.logger.info("=" * 80)
            self.logger.info(f"Processing {len(conversations)} conversations in 1 vLLM call (max_tokens={max_tokens_step1})...")
            all_extraction_results = await self.extractor.extract_batch(conversations, max_tokens=max_tokens_step1)
            
            # Count successful extractions and skipped
            successful_extractions = sum(1 for result in all_extraction_results if len(result) >= 3 and result[2] == 'success')
            skipped_extractions = sum(1 for result in all_extraction_results if len(result) >= 3 and result[2] == 'skipped')
            total_intentions = sum(len(result[0]) for result in all_extraction_results if len(result) >= 3 and result[2] == 'success')
            self.logger.info(f"✓ Step 1 Complete: {successful_extractions}/{len(conversations)} conversations successful, {skipped_extractions} skipped, {total_intentions} total intentions extracted\n")
            
            # Step 2: Collect ALL evaluation prompts across ALL conversations
            self.logger.info("=" * 80)
            self.logger.info(f"[BATCH STEP 2] EVALUATING VERDICTS")
            self.logger.info("=" * 80)
            self.logger.info(f"Preparing evaluation prompts for all conversations...")
            all_eval_prompts = []
            eval_metadata = []  # Track which prompts belong to which conversation
            
            for conv_idx, (conv, extraction_result) in enumerate(zip(conversations, all_extraction_results)):
                # Unpack extraction result (4-tuple: intentions, is_valid, status, error_reason)
                intentions = extraction_result[0]
                is_valid = extraction_result[1] if len(extraction_result) > 1 else False
                status = extraction_result[2] if len(extraction_result) > 2 else 'unknown'
                error_reason = extraction_result[3] if len(extraction_result) > 3 else None
                
                if not is_valid or not intentions:
                    eval_metadata.append({
                        'conv_idx': conv_idx, 
                        'has_prompts': False, 
                        'intentions': [], 
                        'conv': conv,
                        'status': status,
                        'error_reason': error_reason
                    })
                    continue
                
                # Build prompts for this conversation's intentions
                conv_prompts = []
                conv_intentions = []
                skip_whole_conversation = False
                skip_reason = None
                
                for intent in intentions:
                    turns_text, content_ratio = self.evaluator._build_turns_text(conv['turns'], intent)
                    
                    # Check if truncation is too severe (>20% content loss)
                    if content_ratio < self.evaluator.min_content_ratio:
                        # SKIP ENTIRE CONVERSATION if ANY intention requires severe truncation
                        skip_whole_conversation = True
                        skip_reason = f"Severe truncation required in Step 2 (content_ratio: {content_ratio:.1%} < {self.evaluator.min_content_ratio:.0%})"
                        self.logger.warning(
                            f"      ⚠ {conv['conv_id']}: SKIPPING entire conversation - {skip_reason}"
                        )
                        break
                    
                    prompt_template = self.evaluator.PROMPT_WITH_REASON if self.evaluator.include_reason else self.evaluator.PROMPT_WITHOUT_REASON
                    prompt = prompt_template.replace('{turns}', turns_text).replace('{intention}', intent)
                    
                    # FINAL VALIDATION: Verify complete prompt fits within model limit
                    prompt_tokens = self.model.count_tokens(prompt)
                    if prompt_tokens + max_tokens_step2 > 40960:
                        # SKIP ENTIRE CONVERSATION if ANY prompt exceeds limit
                        skip_whole_conversation = True
                        skip_reason = f"Prompt exceeds model limit in Step 2 ({prompt_tokens} + {max_tokens_step2} > 40960)"
                        self.logger.warning(
                            f"      ⚠ {conv['conv_id']}: SKIPPING entire conversation - {skip_reason}"
                        )
                        # important, break here.
                        break
                    
                    conv_prompts.append(prompt)
                    conv_intentions.append(intent)
                    all_eval_prompts.append(prompt)
                
                # If conversation should be skipped, mark it as such
                if skip_whole_conversation:
                    eval_metadata.append({
                        'conv_idx': conv_idx,
                        'has_prompts': False,
                        'intentions': intentions,
                        'conv': conv,
                        'status': 'skipped',
                        'error_reason': skip_reason
                    })
                else:
                    eval_metadata.append({
                        'conv_idx': conv_idx,
                        'has_prompts': len(conv_prompts) > 0,
                        'num_prompts': len(conv_prompts),
                        'intentions': conv_intentions,
                        'conv': conv
                    })
            
            # Step 2b: Evaluate prompts with optional sub-batching
            total_prompts = len(all_eval_prompts)
            max_eval_batch = self.config.get('model', {}).get('max_eval_batch_size', 0)
            
            self.logger.info(f"Prepared {total_prompts} evaluation prompts")
            
            if all_eval_prompts:
                all_responses = []
                
                # Sub-batch if max_eval_batch_size is set
                if max_eval_batch > 0 and total_prompts > max_eval_batch:
                    num_sub_batches = (total_prompts + max_eval_batch - 1) // max_eval_batch
                    self.logger.info(f"Sub-batching: {total_prompts} prompts split into {num_sub_batches} vLLM calls ({max_eval_batch} prompts each, max_tokens={max_tokens_step2})\n")
                    
                    for i in range(0, total_prompts, max_eval_batch):
                        sub_batch = all_eval_prompts[i:i + max_eval_batch]
                        batch_num = i // max_eval_batch + 1
                        self.logger.info(f"  [Sub-batch {batch_num}/{num_sub_batches}] Processing {len(sub_batch)} prompts...")
                        
                        sub_responses = await self.model.a_generate_batch(sub_batch, max_tokens=max_tokens_step2)
                        all_responses.extend(sub_responses)
                        
                        self.logger.info(f"  ✓ Sub-batch {batch_num} complete: {len(sub_responses)} verdicts\n")
                    
                    self.logger.info(f"✓ Step 2 Complete: Received {len(all_responses)} total verdicts from {num_sub_batches} sub-batches\n")
                else:
                    # Process all in one call
                    self.logger.info(f"Batching all {total_prompts} prompts in 1 vLLM call (max_tokens={max_tokens_step2})...\n")
                    all_responses = await self.model.a_generate_batch(all_eval_prompts, max_tokens=max_tokens_step2)
                    self.logger.info(f"✓ Step 2 Complete: Received {len(all_responses)} verdicts\n")
            else:
                all_responses = []
                self.logger.warning("No evaluation prompts to process\n")
            
            # Step 3: Parse responses and organize by conversation
            self.logger.info("=" * 80)
            self.logger.info(f"[BATCH STEP 3] ORGANIZING RESULTS")
            self.logger.info("=" * 80)
            results = []
            response_idx = 0
            
            for metadata in eval_metadata:
                try:
                    if pbar:
                        pbar.update(1)
                    
                    if not metadata['has_prompts']:
                        # Handle skipped/failed conversations from Step 1
                        conv = metadata['conv']
                        status = metadata.get('status', 'failed')
                        error_reason = metadata.get('error_reason', 'No intentions extracted')
                        
                        result = {
                            'conv_id': conv['conv_id'],
                            'platform': conv['platform'],
                            'turn_count': len(conv['turns']),
                            'status': status,
                            'error': error_reason,
                            'intentions': [],
                            'verdicts': [],
                            'completeness_score': None
                        }
                        results.append(result)
                        
                        if status == 'skipped':
                            self.logger.warning(f"⊗ {conv['conv_id']}: SKIPPED - {error_reason}")
                        else:
                            self.logger.warning(f"✗ {conv['conv_id']}: FAILED - {error_reason}")
                        continue
                    
                    conv = metadata['conv']
                    intentions = metadata['intentions']
                    num_prompts = metadata['num_prompts']
                    
                    # Extract this conversation's responses
                    conv_responses = all_responses[response_idx:response_idx + num_prompts]
                    response_idx += num_prompts
                    
                    # Parse verdicts
                    verdicts = []
                    for intent, response in zip(intentions, conv_responses):
                        result = self.evaluator._extract_json_verdict(response)
                        verdict_data = {
                            'intention': intent,
                            'verdict': result.get('verdict', 'unknown'),
                            'content_ratio': 1.0  # Already filtered
                        }
                        if self.evaluator.include_reason and 'reason' in result:
                            verdict_data['reason'] = result['reason']
                        verdicts.append(verdict_data)
                    
                    # Calculate score
                    score = self.calculator.calculate(verdicts)
                    
                    result = {
                        'conv_id': conv['conv_id'],
                        'platform': conv['platform'],
                        'turn_count': len(conv['turns']),
                        'intentions': intentions,
                        'verdicts': verdicts,
                        'completeness_score': score
                    }
                    
                    self.logger.info(f"✓ {conv['conv_id']}: score={score:.3f}")
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"✗ {metadata['conv'].get('conv_id', 'unknown')}: {e}")
                    results.append(Exception(f"{metadata['conv'].get('conv_id', 'unknown')}: {e}"))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}", exc_info=True)
            return [Exception(str(e)) for _ in conversations]
    
    async def process_conversation(self, conv: Dict) -> Optional[Dict]:
        """Process single conversation (legacy method, kept for compatibility)"""
        try:
            if not DataLoader.validate(conv):
                return None
            
            # Extract intentions
            intentions = await self.extractor.extract(conv['turns'])
            if not intentions:
                return None
            
            # Generate verdicts (async/parallel)
            verdicts = await self.evaluator.evaluate_async(conv['turns'], intentions)
            
            # Calculate score
            score = self.calculator.calculate(verdicts)
            
            # If score is None, all verdicts were skipped - treat as failure
            if score is None:
                self.logger.warning(f"✗ {conv['conv_id']}: All verdicts skipped (excessive truncation or evaluation failure)")
                return None
            
            # Build result
            result = {
                'conv_id': conv['conv_id'],
                'platform': conv['platform'],
                'num_turns': len(conv['turns']),
                'num_intentions': len(intentions),
                'intentions': intentions,
                'verdicts': verdicts,
                'completeness_score': score
            }
            
            self.logger.info(f"✓ {conv['conv_id']}: score={score:.3f}")
            return result
        
        except Exception as e:
            self.logger.error(f"✗ {conv.get('conv_id', 'unknown')}: {e}", exc_info=True)
            return None
    
    async def run(self, data_dir: str = 'data'):
        """Run pipeline on all 5 CSV files, processing one platform at a time"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PIPELINE - Processing platform by platform")
        self.logger.info("=" * 80)
        
        # Load checkpoint with platform tracking
        total_count, platform_data = self.checkpoint.load()
        
        self.logger.info(f"Loaded checkpoint: {total_count} conversations processed")
        if platform_data:
            completed_platforms = [p for p, data in platform_data.items() if data.get('completed', False)]
            if completed_platforms:
                self.logger.info(f"  Completed platforms: {', '.join(completed_platforms)}")
        
        # Load data from all 5 CSV files (pass platform_data to check completion)
        loader = DataLoader(data_dir, self.config['data']['batch_size'], platform_data)
        
        # Track current platform and its progress bar
        current_platform = None
        platform_pbar = None
        
        try:
            for platform, batch in loader:
                # Initialize platform data if not exists (completion already checked in DataLoader)
                if platform not in platform_data:
                    platform_data[platform] = {'completed': False, 'processed_ids': [], 'total_count': 0}
                
                # If this is a new platform, create progress bar
                if current_platform != platform:
                    # Close previous platform's progress bar if exists
                    if platform_pbar is not None:
                        platform_pbar.close()
                        # Mark previous platform as completed
                        if current_platform:
                            platform_data[current_platform]['completed'] = True
                            self.checkpoint.save(total_count, platform_data)
                            self.logger.info(f"\n✓ Platform {current_platform.upper()} COMPLETED")
                            self.logger.info(f"  Total conversations processed: {len(platform_data[current_platform]['processed_ids'])}")
                    
                    current_platform = platform
                    processed_ids_set = set(platform_data[platform]['processed_ids'])
                    
                    # Count total conversations for this platform by loading all data once
                    platform_df = loader.load_platform_data(platform)
                    all_convs = loader.convert_to_conversations(platform_df, platform)
                    total_platform_convs = len(all_convs)
                    platform_data[platform]['total_count'] = total_platform_convs
                    already_processed = len(processed_ids_set)
                    
                    # Start processing this platform
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"Platform: {platform.upper()}")
                    self.logger.info(f"{'='*80}")
                    self.logger.info(f"Total conversations: {total_platform_convs}")
                    self.logger.info(f"Already processed: {already_processed}\n")
                    
                    # Create platform-level progress bar (position=0 keeps it at top)
                    platform_pbar = tqdm(total=total_platform_convs, initial=already_processed, 
                                        desc=f"{platform.upper()}", unit="conv", position=0, leave=True)
                
                processed_ids_set = set(platform_data[platform]['processed_ids'])
                
                # Filter out already processed conversations
                unprocessed = [conv for conv in batch if conv['conv_id'] not in processed_ids_set]
                
                if not unprocessed:
                    # All conversations already processed, skip without updating progress bar
                    continue
                
                # OPTIMIZED: Process entire batch with 2 vLLM calls instead of 2*N calls
                # Don't pass progress bar to avoid nested bars
                results = await self.process_batch_optimized(unprocessed, pbar=None)
                
                # Process ALL results (successful, failed, skipped) and save them
                batch_results = []
                for conv, result in zip(unprocessed, results):
                    if isinstance(result, Exception):
                        error_msg = str(result)
                        # Create a result record for failed/skipped conversations
                        error_result = {
                            'conv_id': conv['conv_id'],
                            'platform': conv['platform'],
                            'turn_count': len(conv['turns']),
                            'status': 'failed',
                            'error': str(result),
                            'intentions': [],
                            'verdicts': [],
                            'completeness_score': 0.0
                        }
                        
                        # Check for token overflow error
                        if "sequence length is longer than" in error_msg or "indexing errors" in error_msg:
                            self.logger.warning(f"  ⚠ {conv['conv_id']}: Token overflow, skipped")
                            error_result['status'] = 'skipped'
                            error_result['error'] = 'Token overflow - conversation too long'
                        else:
                            self.logger.error(f"  ✗ {conv['conv_id']}: {result}")
                        
                        batch_results.append(error_result)
                    elif result:
                        # Successful result
                        result['status'] = 'success'
                        batch_results.append(result)
                    else:
                        # No result returned (shouldn't happen, but handle it)
                        empty_result = {
                            'conv_id': conv['conv_id'],
                            'platform': conv['platform'],
                            'turn_count': len(conv['turns']),
                            'status': 'failed',
                            'error': 'No result returned',
                            'intentions': [],
                            'verdicts': [],
                            'completeness_score': 0.0
                        }
                        batch_results.append(empty_result)
                        self.logger.warning(f"  ⚠ {conv['conv_id']}: No result returned")
                    
                    # Mark as processed regardless of success/failure
                    platform_data[platform]['processed_ids'].append(conv['conv_id'])
                    processed_ids_set.add(conv['conv_id'])
                    total_count += 1
                    
                    # Update platform progress bar
                    platform_pbar.update(1)
                
                # Save ALL results (including failures) to output file
                for result in batch_results:
                    self.checkpoint.save_result(result, self.config['output']['results_dir'])
                
                # Save checkpoint after each batch
                self.checkpoint.save(total_count, platform_data)
            
            # Close final platform progress bar if exists
            if platform_pbar is not None:
                platform_pbar.close()
                if current_platform:
                    platform_data[current_platform]['completed'] = True
                    self.checkpoint.save(total_count, platform_data)
                    self.logger.info(f"\n✓ Platform {current_platform.upper()} COMPLETED")
                    self.logger.info(f"  Total conversations processed: {len(platform_data[current_platform]['processed_ids'])}")
            
            # Final checkpoint
            self.checkpoint.save(total_count, platform_data)
            
            # Aggregate results
            self.logger.info("\n" + "=" * 80)
            self.logger.info("AGGREGATING RESULTS")
            self.logger.info("=" * 80)
            
            results_dir = Path(self.config['output']['results_dir'])
            all_platforms_file = results_dir / "all_platforms_completeness.jsonl"
            
            if all_platforms_file.exists():
                all_results = []
                with open(all_platforms_file) as f:
                    for line in f:
                        all_results.append(json.loads(line))
                
                aggregated = self.calculator.aggregate_by_platform(all_results)
                
                # Save aggregated
                agg_file = results_dir / "aggregated_results.json"
                with open(agg_file, 'w') as f:
                    json.dump(aggregated, f, indent=2)
                
                self.logger.info(f"✓ Aggregated results saved to {agg_file}")
                self.logger.info(f"✓ Per-platform files:")
                for platform in aggregated.keys():
                    platform_file = results_dir / f"{platform}_completeness.jsonl"
                    if platform_file.exists():
                        self.logger.info(f"    - {platform_file}")
            
            self.logger.info("=" * 80)
            self.logger.info(f"COMPLETED: {total_count} conversations")
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
