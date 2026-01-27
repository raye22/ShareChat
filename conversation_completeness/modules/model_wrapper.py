import asyncio
import torch
import os
import sys
import multiprocessing
import logging
from threading import Lock
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

# CRITICAL: Set spawn method for multiprocessing (required by vLLM 0.11+)
if sys.platform != 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Set HuggingFace cache to volume with more space
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache'
# Optimize vLLM settings (MUST be set before vLLM import)
os.environ['VLLM_USE_V1'] = '0'  # Force V0 engine (V1 in 0.11.2 has bugs)
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'  # Use FlashAttention-2 (xformers incompatible with paged attention)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

class ModelWrapper:
    def __init__(self, model_name: str, use_vllm: bool = True):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.tokenizer_lock = Lock()  # Prevent concurrent tokenizer access
        self.model = None
        self.llm = None
        self.logger = logging.getLogger("pipeline")  # Use same logger as pipeline
        
        try:
            if use_vllm:
                self._init_vllm()
            else:
                self._init_transformers()
        except Exception as e:
            if use_vllm:
                self.logger.warning(f"⚠ vLLM failed: {e}")
                self.logger.warning("Falling back to Transformers (slower but more stable)")
                self.use_vllm = False
                self._init_transformers()
            else:
                raise
    
    def _init_vllm(self):
        """Initialize vLLM for fast inference with batching support
        
        NOTE: vLLM initialization is LAZY - actual LLM is created on first inference call.
        This avoids multiprocessing issues during Pipeline instantiation.
        """
        try:
            from vllm import SamplingParams
            self.logger.info("="*60)
            self.logger.info("vLLM will initialize on first inference call (lazy loading)")
            self.logger.info("="*60)
            
            # Store configuration, but DON'T create LLM yet
            # Note: VLLM_USE_V1=0 is set at module import time
            
            self.vllm_config = {
                "model": self.model_name,
                "dtype": "float16",
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1,
                "enable_prefix_caching": False,  # Disabled - causes xformers compatibility issues
                "trust_remote_code": True,
                "disable_log_stats": True,
                "enforce_eager": True,  # CRITICAL: Bypass V1 engine compilation bugs
                "kv_cache_dtype": "auto",
            }
            
            # LLM will be created lazily on first _generate() call
            self.llm = None
            
            # Default sampling params (will be overridden with max_tokens parameter in generate calls)
            self.default_sampling_params = {
                'temperature': 0.7,
                'top_p': 0.95
            }
            
            # Load tokenizer separately for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.model_max_length = 131072  # For counting only
            
            self.logger.info(f"✓ vLLM config prepared (lazy loading)")
            self.logger.info(f"  Model: {self.model_name}")
            self.logger.info(f"  GPU: A100 40GB")
            self.logger.info(f"  GPU memory: 90% utilization")
            self.logger.info(f"  Float16: enabled (2x faster)")
            self.logger.info(f"  Prefix caching: DISABLED (xformers compatibility)")
            self.logger.info(f"  Eager mode: ENABLED (bypass V1 engine bugs)")
            self.logger.info(f"  Expected speed: 2-3x faster than Transformers")
            self.logger.info("="*60)
            
        except ImportError as e:
            self.logger.error(f"✗ vLLM ImportError: {e}")
            self.logger.error("Install with: pip install vllm")
            raise
        except Exception as e:
            self.logger.error(f"✗ vLLM config failed: {type(e).__name__}")
            self.logger.error(f"Message: {str(e)[:500]}")
            raise
    
    def _init_transformers(self):
        """Initialize Transformers for fallback (slower but always works)"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = 131072
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use float16 for speed
            device_map="auto"
        )
        
        self.logger.info(f"✓ Transformers initialized: {self.model_name}")
        self.logger.info(f"  Tokenizer max length: {self.tokenizer.model_max_length}")
        self.logger.warning(f"  WARNING: Transformers is slower than vLLM")
        self.logger.info(f"  For faster inference, install vLLM: pip install vllm")
    
    async def a_generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate for single prompt"""
        return await asyncio.to_thread(self._generate, prompt, max_tokens)
    
    async def a_generate_batch(self, prompts: List[str], max_tokens: int = 1024) -> List[str]:
        """Generate for multiple prompts in a single batched call (vLLM native batching)"""
        return await asyncio.to_thread(self._generate_batch, prompts, max_tokens)
    
    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate single prompt"""
        if self.use_vllm:
            # Lazy initialization: create LLM on first call
            if self.llm is None:
                from vllm import LLM
                self.logger.info("="*80)
                self.logger.info("INITIALIZING vLLM ENGINE (this may take 2-5 minutes on first call)...")
                self.logger.info("="*80)
                try:
                    self.llm = LLM(**self.vllm_config)
                    self.logger.info("="*80)
                    self.logger.info("✓ vLLM ENGINE INITIALIZED AND READY!")
                    self.logger.info("="*80)
                except Exception as e:
                    self.logger.error("="*80)
                    self.logger.error("✗ vLLM INITIALIZATION FAILED")
                    self.logger.error("="*80)
                    self.logger.error(f"Error type: {type(e).__name__}")
                    self.logger.error(f"Error message: {str(e)}")
                    self.logger.error("="*80)
                    raise
            
            # Create sampling params with specified max_tokens
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                **self.default_sampling_params
            )
            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text
        else:
            # Truncate prompt if it exceeds model max length (reserve space for generation)
            max_input_length = self.tokenizer.model_max_length - max_tokens - 10
            with self.tokenizer_lock:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_input_length
                ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, top_p=0.95)
            
            with self.tokenizer_lock:
                return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    def _generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Generate for multiple prompts using vLLM's native batching"""
        if self.use_vllm:
            # Lazy initialization: create LLM on first call
            if self.llm is None:
                from vllm import LLM
                self.logger.info("="*80)
                self.logger.info("INITIALIZING vLLM ENGINE (this may take 2-5 minutes on first call)...")
                self.logger.info("="*80)
                try:
                    self.llm = LLM(**self.vllm_config)
                    self.logger.info("="*80)
                    self.logger.info("✓ vLLM ENGINE INITIALIZED AND READY!")
                    self.logger.info("="*80)
                except Exception as e:
                    self.logger.error("="*80)
                    self.logger.error("✗ vLLM INITIALIZATION FAILED")
                    self.logger.error("="*80)
                    self.logger.error(f"Error type: {type(e).__name__}")
                    self.logger.error(f"Error message: {str(e)}")
                    self.logger.error("="*80)
                    raise
            
            # Single vLLM call with all prompts - vLLM handles batching internally
            # Create sampling params with specified max_tokens
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                **self.default_sampling_params
            )
            self.logger.info(f"    Batching {len(prompts)} prompts in single vLLM call (max_tokens={max_tokens})...")
            outputs = self.llm.generate(prompts, sampling_params=sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            # Fallback: Transformers processes sequentially
            self.logger.warning(f"    Transformers: processing {len(prompts)} prompts sequentially...")
            return [self._generate(prompt, max_tokens) for prompt in prompts]
    
    def count_tokens(self, text: str) -> int:
        """Count ACTUAL tokens WITHOUT truncation - thread-safe"""
        with self.tokenizer_lock:
            try:
                # Get ACTUAL token count without truncation
                tokens = self.tokenizer.encode(
                    text,
                    truncation=False,  # Get real count, not truncated
                    add_special_tokens=False
                )
                return len(tokens)
            except Exception as e:
                # Fallback: estimate based on characters (rough approximation)
                return len(text) // 4
