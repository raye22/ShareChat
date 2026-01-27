import json
import re
from typing import List, Dict

class IntentExtractor:
    MAX_CONTEXT_TOKENS = 40960  # Model's actual token limit
    
    PROMPT_TEMPLATE = """### SYSTEM ROLE
You are an expert Conversation Analyst.

### TASK
Extract a chronological list of **User Intentions** from the conversation log.

### INSTRUCTIONS
1. **Identify Distinct Goals:** Focus on information seeking, task requests, or problem-solving goals.
2. **Maintain Order:** The first item in your list must correspond to the user's first real request, and so on.
3. **Ignore Noise:** Skip purely social turns (e.g., "Hello", "Thank you", "Okay") unless they are the only message.

### OUTPUT FORMAT
Respond with a raw JSON object enclosed strictly within <output> tags.
The JSON must have exactly one field: "intentions" (a list of strings).

### EXAMPLE

Input Turns:
[
    {"role": "user", "content": "Hi, I need help with Python."},
    {"role": "user", "content": "How do I reverse a list?"},
    {"role": "user", "content": "Thanks. Also, what is the weather in Tokyo?"}
]

Correct Output:
<output>
{{
    "intentions": [
        "User wants to know how to reverse a list in Python",
        "User wants to check the weather in Tokyo"
    ]
}}
</output>

### CURRENT INPUT
Turns:
{turns}

### RESPONSE
Generate the JSON response now.
1. Start your response with the opening tag <output>.
2. Ensure the JSON is valid.
3. End with the closing tag </output>.
"""
    
    def __init__(self, model_wrapper, max_llm_tokens: int = 1024):
        self.model = model_wrapper
        self.max_llm_tokens = max_llm_tokens
    
    def _truncate_llm_response(self, content: str) -> str:
        """Keep first max_tokens of LLM response"""
        tokens = self.model.tokenizer.encode(content)
        if len(tokens) <= self.max_llm_tokens:
            return content
        truncated_tokens = tokens[:self.max_llm_tokens]
        return self.model.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _extract_json_intentions(self, response: str) -> List[str]:
        """Robustly extract intentions JSON from response with <o> tags
        
        Returns:
            List of intention strings, or empty list on failure
        """
        # Strategy 1: Look for <o>...</o> tags
        output_match = re.search(r'<output>(.*?)</output>', response, re.DOTALL)
        if output_match:
            json_str = output_match.group(1).strip()
            try:
                data = json.loads(json_str)
                # Validate intentions field
                if 'intentions' in data and isinstance(data['intentions'], list):
                    intentions = [str(i).strip() for i in data['intentions'] if str(i).strip()]
                    if intentions:
                        return intentions
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to find JSON without tags (fallback)
        json_match = re.search(r'\{[^{}]*"intentions"[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if 'intentions' in data and isinstance(data['intentions'], list):
                    intentions = [str(i).strip() for i in data['intentions'] if str(i).strip()]
                    if intentions:
                        return intentions
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try more flexible JSON parsing (nested arrays possible)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                if 'intentions' in data and isinstance(data['intentions'], list):
                    intentions = [str(i).strip() for i in data['intentions'] if str(i).strip()]
                    if intentions:
                        return intentions
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Failed all strategies
        return []
    
    async def extract(self, turns: List[Dict]) -> List[str]:
        """Extract intentions from conversation"""
        from modules.logger import get_logger
        logger = get_logger()
        
        # Use full user content, truncate LLM responses to first 1024 tokens
        truncation_stats = []
        processed_turns = []
        
        for i, t in enumerate(turns):
            if t['role'] == 'user':
                processed_turns.append(f"Turn {i} ({t['role']}): {t['content']}")
                truncation_stats.append({'turn': i, 'role': 'user', 'truncated': False, 'original_tokens': self.model.count_tokens(t['content'])})
            # don't feed assistant turns
            # else:
            #     original_content = t['content']
            #     original_tokens = len(self.model.tokenizer.encode(original_content))
            #     truncated_content = self._truncate_llm_response(original_content)
            #     truncated_tokens = len(self.model.tokenizer.encode(truncated_content))
            #     was_truncated = original_tokens > self.max_llm_tokens
                
            #     processed_turns.append(f"Turn {i} ({t['role']}): {truncated_content}")
            #     truncation_stats.append({
            #         'turn': i, 
            #         'role': 'llm', 
            #         'truncated': was_truncated,
            #         'original_tokens': original_tokens,
            #         'kept_tokens': truncated_tokens,
            #         'kept_ratio': truncated_tokens / original_tokens if original_tokens > 0 else 1.0
            #     })
        
        turns_text = "\n".join(processed_turns)
        
        # Log truncation stats
        total_truncated = sum(1 for s in truncation_stats if s['truncated'])
        if total_truncated > 0:
            logger.info(f"  [Step 1] Extracting intentions from {len(turns)} turns (truncated {total_truncated} LLM responses)...")
            for stat in truncation_stats:
                if stat['truncated']:
                    logger.info(f"      Turn {stat['turn']}: {stat['original_tokens']} → {stat['kept_tokens']} tokens ({stat['kept_ratio']:.1%})")
        else:
            logger.info(f"  [Step 1] Extracting intentions from {len(turns)} turns (no truncation needed)...")
        
        prompt = self.PROMPT_TEMPLATE.replace('{turns}', turns_text)
        prompt_tokens = self.model.count_tokens(prompt)
        logger.info(f"      Prompt: {prompt_tokens} tokens")
        
        response = await self.model.a_generate(prompt, max_tokens=256)
        logger.info(f"      Response: {len(response)} chars")
        
        # Use robust extraction
        intentions = self._extract_json_intentions(response)
        
        if not intentions:
            logger.warning(f"      ⚠ Could not extract intentions")
            logger.warning(f"      Raw response: {response[:300]}...")
        else:
            logger.info(f"      ✓ Extracted {len(intentions)} intentions")
        
        return intentions
    
    async def extract_batch(self, conversations: List[Dict], max_tokens: int = 256) -> List[tuple]:
        """Extract intentions for batch of conversations in single vLLM call
        
        Args:
            conversations: List of conversation dicts
            max_tokens: Max tokens for extraction (default 256)
            
        Returns:
            List of tuples: [(intentions_list, is_valid), ...]
        """
        from modules.logger import get_logger
        logger = get_logger()
        
        # Build all prompts
        prompts = []
        conv_metadata = []
        
        for conv in conversations:
            turns = conv.get('turns', [])
            if not turns:
                prompts.append(None)
                conv_metadata.append({'valid': False})
                continue
            
            # Process turns (keep user content only)
            processed_turns = []
            for i, t in enumerate(turns):
                if t['role'] == 'user':
                    processed_turns.append(f"Turn {i} ({t['role']}): {t['content']}")
            
            turns_text = "\n".join(processed_turns)
            
            # TRUNCATE IF NEEDED (same logic as evaluator)
            prompt_template_overhead = 500  # Estimate for template + instructions
            available_tokens = self.MAX_CONTEXT_TOKENS - prompt_template_overhead - max_tokens
            
            # Count tokens in conversation
            conv_tokens = self.model.count_tokens(turns_text)
            
            if conv_tokens > available_tokens:
                # Calculate truncation ratio
                ratio = available_tokens / conv_tokens
                
                # SKIP if truncation is too severe (> 20% content loss)
                if ratio < 0.8:
                    logger.warning(
                        f"      ⚠ {conv.get('conv_id')}: SKIPPING - Severe truncation required {conv_tokens} → {available_tokens} tokens "
                        f"(ratio: {ratio:.1%}, threshold: 80%)"
                    )
                    prompts.append(None)
                    conv_metadata.append({'valid': False, 'conv_id': conv.get('conv_id'), 'skipped': True, 'reason': 'Severe truncation required'})
                    continue
                
                # Truncate by ACTUAL TOKENS to fit exactly within available_tokens
                turns_tokens = self.model.tokenizer.encode(turns_text)
                if len(turns_tokens) > available_tokens:
                    truncated_tokens = turns_tokens[:available_tokens]
                    turns_text = self.model.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    logger.info(f"      Truncated {conv.get('conv_id')}: {len(turns_tokens)} → {available_tokens} tokens")
            
            prompt = self.PROMPT_TEMPLATE.replace('{turns}', turns_text)
            
            # FINAL VALIDATION: Check that complete prompt fits within model limit
            final_prompt_tokens = self.model.count_tokens(prompt)
            if final_prompt_tokens + max_tokens > self.MAX_CONTEXT_TOKENS:
                logger.warning(
                    f"      ⚠ {conv.get('conv_id')}: SKIPPING - Final prompt too long ({final_prompt_tokens} + {max_tokens} > {self.MAX_CONTEXT_TOKENS})"
                )
                prompts.append(None)
                conv_metadata.append({'valid': False, 'conv_id': conv.get('conv_id'), 'skipped': True, 'reason': 'Final prompt exceeds model limit'})
                continue
            
            prompts.append(prompt)
            conv_metadata.append({'valid': True, 'conv_id': conv.get('conv_id'), 'turn_count': len(turns)})
        
        # Filter valid prompts
        valid_prompts = [p for p in prompts if p is not None]
        logger.info(f"    Extracting intentions for {len(valid_prompts)} conversations in single vLLM call...")
        
        # Batch generate
        if valid_prompts:
            responses = await self.model.a_generate_batch(valid_prompts, max_tokens=max_tokens)
        else:
            responses = []
        
        # Process results
        results = []
        response_idx = 0
        
        for metadata in conv_metadata:
            if not metadata['valid']:
                # Check if it was skipped due to truncation
                if metadata.get('skipped'):
                    results.append(([], False, 'skipped', metadata.get('reason', 'Severe truncation required')))
                else:
                    results.append(([], False, 'failed', 'No valid turns'))
            else:
                response = responses[response_idx]
                response_idx += 1
                
                intentions = self._extract_json_intentions(response)
                
                if not intentions:
                    logger.warning(f"      ⚠ {metadata.get('conv_id')}: Could not extract intentions")
                    results.append(([], False, 'failed', 'Could not extract intentions'))
                else:
                    logger.info(f"      ✓ {metadata.get('conv_id')}: {len(intentions)} intentions")
                    results.append((intentions, True, 'success', None))
        
        return results