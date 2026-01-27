import json
import asyncio
import re
from typing import List, Dict

class VerdictEvaluator:
    # Prompt template for MULTIPLE intentions in one prompt
    PROMPT_MULTI_INTENTIONS = """### SYSTEM ROLE
You are an expert Quality Assurance Evaluator for AI conversations.

### TASK
Evaluate whether EACH of the listed User Intentions was satisfied by the LLM based on the conversation history.

### CRITERIA
- **Verdict: "yes"** if:
    1. The LLM provided the correct information, code, or creative output requested.
    2. The user explicitly expressed satisfaction (e.g., "Thanks", "That works").
    3. The interaction reached a logical conclusion where the goal was met.

- **Verdict: "partial"** if:
    1. The LLM started addressing the request but the conversation ended before completion.
    2. The LLM provided some relevant information but missed key aspects of the request.
    3. The LLM gave a partial solution that requires additional steps the user would need to complete.
    4. The user asked follow-up questions indicating partial understanding/satisfaction.

- **Verdict: "no"** if:
    1. The LLM refused the request (unless it was a safety violation).
    2. The LLM completely misunderstood the request or provided irrelevant information.
    3. The user expressed frustration or repeatedly asked the same thing without progress.
    4. The LLM asked for clarification but the conversation ended before any attempt to help.

### OUTPUT FORMAT
Respond with a JSON array of verdict objects enclosed strictly within <output> tags.
Each object must have these fields:
- "intention": (repeat the intention text)
- "verdict": (value must be "yes", "partial", or "no")

### EXAMPLE

Intentions:
1. User wants to learn about Python decorators
2. User wants to see a code example

Turns: [
    {{"role": "user", "content": "Can you explain Python decorators and show me an example?"}},
    {{"role": "assistant", "content": "Sure! Decorators are functions that modify other functions. They use @ syntax. Here's an example:\\n\\n```python\\n@decorator\\ndef my_func():\\n    pass\\n```"}}
]

Correct Output:
<output>
[
    {{"intention": "User wants to learn about Python decorators", "verdict": "yes"}},
    {{"intention": "User wants to see a code example", "verdict": "yes"}}
]
</output>

### CURRENT INPUT
Intentions:
{intentions}

Turns:
{turns}

### RESPONSE
Generate the JSON array response now.
1. Start your response with the opening tag <output>.
2. Ensure the JSON array is valid and contains exactly {num_intentions} verdict objects.
3. End with the closing tag </output>.

"""

    # Prompt template WITH reason (verbose mode) - Single intention
    PROMPT_WITH_REASON = """### SYSTEM ROLE
You are an expert Quality Assurance Evaluator for AI conversations.

### TASK
Determine if the specific **User Intention** was satisfied by the LLM based on the conversation history.

### CRITERIA
- **Verdict: "yes"** if:
    1. The LLM provided the correct information, code, or creative output requested.
    2. The user explicitly expressed satisfaction (e.g., "Thanks", "That works").
    3. The interaction reached a logical conclusion where the goal was met.

- **Verdict: "partial"** if:
    1. The LLM started addressing the request but the conversation ended before completion.
    2. The LLM provided some relevant information but missed key aspects of the request.
    3. The LLM gave a partial solution that requires additional steps the user would need to complete.
    4. The user asked follow-up questions indicating partial understanding/satisfaction.

- **Verdict: "no"** if:
    1. The LLM refused the request (unless it was a safety violation).
    2. The LLM completely misunderstood the request or provided irrelevant information.
    3. The user expressed frustration or repeatedly asked the same thing without progress.
    4. The LLM asked for clarification but the conversation ended before any attempt to help.

### OUTPUT FORMAT
Respond with a raw JSON object enclosed strictly within <output> tags.
The JSON must have these fields:
- "intention": (repeat the intention text)
- "verdict": (value must be "yes", "partial", or "no")
- "reason": (brief explanation of the verdict, 1-2 sentences)

### EXAMPLE

Intention: "User wants to learn about Python decorators"
Turns: [
    {{"role": "user", "content": "Can you explain Python decorators?"}},
    {{"role": "assistant", "content": "Sure! Decorators are functions that modify other functions. They use @ syntax. Would you like to see a code example?"}}
]

Correct Output:
<output>
{{"intention": "User wants to learn about Python decorators", "verdict": "yes", "reason": "The LLM provided a clear explanation and proactively asked if the user wants more details, showing engagement."}}
</output>

### CURRENT INPUT
Intention: {intention}

Turns:
{turns}

### RESPONSE
Generate the JSON response now.
1. Start your response with the opening tag <output>.
2. Ensure the JSON is valid.
3. End with the closing tag </output>.

"""

    # Prompt template WITHOUT reason (concise mode) - Single intention
    PROMPT_WITHOUT_REASON = """### SYSTEM ROLE
You are an expert Quality Assurance Evaluator for AI conversations.

### TASK
Determine if the specific **User Intention** was satisfied by the LLM based on the conversation history.

### CRITERIA
- **Verdict: "yes"** if:
    1. The LLM provided the correct information, code, or creative output requested.
    2. The user explicitly expressed satisfaction (e.g., "Thanks", "That works").
    3. The interaction reached a logical conclusion where the goal was met.

- **Verdict: "partial"** if:
    1. The LLM started addressing the request but the conversation ended before completion.
    2. The LLM provided some relevant information but missed key aspects of the request.
    3. The LLM gave a partial solution that requires additional steps the user would need to complete.
    4. The user asked follow-up questions indicating partial understanding/satisfaction.

- **Verdict: "no"** if:
    1. The LLM refused the request (unless it was a safety violation).
    2. The LLM completely misunderstood the request or provided irrelevant information.
    3. The user expressed frustration or repeatedly asked the same thing without progress.
    4. The LLM asked for clarification but the conversation ended before any attempt to help.

### OUTPUT FORMAT
Respond with a raw JSON object enclosed strictly within <output> tags.
The JSON must have these fields:
- "intention": (repeat the intention text)
- "verdict": (value must be "yes", "partial", or "no")

### EXAMPLE

Intention: "User wants to learn about Python decorators"
Turns: [
    {{"role": "user", "content": "Can you explain Python decorators?"}},
    {{"role": "assistant", "content": "Sure! Decorators are functions that modify other functions. They use @ syntax. Would you like to see a code example?"}}
]

Correct Output:
<output>
{{"intention": "User wants to learn about Python decorators", "verdict": "yes"}}
</output>

### CURRENT INPUT
Intention: {intention}

Turns:
{turns}

### RESPONSE
Generate the JSON response now.
1. Start your response with the opening tag <output>.
2. Ensure the JSON is valid.
3. End with the closing tag </output>.

"""

    MAX_CONTEXT_TOKENS = 40960  # Qwen3-8B actual model limit (max_position_embeddings)
    
    def __init__(self, model_wrapper, min_content_ratio: float = 0.8, include_reason: bool = True):
        self.model = model_wrapper
        self.min_content_ratio = min_content_ratio
        self.include_reason = include_reason
    
    async def evaluate_async(self, turns: List[Dict], intentions: List[str]) -> List[Dict]:
        """Evaluate verdicts for all intentions using separate prompts with vLLM batch generation"""
        from modules.logger import get_logger
        logger = get_logger()
        
        logger.info(f"\n  [Evaluator] Evaluating {len(intentions)} intentions with separate prompts...")
        
        # Build all prompts first
        prompts = []
        prompt_metadata = []
        skipped_indices = []
        
        for idx, intent in enumerate(intentions):
            logger.info(f"    [Intention #{idx+1}] '{intent[:80]}{'...' if len(intent) > 80 else ''}'")
            
            turns_text, content_ratio = self._build_turns_text(turns, intent)
            
            # Skip if too much content was truncated
            if content_ratio < self.min_content_ratio:
                logger.warning(f"      SKIPPED - Content ratio {content_ratio:.1%} < threshold {self.min_content_ratio:.0%}")
                skipped_indices.append(idx)
                prompts.append(None)
                prompt_metadata.append({'intention': intent, 'content_ratio': content_ratio, 'skipped': True})
                continue
            
            logger.info(f"      Content ratio: {content_ratio:.1%}")
            
            # Select appropriate prompt template
            prompt_template = self.PROMPT_WITH_REASON if self.include_reason else self.PROMPT_WITHOUT_REASON
            prompt = prompt_template.replace('{turns}', turns_text).replace('{intention}', intent)
            prompt_tokens = self.model.count_tokens(prompt)
            logger.info(f"      Prompt: {prompt_tokens} tokens")
            
            prompts.append(prompt)
            prompt_metadata.append({'intention': intent, 'content_ratio': content_ratio, 'skipped': False})
        
        # Filter out skipped prompts
        valid_prompts = [p for p in prompts if p is not None]
        
        # Batch generation with vLLM - single call for all prompts!
        if valid_prompts:
            max_tokens = 256 if self.include_reason else 128
            logger.info(f"      Batching {len(valid_prompts)} prompts with max_tokens={max_tokens}...")
            responses = await self.model.a_generate_batch(valid_prompts, max_tokens=max_tokens)
        else:
            responses = []
        
        # Process results
        results = []
        response_idx = 0
        skipped_count = 0
        
        for idx, metadata in enumerate(prompt_metadata):
            if metadata['skipped']:
                results.append({
                    'intention': metadata['intention'],
                    'verdict': 'skipped',
                    'content_ratio': metadata['content_ratio']
                })
                skipped_count += 1
            else:
                response = responses[response_idx]
                response_idx += 1
                
                logger.info(f"    [Intention #{idx+1}] Response: {len(response)} chars")
                
                # Extract verdict
                result = self._extract_json_verdict(response)
                verdict = result.get('verdict', 'unknown')
                reason = result.get('reason', '')
                
                if verdict == 'unknown':
                    logger.warning(f"      ⚠ Could not parse response")
                    logger.warning(f"      Raw response: {response[:200]}...")
                else:
                    logger.info(f"      ✓ Verdict: {verdict}")
                    if self.include_reason and reason:
                        logger.info(f"      ✓ Reason: {reason[:100]}{'...' if len(reason) > 100 else ''}")
                
                output = {
                    'intention': metadata['intention'],
                    'verdict': verdict,
                    'content_ratio': metadata['content_ratio']
                }
                
                if self.include_reason and reason:
                    output['reason'] = reason
                
                results.append(output)
        
        if skipped_count > 0:
            logger.warning(f"  [Evaluator] Skipped {skipped_count}/{len(intentions)} intentions due to excessive truncation")
        
        return results
    
    async def _eval_one(self, turns: List[Dict], intention: str, idx: int = 0) -> Dict:
        """Evaluate single intention"""
        from modules.logger import get_logger
        logger = get_logger()
        
        logger.info(f"    [Intention #{idx+1}] '{intention[:80]}{'...' if len(intention) > 80 else ''}'")
        
        turns_text, content_ratio = self._build_turns_text(turns, intention)
        
        # Skip if too much content was truncated
        if content_ratio < self.min_content_ratio:
            logger.warning(f"      SKIPPED - Content ratio {content_ratio:.1%} < threshold {self.min_content_ratio:.0%}")
            return {
                'intention': intention,
                'verdict': 'skipped',
                'content_ratio': content_ratio
            }
        
        logger.info(f"      Content ratio: {content_ratio:.1%}")
        
        # Select appropriate prompt template based on include_reason setting
        prompt_template = self.PROMPT_WITH_REASON if self.include_reason else self.PROMPT_WITHOUT_REASON
        
        # Build prompt by direct string replacement (no .format() to avoid brace conflicts)
        prompt = prompt_template.replace('{turns}', turns_text).replace('{intention}', intention)
        prompt_tokens = self.model.count_tokens(prompt)
        logger.info(f"      Prompt: {prompt_tokens} tokens")
        
        max_tokens = 256 if self.include_reason else 128
        response = await self.model.a_generate(prompt, max_tokens=max_tokens)
        logger.info(f"      Response: {len(response)} chars")
        
        # Use robust extraction
        result = self._extract_json_verdict(response)
        verdict = result.get('verdict', 'unknown')
        reason = result.get('reason', '')
        
        if verdict == 'unknown':
            logger.warning(f"      ⚠ Could not parse response")
            logger.warning(f"      Raw response: {response[:200]}...")
        else:
            logger.info(f"      ✓ Verdict: {verdict}")
            if self.include_reason and reason:
                logger.info(f"      ✓ Reason: {reason[:100]}{'...' if len(reason) > 100 else ''}")
        
        output = {
            'intention': intention,
            'verdict': verdict,
            'content_ratio': content_ratio
        }
        
        # Include reason in output if requested and available
        if self.include_reason and reason:
            output['reason'] = reason
        
        return output
    
    def _build_turns_text_simple(self, turns: List[Dict]) -> tuple[str, float]:
        """Build turns text with truncation if needed (for multi-intention prompt)
        
        Returns:
            tuple: (turns_text, content_ratio) where content_ratio is the ratio of kept tokens to original tokens
        """
        # Reserve tokens for prompt template + intentions overhead
        overhead_tokens = 2000  # Rough estimate for template + intentions
        available_tokens = self.MAX_CONTEXT_TOKENS - overhead_tokens - 2048  # Reserve for output
        
        # Build full conversation
        full_turns = []
        for i, t in enumerate(turns):
            full_turns.append(f"Turn {i} ({t['role']}): {t['content']}")
        
        turns_text = "\n".join(full_turns)
        turns_tokens = self.model.count_tokens(turns_text)
        
        # If under limit, return as-is
        if turns_tokens <= available_tokens:
            return turns_text, 1.0
        
        # Otherwise, truncate LLM responses proportionally
        truncated_turns = []
        kept_tokens = 0
        for i, t in enumerate(turns):
            if t['role'] == 'user':
                # Keep all user content
                user_text = f"Turn {i} ({t['role']}): {t['content']}"
                truncated_turns.append(user_text)
                kept_tokens += self.model.count_tokens(user_text)
            else:
                # Truncate LLM response
                content_tokens = self.model.tokenizer.encode(t['content'])
                max_llm_tokens = int(len(content_tokens) * available_tokens / turns_tokens)
                if len(content_tokens) > max_llm_tokens:
                    truncated_content = self.model.tokenizer.decode(
                        content_tokens[:max_llm_tokens], skip_special_tokens=True
                    )
                    llm_text = f"Turn {i} ({t['role']}): {truncated_content}..."
                    truncated_turns.append(llm_text)
                    kept_tokens += self.model.count_tokens(llm_text)
                else:
                    llm_text = f"Turn {i} ({t['role']}): {t['content']}"
                    truncated_turns.append(llm_text)
                    kept_tokens += self.model.count_tokens(llm_text)
        
        content_ratio = kept_tokens / turns_tokens if turns_tokens > 0 else 1.0
        return "\n".join(truncated_turns), content_ratio

    def _build_turns_text(self, turns: List[Dict], intention: str) -> tuple[str, float]:
        """Build turns text with full content, truncating LLM responses if needed to stay under 131k tokens
        
        Returns:
            tuple: (turns_text, content_ratio) where content_ratio is the ratio of kept tokens to original tokens
        """
        from modules.logger import get_logger
        logger = get_logger()
        
        # Start with the prompt template overhead
        prompt_template = self.PROMPT_WITH_REASON if self.include_reason else self.PROMPT_WITHOUT_REASON
        overhead_tokens = self.model.count_tokens(prompt_template.replace('{turns}', '').replace('{intention}', intention))
        available_tokens = self.MAX_CONTEXT_TOKENS - overhead_tokens - 256  # Reserve for output
        
        # Build full conversation with all user content
        full_turns = []
        for i, t in enumerate(turns):
            full_turns.append(f"Turn {i} ({t['role']}): {t['content']}")
        
        turns_text = "\n".join(full_turns)
        turns_tokens = self.model.count_tokens(turns_text)
        
        # Only log token details in debug mode
        if hasattr(logger, 'debug_mode') and logger.debug_mode:
            logger.info(f"      Content: {turns_tokens} tokens (limit: {available_tokens})")
        
        # If under limit, return as-is with 100% ratio
        if turns_tokens <= available_tokens:
            if hasattr(logger, 'debug_mode') and logger.debug_mode:
                logger.info(f"      No truncation needed")
            return turns_text, 1.0
        
        # Otherwise, truncate LLM responses proportionally to fit
        logger.warning(f"      ⚠ Content exceeds limit! Truncating LLM responses...")
        truncated_turns = []
        kept_tokens = 0
        truncated_count = 0
        for i, t in enumerate(turns):
            if t['role'] == 'user':
                # Keep all user content
                user_text = f"Turn {i} ({t['role']}): {t['content']}"
                truncated_turns.append(user_text)
                kept_tokens += self.model.count_tokens(user_text)
            else:
                # Truncate LLM response
                content_tokens = self.model.tokenizer.encode(t['content'])
                # Calculate proportional truncation
                max_llm_tokens = int(len(content_tokens) * available_tokens / turns_tokens)
                if len(content_tokens) > max_llm_tokens:
                    truncated_content = self.model.tokenizer.decode(
                        content_tokens[:max_llm_tokens], skip_special_tokens=True
                    )
                    llm_text = f"Turn {i} ({t['role']}): {truncated_content}..."
                    truncated_turns.append(llm_text)
                    kept_tokens += self.model.count_tokens(llm_text)
                    truncated_count += 1
                else:
                    llm_text = f"Turn {i} ({t['role']}): {t['content']}"
                    truncated_turns.append(llm_text)
                    kept_tokens += self.model.count_tokens(llm_text)
        
        content_ratio = kept_tokens / turns_tokens if turns_tokens > 0 else 1.0
        logger.warning(f"      Truncated {truncated_count} turns: {turns_tokens} → {kept_tokens} tokens ({content_ratio:.1%})")
        return "\n".join(truncated_turns), content_ratio
    
    def _extract_json_array_verdicts(self, response: str, intentions: List[str]) -> List[Dict]:
        """Extract JSON array of verdicts from response
        
        Returns:
            List of dicts with 'verdict' field, or [{'verdict': 'unknown'}] * len(intentions) on failure
        """
        # Strategy 1: Look for <output>...</output> tags
        output_match = re.search(r'<output>(.*?)</output>', response, re.DOTALL)
        if output_match:
            json_str = output_match.group(1).strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    results = []
                    for item in data:
                        if isinstance(item, dict) and 'verdict' in item:
                            verdict = str(item['verdict']).lower()
                            if verdict in ['yes', 'partial', 'no']:
                                results.append({'verdict': verdict})
                            else:
                                results.append({'verdict': 'unknown'})
                        else:
                            results.append({'verdict': 'unknown'})
                    return results
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to find JSON array without tags
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, list):
                    results = []
                    for item in data:
                        if isinstance(item, dict) and 'verdict' in item:
                            verdict = str(item['verdict']).lower()
                            if verdict in ['yes', 'partial', 'no']:
                                results.append({'verdict': verdict})
                            else:
                                results.append({'verdict': 'unknown'})
                        else:
                            results.append({'verdict': 'unknown'})
                    return results
            except json.JSONDecodeError:
                pass
        
        # Failed - return unknown for all intentions
        return [{'verdict': 'unknown'} for _ in intentions]

    def _extract_json_verdict(self, response: str) -> Dict:
        """Robustly extract JSON from response with <o> tags
        
        Returns:
            Dict with 'verdict' and optionally 'reason' fields, or {'verdict': 'unknown'} on failure
        """
        # Strategy 1: Look for <o>...</o> tags
        output_match = re.search(r'<output>(.*?)</output>', response, re.DOTALL)
        if output_match:
            json_str = output_match.group(1).strip()
            try:
                data = json.loads(json_str)
                # Validate verdict field
                if 'verdict' in data and str(data['verdict']).lower() in ['yes', 'partial', 'no']:
                    result = {'verdict': str(data['verdict']).lower()}
                    if 'reason' in data and data['reason']:
                        result['reason'] = str(data['reason']).strip()
                    return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to find JSON without tags (fallback)
        json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if 'verdict' in data and str(data['verdict']).lower() in ['yes', 'partial', 'no']:
                    result = {'verdict': str(data['verdict']).lower()}
                    if 'reason' in data and data['reason']:
                        result['reason'] = str(data['reason']).strip()
                    return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Simple keyword matching (last resort)
        response_lower = response.lower()
        if '"partial"' in response_lower or "'partial'" in response_lower:
            return {'verdict': 'partial'}
        elif '"yes"' in response_lower:
            return {'verdict': 'yes'}
        elif '"no"' in response_lower:
            return {'verdict': 'no'}
        elif 'verdict": "partial' in response_lower or "verdict': 'partial" in response_lower:
            return {'verdict': 'partial'}
        elif 'verdict": "yes' in response_lower or "verdict': 'yes" in response_lower:
            return {'verdict': 'yes'}
        elif 'verdict": "no' in response_lower or "verdict': 'no" in response_lower:
            return {'verdict': 'no'}
        
        # Failed all strategies
        return {'verdict': 'unknown'}
    

