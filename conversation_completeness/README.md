# Conversation Completeness Analysis

Analyze whether AI assistant conversations fully address user intentions using LLM-based evaluation.

## Overview

This tool:
1. **Extracts user intentions** from conversations using Claude
2. **Evaluates if each intention was addressed** by checking the assistant's responses
3. **Computes completeness scores** at both intention and conversation levels
4. **Supports multiple platforms**: ChatGPT, Claude, Gemini, Grok, Perplexity

## Features

- 🔄 Asynchronous processing for high throughput
- 💾 Checkpoint/resume support for long-running analyses
- 📊 Built-in visualization and statistical analysis
- 🎯 Sampling utilities for validation studies
- 📈 Token usage tracking and cost estimation

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- `anthropic` - For Claude API access
- `pandas` - Data manipulation
- `pyyaml` - Configuration
- `matplotlib`, `seaborn` - Visualization
- `tqdm` - Progress tracking

### API Keys

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Or set the environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Configuration

Edit `config.yaml` to adjust:
- Model settings (default: claude-3-5-sonnet-20241022)
- API rate limits
- Temperature and token limits
- Output directories

### Data Structure

Place your conversation data in `data/input/`:

```
data/input/
  ├── chatgpt_results_turn_final.csv
  ├── claude_results_turn_final.csv
  ├── gemini_results_turn_final.csv
  ├── grok_results_turn_final.csv
  └── perplexity_turn_final.csv
```

**Required CSV columns:**
- `file_name` or `conv_id`: Unique conversation identifier
- `turn_index` or `message_index`: Turn number (0-indexed or 1-indexed)
- `role`: Message role ('user' or 'assistant'/'llm')
- `plain_text` or `content`: Message content

## Usage

### Quick Test

Test the pipeline with a few sample conversations:

```bash
python main.py --test --num-samples 5
```

### Full Analysis

Analyze all conversations for a specific platform:

```bash
python main.py --platform chatgpt
```

Analyze all platforms:

```bash
python main.py --platform all
```

### Resume from Checkpoint

If analysis is interrupted, resume from the last checkpoint:

```bash
python main.py --platform chatgpt --resume
```

### Limit Number of Conversations

Process only the first N conversations:

```bash
python main.py --platform chatgpt --limit 1000
```

### Sample and Visualize

After running the full analysis, generate visualizations:

```bash
# Analyze completeness results
python analyze_completeness.py

# Sample conversations for manual review
python sample_conversations.py
```

## Output Files

### Main Results

**`output/results/{platform}_completeness.jsonl`**

One JSON object per line, each representing a conversation:

```json
{
  "conv_id": "conversation_123.json",
  "platform": "chatgpt",
  "num_turns": 10,
  "num_intentions": 3,
  "intentions": [
    "Explain quantum computing",
    "Compare it to classical computing",
    "Provide real-world applications"
  ],
  "verdicts": [
    {
      "intention": "Explain quantum computing",
      "verdict": "fully_addressed",
      "content_ratio": 0.85,
      "evidence": "The assistant provided a comprehensive explanation..."
    },
    {
      "intention": "Compare it to classical computing",
      "verdict": "partially_addressed",
      "content_ratio": 0.40,
      "evidence": "Brief comparison was mentioned but not detailed..."
    },
    {
      "intention": "Provide real-world applications",
      "verdict": "fully_addressed",
      "content_ratio": 0.75,
      "evidence": "Multiple applications were discussed..."
    }
  ],
  "completeness_score": 0.833,
  "processing_time_seconds": 12.5,
  "tokens_used": {
    "input": 1234,
    "output": 567
  }
}
```

### Sampled Conversations

**`output/results/sampled_conversations_with_text.csv`**

20 randomly sampled conversations per platform with:
- Full conversation text
- All extracted intentions
- Verdicts for each intention
- Completeness scores

Useful for manual validation and qualitative analysis.

### Visualizations

**`output/results/`**
- `verdict_distribution.png` - Stacked bar chart of verdict proportions
- `completeness_scores.png` - Box plot of completeness scores
- `num_intentions.png` - Box plot of intentions per conversation
- `platform_statistics.csv` - Summary statistics table

### Checkpoints

**`output/checkpoints/{platform}/`**
- Saves progress every N conversations
- Resume interrupted runs without reprocessing
- Includes processed conversation IDs and results

## Completeness Scoring

### Verdict Types

- **`fully_addressed`**: Intention completely addressed (score: 1.0)
- **`partially_addressed`**: Intention partially addressed (score: 0.5)
- **`not_addressed`**: Intention not addressed (score: 0.0)
- **`error`**: Processing error (excluded from scoring)

### Conversation-Level Score

Average of all intention-level scores:

```
completeness_score = (Σ intention_scores) / num_intentions
```

Example:
- 2 fully_addressed (1.0 each)
- 1 partially_addressed (0.5)
- Score = (1.0 + 1.0 + 0.5) / 3 = 0.833

## Project Structure

```
conversation_completeness_public/
├── main.py                     # Main entry point
├── pipeline.py                 # Processing pipeline
├── analyze_completeness.py     # Statistical analysis & visualization
├── sample_conversations.py     # Sampling utility
├── count_token_statistics.py   # Token usage analysis
├── config.yaml                 # Configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── calculator.py           # Completeness score calculation
│   ├── checkpoint.py           # Checkpoint management
│   ├── data_loader.py          # CSV data loading
│   ├── evaluator.py            # Verdict evaluation
│   ├── extractor.py            # Intention extraction
│   ├── logger.py               # Logging utilities
│   └── model_wrapper.py        # Claude API wrapper
├── data/
│   └── input/                  # Your conversation CSV files
└── output/
    ├── results/                # Analysis results (JSONL, CSV, PNG)
    ├── checkpoints/            # Resume checkpoints
    └── logs/                   # Execution logs
```

## Advanced Usage

### Custom Configuration

Override config settings via command line:

```bash
# Use a different model
python main.py --platform chatgpt --model claude-3-opus-20240229

# Adjust batch size for rate limiting
python main.py --platform all --batch-size 50
```

### Debugging

Enable verbose logging:

```bash
python main.py --test --debug
```

This will:
- Print detailed API requests/responses
- Show intermediate processing steps
- Save full prompt/response pairs

### Token Usage Analysis

Analyze token consumption and estimate costs:

```bash
python count_token_statistics.py
```

Outputs:
- Total tokens per platform
- Cost estimates (input/output tokens)
- Average tokens per conversation

## Evaluation Methodology

### Intention Extraction

Uses Claude with a structured prompt to:
1. Read the full conversation
2. Identify distinct user intentions
3. Output as a JSON list

**Prompt strategy**: Few-shot learning with examples

### Verdict Evaluation

For each intention:
1. Extract relevant assistant responses
2. Prompt Claude to judge if intention was addressed
3. Classify as fully/partially/not addressed
4. Calculate content coverage ratio

**Quality controls**:
- Retry logic for API failures
- JSON schema validation
- Fallback parsing for malformed responses

## Limitations

- Requires Claude API access (costs apply)
- Processing speed limited by API rate limits
- Subjective evaluation dependent on LLM judgment
- Works best for English conversations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{conversation_completeness,
  title = {Conversation Completeness Analysis},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]}
}
```

## License

[Specify your license - e.g., MIT, Apache 2.0]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## Support

For questions or issues:
- Open a GitHub issue
- Contact: [your-email]

## Acknowledgments

- Claude API by Anthropic for LLM evaluation
- Inspired by conversation quality research in HCI/NLP
