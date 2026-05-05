# Toxicity Analysis

Per-turn and per-conversation toxicity analysis for the ShareChat corpus.
Scores every assistant/user turn with two complementary detectors and
aggregates the results into platform x role tables and figures used in
the paper.

## Overview

This tool:
1. **Scores each turn** with multilingual [Detoxify](https://github.com/unitaryai/detoxify) (local, free) and OpenAI's `omni-moderation-latest` (API).
2. **Aggregates by role and conversation**: a turn is toxic if its score >= threshold; a conversation is toxic for a role if any turn from that role is toxic.
3. **Reports retention** when restricting Detoxify scoring to its seven supported languages.
4. **Compares the two detectors** with side-by-side LaTeX tables and bar charts.

## Setup

```bash
pip install -r requirements.txt
```

For the OpenAI scorer, put your key in a `.env` next to the scripts (or export it):

```bash
OPENAI_API_KEY=sk-...
```

## Input Format

Drop the per-platform turn CSVs from the ShareChat release (or your own
turn-level data) into `data/input/`:

```
data/input/
  ├── chatgpt_results_turn_final_grouped.csv
  ├── claude_results_turn_final.csv
  ├── gemini_results_turn_final.csv
  ├── grok_results_turn_final.csv
  └── perplexity_turn_final_with_languages.csv
```

Required columns:

| Column | Notes |
|---|---|
| `plain_text` | Message text |
| `role` | `user` or `assistant`/`llm`/`model` (auto-normalized to `llm`) |
| `filename` | Conversation id; required for conversation-level stats |
| `detected_language_final` | Optional; only needed by the Detoxify language filter |

The two scoring scripts pass `filename` straight through, so it ends up in
the score CSV and is available to [analyze_toxicity.py](analyze_toxicity.py).

## Pipeline

The scripts are independent — run only what you need.

### 1. Score with Detoxify (local, GPU recommended)

```bash
python score_detoxify.py \
    --input  data/input/chatgpt_results_turn_final_grouped.csv \
    --output output/results/chatgpt_toxicity_scores_detoxify.csv \
    --threshold 0.1 --workers 5 --device cuda
```

Per-row output columns:
`index, plain_text, is_toxic, max_toxicity_score, toxicity, severe_toxicity,
obscene, threat, insult, identity_attack, role, filename`.

### 2. Score with OpenAI Moderation (API, resumable)

```bash
python score_openai_moderation.py \
    --input  data/input/chatgpt_results_turn_final_grouped.csv \
    --output output/results/chatgpt_toxicity_scores_openai.csv \
    --workers 10 --checkpoint-interval 500
```

Resume by re-running with the same `--output`: rows already in the file
are skipped, and progress is checkpointed every `--checkpoint-interval`
completions.

### 3. Aggregate to platform x role tables

```bash
python analyze_toxicity.py \
    --method detoxify \
    --inputs ChatGPT=output/results/chatgpt_toxicity_scores_detoxify.csv \
             Claude=output/results/claude_toxicity_scores_detoxify.csv \
             Gemini=output/results/gemini_toxicity_scores_detoxify.csv \
             Grok=output/results/grok_toxicity_scores_detoxify.csv \
             Perplexity=output/results/perplexity_toxicity_scores_detoxify.csv \
    --language-files ChatGPT=data/input/chatgpt_results_turn_final_grouped.csv \
                     Claude=data/input/claude_results_turn_final.csv \
                     Gemini=data/input/gemini_results_turn_final.csv \
                     Grok=data/input/grok_results_turn_final.csv \
                     Perplexity=data/input/perplexity_turn_final_with_languages.csv \
    --output-dir output/results
```

Switch `--method openai` to aggregate the moderation outputs (no language
filter, no retention column). The script writes:

- `turn_level_toxicity_ratios_{detoxify|openai}.csv`
- `conversation_level_toxicity_ratios_{detoxify|openai}.csv`
- `toxicity_ratios_{detoxify|openai}_latex.txt`
- `{platform}_toxicity_{detoxify|openai}_processed.csv`

### 4. Compare the two detectors

```bash
python compare_methods.py \
    --detoxify-turn output/results/turn_level_toxicity_ratios_detoxify.csv \
    --detoxify-conv output/results/conversation_level_toxicity_ratios_detoxify.csv \
    --openai-turn   output/results/turn_level_toxicity_ratios_openai.csv \
    --openai-conv   output/results/conversation_level_toxicity_ratios_openai.csv \
    --output-dir    output/results
```

Produces `toxicity_comparison_latex.txt` and four PDF/PNG bar charts
(turn/conversation x user/llm).

## Methodology Notes

**Toxic-turn definition.** A turn is toxic when the chosen detector's
output meets the binary criterion: for Detoxify, `max(category_scores) >=
threshold` (default 0.1); for OpenAI moderation, `flagged == True`.

**Toxic-conversation definition.** Group rows by `filename`. A
conversation is toxic for a role if **any** turn from that role is toxic.
Conversations with `filename == "unknown"` are dropped from
conversation-level stats.

**Detoxify language filter.** The multilingual model is trained on seven
languages: English, French, Spanish, Portuguese, Italian, German,
Russian. We restrict to those for the headline numbers and report the
fraction of original turns retained ("ret.%"). OpenAI moderation runs
on all languages (no retention column).

**Role normalization.** `assistant` and `model` are mapped to `llm`
before aggregation; `user` is left as-is.

## Project Structure

```
toxicity_analysis/
├── README.md                  # This file
├── requirements.txt           # Python deps
├── .gitignore
├── score_detoxify.py          # Step 1a: local Detoxify scorer
├── score_openai_moderation.py # Step 1b: OpenAI moderation scorer (resumable)
├── analyze_toxicity.py        # Step 2: aggregate to platform x role tables
├── compare_methods.py         # Step 3: side-by-side comparison + figures
├── data/
│   └── input/                 # Per-platform turn CSVs (gitignored)
└── output/
    ├── results/               # Score CSVs, summary CSVs, LaTeX, figures
    ├── checkpoints/           # (reserved for future use)
    └── logs/                  # OpenAI scorer logs
```

## Citation

If you use this code, please cite:

```bibtex
@misc{yan2026sharechatdatasetchatbotconversations,
      title={ShareChat: A Dataset of Chatbot Conversations in the Wild},
      author={Yueru Yan and Tuc Nguyen and Bo Su and Melissa Lieffers and Thai Le},
      year={2026},
      eprint={2512.17843},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.17843},
}
```
