# SHARECHAT: A Dataset of Chatbot Conversations in the Wild

[![Paper](https://img.shields.io/badge/Paper-Arxiv%202026-blue)](https://arxiv.org/abs/2512.17843)
[![Dataset](https://img.shields.io/badge/Conversations-142%2C808-orange)](https://huggingface.co/datasets/tucnguyen/ShareChat)
[![Code License](https://img.shields.io/badge/Code-Apache%202.0-green)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-yellow)](DATA_LICENSE)

**SHARECHAT** is a large-scale corpus of authentic user-LLM conversations sourced directly from publicly shared URLs across five major chatbot platforms. Unlike existing datasets that homogenize interactions through uniform interfaces, SHARECHAT preserves native platform affordances and captures real-world usage patterns (hence, we called it "in the wild"). More detials could be found in our paper here: [ShareChat: A Dataset of Chatbot Conversations in the Wild](https://arxiv.org/abs/2512.17843). The dataset is available on Hugging Face: [ShareChat](https://huggingface.co/datasets/tucnguyen/ShareChat).

## Overview

While many existing research typically treat Large Language Models (LLMs) as generic text generators, they are often integrated as distinct commercial chatbots with unique interfaces and capabilities that fundamentally shape user behavior. Current datasets obscure this reality by collecting text-only data through uniform interfaces that fail to capture authentic human-chatbot interactions.

SHARECHAT addresses these limitations by:

- **Preserving Native Affordances**: Captures platform-specific features like citations, thinking traces, and code artifacts
- **Multi-Platform Coverage**: Spans five major platforms with distinct design philosophies
- **Authentic Usage**: Sourced from voluntarily shared conversations, reducing observer bias
- **Extended Interactions**: Substantially longer conversations than prior datasets (avg. 4.62 turns vs. 2.02 in LMSYS-Chat-1M)
- **Linguistic Diversity**: Covers 101 distinct languages

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Conversations** | 142,808 |
| **Total Turns** | 660,293 |
| **Average Turns per Conversation** | 4.62 |
| **Languages Covered** | 101 |
| **Collection Period** | April 2023 – October 2025 |
| **Avg. User Tokens** | 135.04 ± 1,820.88 |
| **Avg. Chatbot Tokens** | 1,115.30 ± 1,764.81 |

### Per-Platform Breakdown

| Platform | Conversations | Turns | Avg. Turns | Languages |
|----------|---------------|-------|------------|-----------|
| **ChatGPT** | 102,740 | 542,148 | 5.28 | 101 |
| **Perplexity** | 17,305 | 24,378 | 1.41 | 45 |
| **Grok** | 14,415 | 53,094 | 3.69 | 60 |
| **Gemini** | 7,402 | 36,422 | 4.92 | 47 |
| **Claude** | 946 | 4,251 | 4.49 | 19 |

*Token statistics computed using the Llama-2 tokenizer for consistent cross-platform comparison.*

## Data Collection

Conversations were collected from publicly shared URLs discovered via Internet archival services (Wayback Machine). 

| Platform | Share URL Format | Collection Period |
|----------|-----------------|-------------------|
| ChatGPT | `chatgpt.com/share/*` | May 2023 – Aug 2025 |
| Perplexity | `perplexity.ai/search/*` | Apr 2023 – Oct 2025 |
| Grok | `grok.com/share/*` | Dec 2024 – Oct 2025 |
| Gemini | `gemini.google.com/share/*` | Apr 2024 – Sep 2025 |
| Claude | `claude.ai/share/*` | — |

And different platforms capture distinct metadata and structural elements:

| Feature | ChatGPT | Perplexity | Grok | Gemini | Claude |
|---------|:-------:|:----------:|:----:|:------:|:------:|
| Textual Content | ✓ | ✓ | ✓ | ✓ | ✓ |
| Source Citations | – | ✓ | ✓ | – | – |
| Thinking Blocks | – | – | ✓ | – | ✓ |
| Code Artifacts | – | – | – | – | ✓ |
| Analysis Blocks | – | – | – | – | ✓ |
| Turn Timestamps | ✓ | – | ✓ | – | – |
| Model Version | ✓ | – | ✓ | ✓ | – |
| View/Share Counts | – | ✓ | – | – | – |

**IRB Approval**: Data collection conducted under IRB approval (#28569).

## Privacy and PII Removal

We prioritize user privacy through a rigorous de-identification pipeline. First, We employed **Microsoft's Presidio** as the core framework to identify and remove personally identifiable information across multiple data types:

- Names and personal identifiers
- Phone numbers
- Email addresses
- Credit card numbers
- URLs and web addresses
- Other sensitive identifiers

PII detection covers conversations in:
- English, Spanish, German, French, Italian, Portuguese, Dutch, Chinese, Japanese, Russian, and Hebre.
> **Note**: For the released dataset, we retain only conversations in the supported languages listed here and provide a separate URL list for conversations in other languages.

And then we used GPT-OSS-120B to assess the accuracy of PII identification and by verifying that PII has been successfully removed from each message. The removal success rates by platform are:

| Platform | Success Rate | Records with PII | Total Records |
|----------|-------------|------------------|---------------|
| ChatGPT | 95.20%  | 51041 | 1062949 |
| Claude | 97.01% | 252 | 8,504 | 
| Gemini | 95.43% | 3,302 | 72,746 | 
| Grok   |  94.15| 6,010 | 106,168 |
| Perplexity | 94.42% | 2,899 | 54,355 | 

Lastly, to validate detection accuracy, we manually coded 50 randomly selected conversations (288 turns) that were flagged as containing PII. We observe that the Presidio is rather conservative.

### Additional Privacy Measures

- Original platform-specific user IDs and usernames are **not stored or released**
- Analyses are conducted on aggregated statistics only

## Data Format

### Available Files

The dataset is released in **CSV format** for ease of use and accessibility.

> **Note**: Raw HTML/MHTML archives are not available in the current release.

### CSV Structure

Each conversation record contains:
- Complete sequence of user and assistant turns
- Platform-specific metadata:
  - Timestamps (ChatGPT, Grok)
  - Model version information (ChatGPT, Grok, Gemini)
  - Source citations (Perplexity, Grok)
  - Thinking traces (Claude, Grok)
 
The final released DataFrames provide turn level conversation records from five platforms with a shared core schema, where each row is one message. All datasets include `platform`, `url`, `turns_count`, `message_index`, `role`, `plain_text`, and `detected_language_final`, enabling consistent cross platform analysis of conversation structure, content, and language. Platform specific metadata is kept in additional columns: Claude includes `thinking`, `code`, `analysis`, and `version`; Gemini adds `model` plus two timestamps, `created_at` and `published_at`; Grok adds per message timing and provenance through `message_create_time`, `links`, `source`, `model`, and `last_updated`, as well as `thinking`; Perplexity adds citation and engagement context with `source_bar`, `source`, `last_updated`, `views`, `shares`, and `other_info`; and GPT includes `model` along with both a per message timestamp (`message_create_time`) and a conversation level timestamp (`create_time`).

## Caution
- You must not attempt to identify the identities of individuals or infer any sensitive personal data encompassed in this dataset.
- When leveraging direct outputs of a specific model, users must adhere to its corresponding terms of use.
- The views and opinions depicted in this dataset do not reflect the perspectives of the researchers or affiliated institutions engaged in the data collection process.

## License

This repository ships under two separate licenses:

- **Dataset (CSVs and derived statistics)** — [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](DATA_LICENSE). You may share and adapt the data for non-commercial research with attribution. Commercial use requires a separate agreement.
- **Analysis code** (this repository's `.py` files, including `conversation_completeness/` and `toxicity_analysis/`) — [Apache License 2.0](LICENSE).

In addition to the CC BY-NC terms, dataset users must comply with the privacy and re-identification clauses in [DATA_LICENSE](DATA_LICENSE) and respect the underlying chatbot platforms' own terms of use when working with their outputs.

## Analysis Code

The two analyses reported in the paper ship as self-contained subprojects with
their own READMEs, requirements, and CLI entry points:

- [conversation_completeness/](conversation_completeness/) — LLM-judged whether each user intention in a conversation was fully, partially, or not addressed by the assistant. Outputs per-platform completeness scores and the figures in Section 4.1 of the paper.
- [toxicity_analysis/](toxicity_analysis/) — Per-turn and per-conversation toxicity ratios using both Detoxify (multilingual, local) and OpenAI's `omni-moderation-latest`. Outputs the platform x role tables and bar charts in Section 3.3 of the paper.

Each subproject can be installed and run independently:

```bash
cd conversation_completeness && pip install -r requirements.txt
cd toxicity_analysis        && pip install -r requirements.txt
```

## Citation
If you use SHARECHAT in your research, please cite our paper:

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
---

## Appendix: Detailed Platform Documentation

For technical details about the data extraction process and field definitions for each platform, see the platform-specific documentation:

- [ChatGPT Scraper Documentation](docs/chatgpt_scraper_readme.md)
- [Perplexity Scraper Documentation](docs/perplexity_scraper_readme.md)
- [Grok Scraper Documentation](docs/grok_scraper_readme.md)
- [Gemini Scraper Documentation](docs/gemini_scraper_readme.md)
- [Claude Scraper Documentation](docs/claude_scraper_readme.md)
