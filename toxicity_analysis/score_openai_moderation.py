#!/usr/bin/env python3
"""
Score every turn in a per-platform turn CSV with OpenAI's moderation API
(`omni-moderation-latest`) and write a per-row CSV of toxicity scores.

Features:
  - Granular checkpointing: rows already present in --output are skipped on re-run
  - Periodic incremental writes so progress survives a crash
  - Rate-limit-aware retries

Input CSV columns required:
  - plain_text   (str)
  - role         (str, optional; defaults to 'unknown')
  - filename     (str, optional; conversation id used by analyze_toxicity.py)

Output CSV columns:
  index, plain_text, toxicity_score, flagged, categories, role, filename

`OPENAI_API_KEY` is read from the environment (or a local .env via python-dotenv).

Example:
  python score_openai_moderation.py \
      --input data/input/chatgpt_results_turn_final.csv \
      --output output/results/chatgpt_toxicity_scores_openai.csv \
      --workers 10 --checkpoint-interval 500
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f'toxicity_openai_{time.strftime("%Y%m%d-%H%M%S")}.log')

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("toxicity_openai")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_moderation(client: OpenAI, text: str, logger: logging.Logger,
                   max_retries: int = 3, wait_seconds: int = 60) -> Dict:
    if not text or pd.isna(text):
        return {"flagged": False, "categories": {}, "category_scores": {}}

    attempt = 0
    while attempt < max_retries:
        try:
            response = client.moderations.create(
                input=str(text), model="omni-moderation-latest"
            )
            r = response.results[0]
            return {
                "flagged": r.flagged,
                "categories": r.categories.model_dump(),
                "category_scores": r.category_scores.model_dump(),
            }
        except Exception as e:
            err = str(e).lower()
            is_rate = any(s in err for s in
                          ["rate limit", "rate_limit", "too many requests", "429"])
            attempt += 1
            if is_rate and attempt < max_retries:
                logger.warning(
                    f"Rate limit hit, sleeping {wait_seconds}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(wait_seconds)
                continue
            logger.error(f"Moderation error (attempt {attempt}/{max_retries}): {e}")
            return {"flagged": False, "categories": {}, "category_scores": {},
                    "error": str(e)}


def highest_score(category_scores: Dict) -> float:
    if not category_scores:
        return 0.0
    try:
        return max(float(v) for v in category_scores.values())
    except Exception:
        return 0.0


def process_row(client: OpenAI, logger: logging.Logger, idx: int, text: str,
                role: str, filename: str) -> Dict:
    res = get_moderation(client, text, logger)
    return {
        "index": idx,
        "plain_text": text,
        "toxicity_score": highest_score(res.get("category_scores", {})),
        "flagged": res.get("flagged", False),
        "categories": json.dumps(res.get("categories", {})),
        "role": role or "unknown",
        "filename": filename or "unknown",
    }


def load_processed_indices(output_path: str, logger: logging.Logger) -> Set[int]:
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path)
        processed = set(df["index"].tolist())
        logger.info(f"Resuming: {len(processed)} rows already in {output_path}")
        return processed
    except Exception as e:
        logger.warning(f"Could not read existing output: {e}")
        return set()


def write_combined(existing: Optional[pd.DataFrame], new_rows: list,
                   output_path: str) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing is not None and len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["index"], keep="last")
    else:
        combined = new_df
    combined = combined.sort_values("index").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    return combined


def score_file(input_path: str, output_path: str, workers: int,
               checkpoint_interval: int, logger: logging.Logger):
    df = pd.read_csv(input_path)
    if "plain_text" not in df.columns:
        raise ValueError(
            f"'plain_text' missing from {input_path}. Found: {df.columns.tolist()}"
        )
    has_role = "role" in df.columns
    has_filename = "filename" in df.columns

    processed = load_processed_indices(output_path, logger)
    existing = pd.read_csv(output_path) if os.path.exists(output_path) else None

    df_remaining = df[~df.index.isin(processed)]
    if len(df_remaining) == 0:
        logger.info(f"All {len(df)} rows already processed.")
        return existing

    logger.info(
        f"Scoring {len(df_remaining)} remaining rows "
        f"(skipping {len(processed)} already done) with {workers} workers."
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows_to_process = [
        (
            idx,
            row.get("plain_text", ""),
            row.get("role", "unknown") if has_role else "unknown",
            row.get("filename", "unknown") if has_filename else "unknown",
        )
        for idx, row in df_remaining.iterrows()
    ]

    new_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_row, client, logger, i, t, r, f): i
            for i, t, r, f in rows_to_process
        }
        with tqdm(total=len(futures), desc="OpenAI", unit="texts") as pbar:
            count = 0
            for fut in as_completed(futures):
                try:
                    new_rows.append(fut.result())
                    count += 1
                    if count % checkpoint_interval == 0:
                        write_combined(existing, new_rows, output_path)
                        logger.info(f"Checkpoint: wrote {count} new rows to {output_path}")
                except Exception as e:
                    logger.error(f"Row error: {e}")
                finally:
                    pbar.update(1)

    final = write_combined(existing, new_rows, output_path)
    logger.info(f"Done. Total rows in {output_path}: {len(final)}")
    return final


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="Path to per-platform turn CSV.")
    parser.add_argument("--output", required=True,
                        help="Path to write moderation score CSV (resumed if it exists).")
    parser.add_argument("--workers", type=int, default=10,
                        help="ThreadPool workers for API calls (default: 10).")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Write incremental CSV after every N completed rows.")
    parser.add_argument("--log-dir", default="output/logs",
                        help="Directory for run logs (default: output/logs).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; export it or put it in a .env file")
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    logger = setup_logger(args.log_dir)
    score_file(args.input, args.output, args.workers,
               args.checkpoint_interval, logger)


if __name__ == "__main__":
    main()
