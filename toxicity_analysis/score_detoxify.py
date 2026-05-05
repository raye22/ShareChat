#!/usr/bin/env python3
"""
Score every turn in a per-platform turn CSV with the multilingual Detoxify
classifier and write a per-row CSV of toxicity scores.

Input CSV columns required:
  - plain_text   (str)
  - role         (str, optional; defaults to 'unknown')
  - filename     (str, optional; conversation id used by analyze_toxicity.py)

Output CSV columns:
  index, plain_text, is_toxic, max_toxicity_score,
  toxicity, severe_toxicity, obscene, threat, insult, identity_attack,
  role, filename

Example:
  python score_detoxify.py \
      --input data/input/chatgpt_results_turn_final.csv \
      --output output/results/chatgpt_toxicity_scores_detoxify.csv \
      --threshold 0.1 --workers 5 --device cuda
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm


CATEGORIES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def empty_scores() -> Dict[str, float]:
    return {c: 0.0 for c in CATEGORIES}


def get_toxicity_score(model: Detoxify, text: str) -> Dict[str, float]:
    if not text or pd.isna(text):
        return empty_scores()
    try:
        results = model.predict(str(text))
        return {c: float(results.get(c, 0.0)) for c in CATEGORIES}
    except Exception as e:
        print(f"Error scoring text: {e}")
        scores = empty_scores()
        scores["error"] = str(e)
        return scores


def process_row(model: Detoxify, idx: int, text: str, role: str, filename: str,
                threshold: float) -> Dict:
    scores = get_toxicity_score(model, text)
    max_score = max(scores.get(c, 0.0) for c in CATEGORIES)
    return {
        "index": idx,
        "plain_text": text,
        "is_toxic": max_score >= threshold,
        "max_toxicity_score": max_score,
        **{c: scores.get(c, 0.0) for c in CATEGORIES},
        "role": role or "unknown",
        "filename": filename or "unknown",
    }


def score_file(input_path: str, output_path: str, threshold: float,
               workers: int, device: str, model_variant: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "plain_text" not in df.columns:
        raise ValueError(
            f"'plain_text' column missing from {input_path}. "
            f"Found: {df.columns.tolist()}"
        )

    has_role = "role" in df.columns
    has_filename = "filename" in df.columns

    print(f"Loading Detoxify ({model_variant}) on {device}...")
    model = Detoxify(model_variant, device=device)

    rows = [
        (
            idx,
            row.get("plain_text", ""),
            row.get("role", "unknown") if has_role else "unknown",
            row.get("filename", "unknown") if has_filename else "unknown",
        )
        for idx, row in df.iterrows()
    ]

    print(f"Scoring {len(rows)} rows with {workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_row, model, i, t, r, f, threshold): i
            for i, t, r, f in rows
        }
        with tqdm(total=len(futures), desc="Detoxify", unit="texts") as pbar:
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"Row error: {e}")
                finally:
                    pbar.update(1)

    results.sort(key=lambda r: r["index"])
    out_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} rows to {output_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Path to per-platform turn CSV.")
    parser.add_argument("--output", required=True,
                        help="Path to write the toxicity score CSV.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Score threshold for the binary is_toxic label (default: 0.1).")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of ThreadPool workers (default: 5).")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for Detoxify inference (default: cuda).")
    parser.add_argument("--model", default="multilingual",
                        choices=["original", "unbiased", "multilingual"],
                        help="Detoxify model variant (default: multilingual).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    score_file(args.input, args.output, args.threshold,
               args.workers, args.device, args.model)


if __name__ == "__main__":
    main()
