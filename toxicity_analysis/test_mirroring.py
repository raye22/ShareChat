#!/usr/bin/env python3
"""
Test the user-LLM toxicity "mirroring" hypothesis at the conversation level.

For each platform separately:
  1. Load per-turn toxicity scores produced by score_detoxify.py or
     score_openai_moderation.py.
  2. For each conversation (grouped by `filename`), compute the mean
     user-turn toxicity score and the mean LLM-turn toxicity score.
  3. Across all conversations in that platform, compute the Spearman
     and Pearson correlation between the two per-conversation means.

This replaces the n=5 platform-aggregate Spearman test with a
within-platform conversation-level test that has real statistical power
(n in the hundreds to hundreds of thousands per platform).

Bonferroni correction is applied across the platforms tested.

Example:
  python test_mirroring.py \\
      --method detoxify \\
      --inputs ChatGPT=output/results/chatgpt_toxicity_scores_detoxify_with_filename.csv \\
               Claude=output/results/claude_toxicity_scores_detoxify_with_filename.csv \\
               Gemini=output/results/gemini_toxicity_scores_detoxify_with_filename.csv \\
               Grok=output/results/grok_toxicity_scores_detoxify_with_filename.csv \\
               Perplexity=output/results/perplexity_toxicity_scores_detoxify_with_filename.csv \\
      --output-dir output/results
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats


def parse_kv_args(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected Platform=path, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def detect_score_column(df: pd.DataFrame, method: str) -> str:
    """Pick the continuous toxicity score column for the chosen method."""
    if method == "detoxify":
        if "max_toxicity_score" in df.columns:
            return "max_toxicity_score"
        if "toxicity" in df.columns:
            return "toxicity"
    if method == "openai":
        if "toxicity_score" in df.columns:
            return "toxicity_score"
    # Fallback: try whichever exists.
    for c in ("max_toxicity_score", "toxicity_score", "toxicity"):
        if c in df.columns:
            return c
    raise ValueError(
        f"No toxicity score column found. Need one of "
        f"max_toxicity_score / toxicity_score / toxicity. "
        f"Columns present: {list(df.columns)}"
    )


def per_conversation_means(path: str, method: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(path)
    if "filename" not in df.columns or "role" not in df.columns:
        print(f"  SKIP: {path} missing 'filename' or 'role'.")
        return None

    score_col = detect_score_column(df, method)
    df = df[df["filename"].fillna("unknown") != "unknown"].copy()
    df["role"] = df["role"].replace({"assistant": "llm", "model": "llm"})
    df = df[df["role"].isin(["user", "llm"])]

    # Mean score per (conversation, role).
    pivot = (
        df.groupby(["filename", "role"])[score_col]
          .mean()
          .unstack(fill_value=None)
    )
    if "user" not in pivot.columns or "llm" not in pivot.columns:
        print(f"  SKIP: missing user or llm rows after filtering.")
        return None

    paired = pivot[["user", "llm"]].dropna()
    paired.attrs["score_col"] = score_col
    return paired


def test_platform(paired: pd.DataFrame, name: str) -> Dict:
    n = len(paired)
    if n < 5:
        return {"platform": name, "n": n, "spearman_rho": None,
                "spearman_p": None, "pearson_r": None, "pearson_p": None,
                "score_col": paired.attrs.get("score_col")}
    rho, p_s = stats.spearmanr(paired["user"], paired["llm"])
    r, p_p = stats.pearsonr(paired["user"], paired["llm"])
    return {
        "platform": name, "n": n,
        "spearman_rho": float(rho), "spearman_p": float(p_s),
        "pearson_r": float(r), "pearson_p": float(p_p),
        "score_col": paired.attrs.get("score_col"),
    }


def bonferroni(p_values: List[float]) -> List[float]:
    k = len([p for p in p_values if p is not None])
    return [min(p * k, 1.0) if p is not None else None for p in p_values]


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "-"
    if p < 1e-300:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def fmt_corr(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def latex_table(rows: List[Dict], method: str) -> str:
    out = (
        "\\begin{table}[h]\n\\centering\n"
        f"\\caption{{Within-platform conversation-level mirroring test "
        f"({method.capitalize()}). For each conversation we compute the mean "
        f"toxicity score over user turns and over LLM turns; we then correlate "
        f"these two per-conversation means across all conversations in a "
        f"platform. Bonferroni-corrected $p$-values account for the 5 "
        f"platforms tested.}}\n"
        f"\\label{{tab:mirroring_{method}}}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "\\textbf{Platform} & \\textbf{n (convs)} & "
        "\\textbf{Spearman $\\rho$} & \\textbf{$p_{\\mathrm{raw}}$} & "
        "\\textbf{$p_{\\mathrm{Bonf}}$} & \\textbf{Pearson $r$} \\\\\n"
        "\\midrule\n"
    )
    for r in rows:
        out += (
            f"{r['platform']} & {r['n']:,} & {fmt_corr(r['spearman_rho'])} & "
            f"{fmt_p(r['spearman_p'])} & {fmt_p(r['spearman_p_bonf'])} & "
            f"{fmt_corr(r['pearson_r'])} \\\\\n"
        )
    out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True, choices=["detoxify", "openai"])
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Platform=path pairs for per-turn toxicity CSVs.")
    parser.add_argument("--output-dir", default="output/results")
    args = parser.parse_args()

    inputs = parse_kv_args(args.inputs)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for platform, path in inputs.items():
        print(f"\n=== {platform} ===")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        paired = per_conversation_means(path, args.method)
        if paired is None or len(paired) < 5:
            print(f"  Insufficient paired conversations ({0 if paired is None else len(paired)})")
            continue
        result = test_platform(paired, platform)
        print(f"  n={result['n']:,} conversations with both user and LLM turns")
        print(f"  score column: {result['score_col']}")
        print(f"  Spearman rho={result['spearman_rho']:.4f}  p={result['spearman_p']:.3e}")
        print(f"  Pearson  r  ={result['pearson_r']:.4f}  p={result['pearson_p']:.3e}")
        rows.append(result)

    if not rows:
        print("\nNo platforms produced results. Exiting.")
        return

    # Bonferroni correction across platforms.
    bonf_s = bonferroni([r["spearman_p"] for r in rows])
    bonf_p = bonferroni([r["pearson_p"] for r in rows])
    for r, bs, bp in zip(rows, bonf_s, bonf_p):
        r["spearman_p_bonf"] = bs
        r["pearson_p_bonf"] = bp

    # Save CSV summary.
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f"mirroring_test_{args.method}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Save LaTeX table.
    latex_path = os.path.join(args.output_dir, f"mirroring_test_{args.method}_latex.txt")
    with open(latex_path, "w") as f:
        f.write(latex_table(rows, args.method))
    print(f"Wrote {latex_path}")

    # Quick prose summary for the paper.
    rhos = [r["spearman_rho"] for r in rows if r["spearman_rho"] is not None]
    sig = sum(1 for r in rows if r["spearman_p_bonf"] is not None
              and r["spearman_p_bonf"] < 0.05)
    print(
        f"\nSummary: {sig}/{len(rows)} platforms show a Bonferroni-significant "
        f"positive Spearman correlation between per-conversation mean user "
        f"toxicity and per-conversation mean LLM toxicity. "
        f"rho range: {min(rhos):.3f} to {max(rhos):.3f}."
    )


if __name__ == "__main__":
    main()
