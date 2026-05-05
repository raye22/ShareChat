#!/usr/bin/env python3
"""
Aggregate per-row toxicity scores into platform x role tables, at both turn
level and conversation level. Works with output from either:
  - score_detoxify.py      (binary indicator: `is_toxic`)
  - score_openai_moderation.py  (binary indicator: `flagged`)

Conversation-level aggregation groups by `filename` and marks a conversation
toxic if ANY turn from that role is toxic.

For Detoxify, optionally restricts to the seven languages Detoxify
multilingual was trained on, and reports the fraction of original turns
retained after that filter ("ret.%").

Outputs in --output-dir:
  turn_level_toxicity_ratios_{detoxify|openai}.csv
  conversation_level_toxicity_ratios_{detoxify|openai}.csv
  toxicity_ratios_{detoxify|openai}_latex.txt
  {platform}_toxicity_{detoxify|openai}_processed.csv

Example:
  python analyze_toxicity.py \
      --method detoxify \
      --inputs ChatGPT=output/results/chatgpt_toxicity_scores_detoxify_with_filename.csv \
               Claude=output/results/claude_toxicity_scores_detoxify_with_filename.csv \
      --language-files ChatGPT=data/input/chatgpt_results_turn_final_with_languages.csv \
                       Claude=data/input/claude_results_turn_final_with_languages.csv \
      --output-dir output/results
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DETOXIFY_SUPPORTED_LANGS = {
    "en": "English", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "de": "German", "ru": "Russian",
}


def parse_kv_args(items: List[str]) -> Dict[str, str]:
    """Parse `Platform=path` pairs into a dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected Platform=path, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def detect_toxic_column(df: pd.DataFrame, method: str) -> str:
    """Return the boolean toxic-indicator column. Method overrides auto-detect."""
    expected = "is_toxic" if method == "detoxify" else "flagged"
    if expected in df.columns:
        return expected
    if "is_toxic" in df.columns:
        return "is_toxic"
    if "flagged" in df.columns:
        return "flagged"
    raise ValueError(
        f"No toxicity indicator found. Need 'is_toxic' or 'flagged'. "
        f"Have: {list(df.columns)}"
    )


def standardize_roles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["role"] = df["role"].replace({"assistant": "llm", "model": "llm"})
    return df


def apply_language_filter(df: pd.DataFrame, lang_path: Optional[str]) -> pd.DataFrame:
    """Attach detected_language_final and keep only Detoxify-supported languages.

    If no language file is supplied, skip filtering entirely and report 100%
    retention. The filter accepts either ISO codes ('en') or full names
    ('English') in detected_language_final.
    """
    df = df.copy()

    if lang_path is None:
        print("  No language file supplied; skipping language filter")
        df.attrs["retention_percentage"] = 100.0
        df.attrs["original_count"] = len(df)
        return df

    if not os.path.exists(lang_path):
        print(f"  Language file not found: {lang_path}; skipping language filter")
        df.attrs["retention_percentage"] = 100.0
        df.attrs["original_count"] = len(df)
        return df

    try:
        lang_df = pd.read_csv(lang_path, usecols=["detected_language_final"],
                              on_bad_lines="skip")
    except Exception as e:
        print(f"  Could not read language file ({e}); skipping language filter")
        df.attrs["retention_percentage"] = 100.0
        df.attrs["original_count"] = len(df)
        return df

    if len(lang_df) != len(df):
        print(f"  Length mismatch ({len(df)} vs {len(lang_df)}); "
              f"skipping language filter")
        df.attrs["retention_percentage"] = 100.0
        df.attrs["original_count"] = len(df)
        return df

    df["detected_language_final"] = lang_df["detected_language_final"].values
    original = len(df)
    accepted = set(DETOXIFY_SUPPORTED_LANGS.keys()) | set(DETOXIFY_SUPPORTED_LANGS.values())
    norm = df["detected_language_final"].astype(str).str.strip()
    keep = norm.isin(accepted) | norm.str.lower().isin(accepted)
    df_kept = df[keep].copy()
    retention = len(df_kept) / original * 100 if original else 100.0
    df_kept.attrs["retention_percentage"] = retention
    df_kept.attrs["original_count"] = original
    print(f"  Language filter: {len(df_kept)}/{original} kept ({retention:.1f}%)")
    return df_kept


def load_one(platform: str, path: str, method: str,
             language_path: Optional[str]) -> Optional[pd.DataFrame]:
    print(f"\n=== {platform} ===")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
    if "role" not in df.columns:
        print(f"  Missing 'role' column; skipping")
        return None
    toxic_col = detect_toxic_column(df, method)
    df["toxic"] = df[toxic_col].astype(bool)
    df = standardize_roles(df)
    if "filename" not in df.columns:
        print(f"  WARNING: 'filename' column missing — conversation-level "
              f"stats will be skipped. Run add_filename.py first.")
        df["filename"] = "unknown"

    if method == "detoxify":
        df = apply_language_filter(df, language_path)
    else:
        df.attrs["retention_percentage"] = 100.0
        df.attrs["original_count"] = len(df)
    return df


def turn_stats(df: pd.DataFrame, platform: str) -> List[Dict]:
    retention = df.attrs.get("retention_percentage", 100.0)
    rows = [{
        "Platform": platform, "Role": "All",
        "Total_Turns": len(df),
        "Toxic_Turns": int(df["toxic"].sum()),
        "Toxicity_Ratio": float(df["toxic"].mean()) if len(df) else 0.0,
        "Retention_Percentage": retention,
    }]
    for role in sorted(df["role"].dropna().unique()):
        sub = df[df["role"] == role]
        rows.append({
            "Platform": platform, "Role": role,
            "Total_Turns": len(sub),
            "Toxic_Turns": int(sub["toxic"].sum()),
            "Toxicity_Ratio": float(sub["toxic"].mean()) if len(sub) else 0.0,
            "Retention_Percentage": retention,
        })
        print(f"  turn  {role}: {len(sub)} turns, {int(sub['toxic'].sum())} toxic "
              f"({sub['toxic'].mean()*100:.2f}%)")
    return rows


def conv_stats(df: pd.DataFrame, platform: str) -> List[Dict]:
    retention = df.attrs.get("retention_percentage", 100.0)
    df_known = df[df["filename"] != "unknown"]
    if len(df_known) == 0:
        print(f"  No known filenames; skipping conversation-level stats")
        return []
    n_total = df_known["filename"].nunique()
    toxic_per_conv = df_known.groupby("filename")["toxic"].max()
    rows = [{
        "Platform": platform, "Role": "All",
        "Total_Conversations": n_total,
        "Toxic_Conversations": int(toxic_per_conv.sum()),
        "Toxicity_Ratio": (int(toxic_per_conv.sum()) / n_total) if n_total else 0.0,
        "Retention_Percentage": retention,
    }]
    for role in sorted(df_known["role"].dropna().unique()):
        sub = df_known[df_known["role"] == role]
        per_conv = sub.groupby("filename")["toxic"].max()
        n = len(per_conv)
        toxic = int(per_conv.sum())
        rows.append({
            "Platform": platform, "Role": role,
            "Total_Conversations": n,
            "Toxic_Conversations": toxic,
            "Toxicity_Ratio": (toxic / n) if n else 0.0,
            "Retention_Percentage": retention,
        })
        print(f"  conv  {role}: {n} conversations, {toxic} toxic "
              f"({(toxic/n*100 if n else 0):.2f}%)")
    return rows


def latex_table(stats_df: pd.DataFrame, caption: str, label: str,
                is_conversation: bool, include_retention: bool) -> str:
    platforms = sorted(stats_df["Platform"].unique())
    roles = ["user", "llm", "All"]
    width = 3 if include_retention else 2
    col_spec = "l" + ("rrr" if include_retention else "rr") * (len(platforms) + 1)

    head1 = "\\textbf{Role}"
    for p in platforms:
        head1 += f" & \\multicolumn{{{width}}}{{c}}{{\\textbf{{{p}}}}}"
    head1 += f" & \\multicolumn{{{width}}}{{c}}{{\\textbf{{All Platforms}}}} \\\\\n"

    head2 = ""
    for _ in platforms + ["All"]:
        head2 += " & \\textbf{n} & \\textbf{\\%}"
        if include_retention:
            head2 += " & \\textbf{ret.\\%}"
    head2 += " \\\\\n"

    cmid = []
    for i in range(len(platforms) + 1):
        a = 2 + i * width
        b = a + width - 1
        cmid.append(f"\\cmidrule(lr){{{a}-{b}}}")

    out = (
        f"\\begin{{table}}[h]\n\\centering\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
        + head1 + "".join(cmid) + "\n" + head2 + "\\midrule\n"
    )

    # Aggregated stats per role across platforms
    agg = {}
    for role in roles:
        sub = stats_df[stats_df["Role"] == role]
        if is_conversation:
            tot_tox = int(sub["Toxic_Conversations"].sum())
            tot_n = int(sub["Total_Conversations"].sum())
            weights = sub["Total_Conversations"]
        else:
            tot_tox = int(sub["Toxic_Turns"].sum())
            tot_n = int(sub["Total_Turns"].sum())
            weights = sub["Total_Turns"]
        pct = (tot_tox / tot_n * 100) if tot_n else 0.0
        ret = (np.average(sub["Retention_Percentage"], weights=weights)
               if len(weights) and weights.sum() > 0 else 0.0)
        agg[role] = (tot_tox, pct, ret)

    for role in roles:
        row = f"\\textbf{{{role}}}"
        for p in platforms:
            sub = stats_df[(stats_df["Platform"] == p) & (stats_df["Role"] == role)]
            if not sub.empty:
                r = sub.iloc[0]
                if is_conversation:
                    tox, n = int(r["Toxic_Conversations"]), int(r["Total_Conversations"])
                else:
                    tox, n = int(r["Toxic_Turns"]), int(r["Total_Turns"])
                pct = (tox / n * 100) if n else 0.0
                row += f" & {tox:,} & {pct:.1f}"
                if include_retention:
                    row += f" & {r['Retention_Percentage']:.1f}"
            else:
                row += " & - & -"
                if include_retention:
                    row += " & -"
        tox, pct, ret = agg[role]
        row += f" & {tox:,} & {pct:.1f}"
        if include_retention:
            row += f" & {ret:.1f}"
        row += " \\\\\n"
        out += row

    out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True, choices=["detoxify", "openai"],
                        help="Which scorer produced the inputs.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Platform=path pairs for toxicity score CSVs.")
    parser.add_argument("--language-files", nargs="*",
                        help="Optional Platform=path pairs for the per-platform "
                             "turn CSVs that carry detected_language_final. "
                             "Used by --method detoxify to filter to the seven "
                             "supported languages and report retention.")
    parser.add_argument("--output-dir", default="output/results",
                        help="Directory for CSV summaries and LaTeX (default: output/results).")
    args = parser.parse_args()

    inputs = parse_kv_args(args.inputs)
    lang_inputs = parse_kv_args(args.language_files or [])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_turn, all_conv = [], []
    for platform, path in inputs.items():
        df = load_one(platform, path, args.method, lang_inputs.get(platform))
        if df is None:
            continue
        all_turn.extend(turn_stats(df, platform))
        all_conv.extend(conv_stats(df, platform))

        proc_path = os.path.join(
            args.output_dir,
            f"{platform.lower()}_toxicity_{args.method}_processed.csv"
        )
        df.to_csv(proc_path, index=False)
        print(f"  wrote {proc_path}")

    if not all_turn:
        print("\nNo turn statistics produced. Exiting.")
        return

    suffix = args.method
    turn_df = pd.DataFrame(all_turn)
    conv_df = pd.DataFrame(all_conv)
    turn_csv = os.path.join(args.output_dir, f"turn_level_toxicity_ratios_{suffix}.csv")
    conv_csv = os.path.join(args.output_dir, f"conversation_level_toxicity_ratios_{suffix}.csv")
    turn_df.to_csv(turn_csv, index=False)
    print(f"\nWrote {turn_csv}")
    if not conv_df.empty:
        conv_df.to_csv(conv_csv, index=False)
        print(f"Wrote {conv_csv}")

    include_retention = (args.method == "detoxify")
    latex = latex_table(
        turn_df,
        f"Turn-Level Toxicity by Platform and Role ({suffix.capitalize()})",
        f"tab:turn_toxicity_ratios_{suffix}",
        is_conversation=False,
        include_retention=include_retention,
    )
    if not conv_df.empty:
        latex += latex_table(
            conv_df,
            f"Conversation-Level Toxicity by Platform and Role ({suffix.capitalize()})",
            f"tab:conv_toxicity_ratios_{suffix}",
            is_conversation=True,
            include_retention=include_retention,
        )
    latex_path = os.path.join(args.output_dir, f"toxicity_ratios_{suffix}_latex.txt")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
