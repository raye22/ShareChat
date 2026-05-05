#!/usr/bin/env python3
"""
Compare Detoxify vs OpenAI Moderation toxicity ratios. Reads the per-method
summary CSVs produced by analyze_toxicity.py and emits:

  - toxicity_comparison_latex.txt — side-by-side LaTeX tables (turn + conv)
  - turn_level_user_toxicity.{pdf,png}
  - turn_level_llm_toxicity.{pdf,png}
  - conversation_level_user_toxicity.{pdf,png}
  - conversation_level_llm_toxicity.{pdf,png}

Example:
  python compare_methods.py \
      --detoxify-turn output/results/turn_level_toxicity_ratios_detoxify.csv \
      --detoxify-conv output/results/conversation_level_toxicity_ratios_detoxify.csv \
      --openai-turn   output/results/turn_level_toxicity_ratios_openai.csv \
      --openai-conv   output/results/conversation_level_toxicity_ratios_openai.csv \
      --output-dir    output/results
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Aggregated row label that analyze_toxicity.py writes
PLATFORMS = ["ChatGPT", "Claude", "Gemini", "Grok", "Perplexity"]
ROLES = ["user", "llm", "All"]


def load_summary(path: str, level: str) -> Dict[str, Dict[str, float]]:
    """
    Convert a turn- or conversation-level summary CSV into a
    {platform: {role: percentage}} nested dict.
    """
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, float]] = {p: {} for p in PLATFORMS + ["All Platforms"]}
    is_conv = level == "conv"
    n_col = "Total_Conversations" if is_conv else "Total_Turns"
    tox_col = "Toxic_Conversations" if is_conv else "Toxic_Turns"

    for p in PLATFORMS:
        sub = df[df["Platform"] == p]
        for r in ROLES:
            row = sub[sub["Role"] == r]
            if not row.empty:
                rec = row.iloc[0]
                pct = (rec[tox_col] / rec[n_col] * 100) if rec[n_col] else 0.0
                out[p][r] = float(pct)

    # Aggregated across platforms
    for r in ROLES:
        sub = df[df["Role"] == r]
        n = sub[n_col].sum()
        tox = sub[tox_col].sum()
        out["All Platforms"][r] = (tox / n * 100) if n else 0.0
    return out


def comparison_latex(detox_turn, openai_turn, detox_conv, openai_conv) -> str:
    def make_table(d_data, o_data, caption, label):
        cols = "l" + "rr" * (len(PLATFORMS) + 1)
        head1 = "\\textbf{Role}"
        for p in PLATFORMS + ["All Platforms"]:
            head1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{p}}}}}"
        head1 += " \\\\\n"
        head2 = ""
        for _ in PLATFORMS + ["All Platforms"]:
            head2 += " & \\textbf{Detox} & \\textbf{OpenAI}"
        head2 += " \\\\\n"
        cmid = "".join(
            f"\\cmidrule(lr){{{2 + i*2}-{3 + i*2}}}"
            for i in range(len(PLATFORMS) + 1)
        )

        out = (
            f"\\begin{{table}}[h]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + head1 + cmid + "\n" + head2 + "\\midrule\n"
        )
        for r in ROLES:
            row = f"\\textbf{{{r}}}"
            for p in PLATFORMS + ["All Platforms"]:
                d = d_data.get(p, {}).get(r, float("nan"))
                o = o_data.get(p, {}).get(r, float("nan"))
                row += f" & {d:.1f} & {o:.1f}"
            row += " \\\\\n"
            out += row
        out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        return out

    return (
        make_table(detox_turn, openai_turn,
                   "Turn-Level Toxicity (\\%): Detoxify vs OpenAI Moderation",
                   "tab:turn_toxicity_comparison")
        + make_table(detox_conv, openai_conv,
                     "Conversation-Level Toxicity (\\%): Detoxify vs OpenAI Moderation",
                     "tab:conv_toxicity_comparison")
    )


def bar_chart(d_data, o_data, role: str, output_path: str,
              detox_color: str, openai_color: str):
    detox = [d_data.get(p, {}).get(role, 0.0) for p in PLATFORMS]
    openai = [o_data.get(p, {}).get(role, 0.0) for p in PLATFORMS]
    x = np.arange(len(PLATFORMS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, detox, w, label="Detoxify", color=detox_color, alpha=0.8)
    b2 = ax.bar(x + w/2, openai, w, label="OpenAI", color=openai_color, alpha=0.8)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel("Toxicity percentage (%)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(PLATFORMS, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detoxify-turn", required=True)
    parser.add_argument("--detoxify-conv", required=True)
    parser.add_argument("--openai-turn", required=True)
    parser.add_argument("--openai-conv", required=True)
    parser.add_argument("--output-dir", default="output/results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    detox_turn = load_summary(args.detoxify_turn, "turn")
    detox_conv = load_summary(args.detoxify_conv, "conv")
    openai_turn = load_summary(args.openai_turn, "turn")
    openai_conv = load_summary(args.openai_conv, "conv")

    latex = comparison_latex(detox_turn, openai_turn, detox_conv, openai_conv)
    latex_path = Path(args.output_dir) / "toxicity_comparison_latex.txt"
    latex_path.write_text(latex)
    print(f"Wrote {latex_path}")

    charts = [
        ("user", "turn_level_user_toxicity.pdf",  detox_turn, openai_turn, "#BEA9D3", "#CFCFA4"),
        ("llm",  "turn_level_llm_toxicity.pdf",   detox_turn, openai_turn, "#BEA9D3", "#CFCFA4"),
        ("user", "conversation_level_user_toxicity.pdf", detox_conv, openai_conv, "#6B8E8E", "#8E9B7C"),
        ("llm",  "conversation_level_llm_toxicity.pdf",  detox_conv, openai_conv, "#6B8E8E", "#8E9B7C"),
    ]
    for role, fname, d, o, dc, oc in charts:
        path = str(Path(args.output_dir) / fname)
        bar_chart(d, o, role, path, dc, oc)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
