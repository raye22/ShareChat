#!/usr/bin/env python3
"""
Conversation Completeness Analysis and Visualization

This script analyzes conversation completeness data across five platforms
(chatgpt, claude, gemini, grok, perplexity) and generates three visualizations:
1. Verdict distribution by platform (horizontal stacked bar chart)
2. Completeness scores by platform (boxplot)
3. Number of intentions per conversation by platform (boxplot)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_FILE = "./conversation_completeness/output/results/all_platforms_completeness.jsonl"
OUTPUT_DIR = Path("./conversation_completeness/output/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Platform order and display names
PLATFORM_ORDER = ['chatgpt', 'claude', 'gemini', 'grok', 'perplexity']
PLATFORM_DISPLAY = {
    'chatgpt': 'ChatGPT',
    'claude': 'Claude',
    'gemini': 'Gemini',
    'grok': 'Grok',
    'perplexity': 'Perplexity'
}

# Color palette for platforms (muted, academic)
PLATFORM_COLORS = {
    'chatgpt': '#5C7A8C',
    'claude': '#8C6C5C',
    'gemini': '#6C7A8C',
    'grok': '#7C6C8C',
    'perplexity': '#6C8C7C'
}

# Verdict colors for stacked bar chart (muted)
VERDICT_COLORS = {
    'yes': '#7CAF7C',      # Muted green
    'partial': '#D4A574',  # Muted yellow-orange
    'no': '#C47C7C',       # Muted red
    'unknown': '#999999'   # Gray
}


def load_data(filepath):
    """Load JSONL data and return as list of dictionaries"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} conversations from {filepath}")
    return data


def extract_intention_verdicts(data):
    """
    Extract intention-level verdicts for Plot 1
    
    Returns:
        DataFrame with columns: platform, verdict
    """
    records = []
    for conv in data:
        platform = conv.get('platform')
        verdicts = conv.get('verdicts', [])
        
        for verdict_obj in verdicts:
            verdict = verdict_obj.get('verdict')
            if verdict and platform:
                records.append({
                    'platform': platform,
                    'verdict': verdict
                })
    
    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} intention-level verdicts")
    print(f"Verdict value counts:\n{df['verdict'].value_counts()}")
    return df


def extract_completeness_scores(data):
    """
    Extract conversation-level completeness scores for Plot 2
    
    Returns:
        DataFrame with columns: platform, completeness_score
    """
    records = []
    for conv in data:
        platform = conv.get('platform')
        score = conv.get('completeness_score')
        
        if platform and score is not None:
            records.append({
                'platform': platform,
                'completeness_score': score
            })
    
    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} conversations with completeness scores")
    return df


def extract_intention_counts(data):
    """
    Extract conversation-level intention counts for Plot 3
    
    Returns:
        DataFrame with columns: platform, num_intentions
    """
    records = []
    for conv in data:
        platform = conv.get('platform')
        intentions = conv.get('intentions', [])
        num_intentions = len(intentions)
        
        if platform and num_intentions > 0:
            records.append({
                'platform': platform,
                'num_intentions': num_intentions
            })
    
    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} conversations with intentions")
    return df


def plot_verdict_distribution(df_verdicts, output_path):
    """
    Plot 1: Horizontal 100% stacked bar chart of verdict distribution by platform
    """
    # Count verdicts by platform
    verdict_counts = df_verdicts.groupby(['platform', 'verdict']).size().unstack(fill_value=0)
    
    # Ensure all verdict categories are present
    for verdict in ['yes', 'partial', 'no', 'unknown']:
        if verdict not in verdict_counts.columns:
            verdict_counts[verdict] = 0
    
    # Reorder columns
    verdict_counts = verdict_counts[['yes', 'partial', 'no', 'unknown']]
    
    # Reorder rows by PLATFORM_ORDER
    verdict_counts = verdict_counts.reindex(PLATFORM_ORDER)
    
    # Calculate percentages
    verdict_pcts = verdict_counts.div(verdict_counts.sum(axis=1), axis=0) * 100
    
    # Rename index to display names
    verdict_pcts.index = [PLATFORM_DISPLAY[p] for p in verdict_pcts.index]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create horizontal stacked bar chart with minimal spacing
    verdict_pcts.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=[VERDICT_COLORS[v] for v in verdict_pcts.columns],
        width=0.8
    )
    
    # Formatting
    # ax.set_xlabel('Percentage (%)', fontsize=14)
    # ax.set_ylabel('Platform', fontsize=14)
    # ax.set_title('Verdict distribution by platform', fontsize=12, pad=15)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Legend - one line under the plot
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center',
              frameon=False, fontsize=18, ncol=4, title_fontsize=14)
    
    # Add percentage labels on bars
    for i, platform in enumerate(verdict_pcts.index):
        cumulative = 0
        for verdict in verdict_pcts.columns:
            pct = verdict_pcts.iloc[i][verdict]
            if pct >= 3:  # Show label if segment is >= 3%
                ax.text(cumulative + pct/2, i, f'{pct:.0f}%',
                       ha='center', fontweight='bold', va='center', fontsize=15,
                       color='black')
            cumulative += pct
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved Plot 1 to: {output_path}")
    print(f"Saved Plot 1 (PDF) to: {str(output_path).replace('.png', '.pdf')}")
    
    # Print summary statistics
    print("\nVerdict Distribution Summary:")
    print(verdict_pcts.round(1))

def plot_completeness_boxplot(df_scores, output_path):
    """
    Plot 2: Violin plot of conversation-level completeness scores by platform
    """
    # Prepare data in platform order
    data_by_platform = [
        df_scores[df_scores['platform'] == p]['completeness_score'].values 
        for p in PLATFORM_ORDER
    ]
    labels = [PLATFORM_DISPLAY[p] for p in PLATFORM_ORDER]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create violin plot
    parts = ax.violinplot(
        data_by_platform,
        positions=range(len(labels)),
        showmeans=False,
        showmedians=True,
        widths=0.7
    )
    
    # Style violin plot with platform colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PLATFORM_COLORS[PLATFORM_ORDER[i]])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style median lines
    parts['cmedians'].set_edgecolor('black')
    parts['cmedians'].set_linewidth(2)
    
    # Style other elements
    for partname in ['cbars', 'cmins', 'cmaxes']:
        parts[partname].set_edgecolor('black')
        parts[partname].set_linewidth(1)
    
    # Formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # ax.set_xlabel('Platform', fontsize=14)
    ax.set_ylabel('Completeness score', fontsize=20)
    # ax.set_title('Conversation completeness by platform', fontsize=12, pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved Plot 2 to: {output_path}")
    print(f"Saved Plot 2 (PDF) to: {str(output_path).replace('.png', '.pdf')}")
    
    # Print summary statistics
    print("\nCompleteness Score Summary:")
    summary = df_scores.groupby('platform')['completeness_score'].describe()
    print(summary.round(3))


def plot_intentions_boxplot(df_intentions, output_path):
    """
    Plot 3: Boxplot of number of intentions per conversation by platform (log scale)
    """
    import numpy as np
    
    # Prepare data in platform order
    data_by_platform = [
        df_intentions[df_intentions['platform'] == p]['num_intentions'].values 
        for p in PLATFORM_ORDER
    ]
    labels = [PLATFORM_DISPLAY[p] for p in PLATFORM_ORDER]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create simple boxplot
    box_parts = ax.boxplot(
        data_by_platform,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True
    )
    
    # Apply platform colors to boxes
    for patch, platform in zip(box_parts['boxes'], PLATFORM_ORDER):
        patch.set_facecolor(PLATFORM_COLORS[platform])
        patch.set_alpha(0.6)
    
    # Style median lines (darker, less bright)
    for median in box_parts['medians']:
        # median.set_color('#8B4513')  # Saddle brown - muted color
        median.set_color('black')
        median.set_linewidth(2)
    
    # Style outliers (lighter)
    for flier in box_parts['fliers']:
        flier.set(marker='o', markerfacecolor='lightgray', markeredgecolor='gray', 
                 markersize=4, alpha=0.4)
    
    # Display median values on plot
    for i, (pos, data) in enumerate(zip(range(1, len(data_by_platform) + 1), data_by_platform)):
        median_val = np.median(data)
        ax.text(pos, median_val, f'{median_val:.1f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    # Formatting
    # ax.set_xlabel('Platform', fontsize=14)
    ax.set_ylabel('Number of intentions (log scale)', fontsize=20)
    # ax.set_title('Intention count by platform', fontsize=12, pad=15)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, which='both')
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved Plot 3 to: {output_path}")
    print(f"Saved Plot 3 (PDF) to: {str(output_path).replace('.png', '.pdf')}")
    
    # Print summary statistics
    print("\nIntention Count Summary:")
    summary = df_intentions.groupby('platform')['num_intentions'].describe()
    print(summary.round(2))


def main():
    """Main execution function"""
    print("=" * 80)
    print("Conversation Completeness Analysis")
    print("=" * 80)
    
    # Import numpy here (needed for jitter)
    import numpy as np
    globals()['np'] = np
    
    # Load data
    data = load_data(DATA_FILE)
    
    # Extract data for each plot
    df_verdicts = extract_intention_verdicts(data)
    df_scores = extract_completeness_scores(data)
    df_intentions = extract_intention_counts(data)
    
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # Generate plots
    plot_verdict_distribution(
        df_verdicts,
        OUTPUT_DIR / "plot1_verdict_distribution.png"
    )
    
    plot_completeness_boxplot(
        df_scores,
        OUTPUT_DIR / "plot2_completeness_scores.png"
    )
    
    plot_intentions_boxplot(
        df_intentions,
        OUTPUT_DIR / "plot3_intention_counts.png"
    )
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
