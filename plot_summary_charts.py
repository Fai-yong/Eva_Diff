import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 全局样式：使用高级会议排版级别的字体与尺寸
sns.set_theme(style="whitegrid", palette="colorblind")
sns.set_context("talk", font_scale=1.2, rc={
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# --------------------------------------------------
# Config
# --------------------------------------------------
results_bases = [
    "sd_xl/results",
    "sd_xl/results/lcm", 
    "sd_xl/results/slider_extrem",
    "sd_xl/results/slider_ultra",
    "sd1-5/results",
    "sd1-5/results/hands",
    "sd1-5/results/impasto",
    "sd1-5/results/retro"
]

subconfig_labels = [
    "sd_xl/base",
    "sd_xl/lcm",
    "sd_xl/slider_extrem",
    "sd_xl/slider_ultra",
    "sd1-5/base",
    "sd1-5/hands",
    "sd1-5/impasto",
    "sd1-5/retro"
]

results_tags = ["llava-1_6", "llava-ov", "qwen2_5-vl"]

metrics = ["semantic_coverage", "relation_validity", "style_score", "object_num", "total_attrs"]
metric_labels = {
    "semantic_coverage": "Semantic Coverage",
    "relation_validity": "Relation Validity",
    "style_score": "Style Score",
    "object_num": "Object Number",
    "total_attrs": "Total Attributes"
}

colors = {
    "llava-1_6": "#2E86AB",    # 深蓝
    "llava-ov": "#A23B72",     # 深玫红  
    "qwen2_5-vl": "#F18F01"    # 深橙
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------

# 预生成映射：下划线形式 -> 原始 results_bases 路径
underscore_to_base = {base.replace('/', '_'): base for base in results_bases}

# 解析 summary key，提取 base_path 与 tag
def parse_summary_key(key: str):
    """Return (base_path, tag) from a summary json key."""
    tag_found = None
    for t in results_tags:
        if key.endswith(t):
            tag_found = t
            base_part = key[: -(len(t) + 1)]  # strip '_' + tag
            break
    if tag_found is None:
        return None, None

    # map underscored base_part to canonical base_path
    base_path = underscore_to_base.get(base_part)
    if base_path is None:
        # fallback—convert patterns generically (may not be in predefined list)
        base_path = base_part.replace('_results', '/results').replace('_', '/')
    return base_path, tag_found

def load_summary(path="scores_summary_statistics.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_scores(file_base, tag):
    path = f"{file_base}/{tag}_scores.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

summary = load_summary()

# --------------------------------------------------
# Build tidy DataFrame from summary json
# --------------------------------------------------
records = []
for metric in metrics:
    metric_dict = summary.get(metric, {})
    for key, stats in metric_dict.items():
        base_path, tag = parse_summary_key(key)
        if base_path is None:
            continue  # skip unknown pattern

        subconfig = subconfig_labels[results_bases.index(base_path)] if base_path in results_bases else base_path

        records.append({
            'file_base': base_path,
            'subconfig': subconfig,
            'tag': tag,
            'metric': metric,
            'mean': stats['mean'],
            'std': stats['std'],
            'count': stats['count']
        })

df = pd.DataFrame(records)

# ensure subconfig order categorical
cat_order = subconfig_labels

df['subconfig'] = pd.Categorical(df['subconfig'], categories=cat_order, ordered=True)

# --------------------------------------------------
# 1. Bar plots (mean ± std) for semantic_coverage & relation_validity
# --------------------------------------------------

bar_metrics = ["semantic_coverage", "relation_validity", "style_score"]
for metric in bar_metrics:
    plt.figure(figsize=(16, 8))
    metric_df = df[df['metric'] == metric]
    ax = sns.barplot(data=metric_df, x='subconfig', y='mean', hue='tag', palette=colors,
                     width=0.5, alpha=0.8, linewidth=0.8, edgecolor='white')
    ax.grid(axis='y', linewidth=0.4, alpha=0.3)
    sns.despine(ax=ax)
    ax.set_ylabel("Mean Score")
    plt.title(f"{metric_labels[metric]} across Subconfigs and MLLMs")
    ax.set_xlabel("SD Model Version")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MLLM', loc='upper right', bbox_to_anchor=(1.15, 1.0), frameon=False)
    Path('summary_plots').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"summary_plots/{metric}_bar.png", dpi=300)
    plt.close()

# --------------------------------------------------
# 2. Heatmaps for semantic_coverage and relation_validity (mean & std)
# --------------------------------------------------
heat_metrics = ["semantic_coverage", "relation_validity", "style_score"]
for metric in heat_metrics:
    for stat in ['mean', 'std']:
        pivot = df[df['metric'] == metric].pivot(index='subconfig', columns='tag', values=stat).loc[cat_order]
        plt.figure(figsize=(8, 6))
        cmap_choice = 'rocket_r' if stat == 'mean' else 'mako_r'
        sns.heatmap(pivot, annot=True, cmap=cmap_choice, fmt='.2f',
                    annot_kws={'size':10}, linewidths=0.4, linecolor='white', cbar_kws={'shrink':0.8})
        plt.title(f"Heatmap of {metric_labels[metric]} {stat.upper()}")
        plt.xlabel('MLLM')
        plt.ylabel('SD Model Version')
        plt.tight_layout()
        plt.savefig(f"summary_plots/{metric}_{stat}_heatmap.png", dpi=300)
        plt.close()

# --------------------------------------------------
# 3. Line plot for counts
# --------------------------------------------------
count_df = df[df['metric'] == 'semantic_coverage'][['subconfig', 'tag', 'count']]
plt.figure(figsize=(12, 6))
for tag in results_tags:
    sub = count_df[count_df['tag'] == tag].sort_values('subconfig')
    plt.plot(sub['subconfig'], sub['count'], marker='o', markersize=6, linewidth=2.2, label=tag, color=colors[tag])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Image Count')
plt.title('Sample Count per Dataset Configuration for each MLLM')
plt.xlabel('Dataset Configuration')
plt.legend(title='MLLM')
plt.tight_layout()
plt.savefig('summary_plots/count_line.png', dpi=300)
plt.close()

# --------------------------------------------------
# 4. Violin plot for semantic_coverage, relation_validity, style_score distributions (aggregate)
#     Need raw data
# --------------------------------------------------
all_style_records = []
all_coverage_records = []
all_relation_records = []
for base in results_bases:
    for tag in results_tags:
        scores = load_scores(base, tag)
        if scores is None:
            continue
        for img, metrics_dict in scores.items():
            if 'style_score' in metrics_dict:
                all_style_records.append({'subconfig': subconfig_labels[results_bases.index(base)],
                                           'tag': tag,
                                           'style_score': metrics_dict['style_score']})
            if 'semantic_coverage' in metrics_dict:
                all_coverage_records.append({'subconfig': subconfig_labels[results_bases.index(base)],
                                           'tag': tag,
                                           'semantic_coverage': metrics_dict['semantic_coverage']})
            if 'relation_validity' in metrics_dict:
                all_relation_records.append({'subconfig': subconfig_labels[results_bases.index(base)],
                                           'tag': tag,
                                           'relation_validity': metrics_dict['relation_validity']})

coverage_df = pd.DataFrame(all_coverage_records)
relation_df = pd.DataFrame(all_relation_records)
style_df = pd.DataFrame(all_style_records)

# style_score
plt.figure(figsize=(8, 6))
sns.violinplot(data=style_df, x='tag', y='style_score', palette=colors, inner='quartile', alpha=0.7,
               linewidth=0.8)
sns.despine()
plt.title('Style Score Distribution')
plt.xlabel('MLLM')
plt.ylabel('Style Score')
plt.tight_layout()
plt.savefig('summary_plots/style_score_violin.png', dpi=300)
plt.close()

# semantic_coverage
plt.figure(figsize=(8, 6))
sns.violinplot(data=coverage_df, x='tag', y='semantic_coverage', palette=colors, inner='quartile', alpha=0.7,
               linewidth=0.8)
sns.despine()
plt.title('Semantic Coverage Distribution')
plt.xlabel('MLLM')
plt.ylabel('Semantic Coverage')
plt.tight_layout()
plt.savefig('summary_plots/semantic_coverage_violin.png', dpi=300)
plt.close()

# relation_validity
plt.figure(figsize=(8, 6))
sns.violinplot(data=relation_df, x='tag', y='relation_validity', palette=colors, inner='quartile', alpha=0.7,
               linewidth=0.8)
sns.despine()
plt.title('Relation Validity Distribution')
plt.xlabel('MLLM')
plt.ylabel('Relation Validity')
plt.tight_layout()
plt.savefig('summary_plots/relation_validity_violin.png', dpi=300)
plt.close()

# --------------------------------------------------
# 5. Radar charts for each (subconfig, tag)
# --------------------------------------------------
radar_metrics = ["semantic_coverage", "relation_validity", "style_score", "object_num", "total_attrs"]
# normalisation: for each metric, compute global min-max across means
norm_values = {}
for m in radar_metrics:
    mm = df[df['metric'] == m]['mean']
    norm_values[m] = (mm.min(), mm.max())

def radar_factory(num_vars, frame='circle'):
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    import math

    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1
        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
    register_projection(RadarAxes)
    return theta

theta = radar_factory(len(radar_metrics))

def normalize(metric, value):
    min_val, max_val = norm_values[metric]
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

for base in results_bases:
    for tag in results_tags:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='radar')
        means = []
        for m in radar_metrics:
            row = df[(df['subconfig'] == subconfig_labels[results_bases.index(base)]) & (df['tag'] == tag) & (df['metric'] == m)]
            if row.empty:
                means.append(0)
            else:
                means.append(normalize(m, row.iloc[0]['mean']))
        # close the loop by repeating the first value and theta
        theta_closed = np.concatenate([theta, theta[:1]])
        means_closed = means + means[:1]
        ax.plot(theta_closed, means_closed, color=colors[tag], linewidth=2.2)
        ax.fill(theta_closed, means_closed, color=colors[tag], alpha=0.18)
        ax.set_varlabels([metric_labels[m] for m in radar_metrics])
        ax.set_title(f"{subconfig_labels[results_bases.index(base)]} - {tag}", position=(0.5, 1.1), fontsize=15)
        ax.spines['polar'].set_visible(False)
        ax.grid(color='grey', linewidth=0.5, linestyle='dashed', alpha=0.6)
        plt.tight_layout()
        Path('summary_plots/radar').mkdir(parents=True, exist_ok=True)
        safe_sub = subconfig_labels[results_bases.index(base)].replace('/', '_')
        plt.savefig(f"summary_plots/radar/{safe_sub}_{tag}_radar.png", dpi=300)
        plt.close()

# --------------------------------------------------
# 6. Grouped radar charts: one for sd_xl, one for sd1-5 (lines = MLLM)
# --------------------------------------------------

group_bases = {
    'sd_xl': [lbl for lbl in subconfig_labels if lbl.startswith('sd_xl')],
    'sd1-5': [lbl for lbl in subconfig_labels if lbl.startswith('sd1-5')]
}

for base_key, subs in group_bases.items():
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='radar')

    for tag in results_tags:
        values = []
        for m in radar_metrics:
            # 取该 base 下所有 subconfig 的该 MLLM 的平均，再对均值再平均
            rows = df[(df['subconfig'].isin(subs)) & (df['tag'] == tag) & (df['metric'] == m)]
            if rows.empty:
                values.append(0)
            else:
                values.append(normalize(m, rows['mean'].mean()))
        theta_closed = np.concatenate([theta, theta[:1]])
        vals_closed = values + values[:1]
        ax.plot(theta_closed, vals_closed, color=colors[tag], linewidth=2.2, label=tag)
        ax.fill(theta_closed, vals_closed, color=colors[tag], alpha=0.12)

    ax.set_varlabels([metric_labels[m] for m in radar_metrics])
    ax.set_title(f"{base_key.upper()} - Aggregated Radar", position=(0.5, 1.12), fontsize=18, weight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=False)
    # 美化网格
    ax.spines['polar'].set_visible(False)
    ax.grid(color='grey', linewidth=0.5, linestyle='dashed', alpha=0.6)
    plt.tight_layout(pad=2.0)
    Path('summary_plots/radar_grouped').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'summary_plots/radar_grouped/{base_key}_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Grouped radar charts saved to summary_plots/radar_grouped/")

print("All summary plots generated in 'summary_plots/' directory.") 