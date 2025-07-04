import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# 设置matplotlib支持英文显示
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn美观风格
sns.set_theme(style="whitegrid", palette="muted")

# 参考matrics_eval.py中的路径定义
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

results_tags = [
    "llava-1_6",
    "llava-ov", 
    "qwen2_5-vl"
]

# 定义维度名称和英文标签
dimensions = {
    'semantic_coverage': 'Semantic Coverage',
    'relation_validity': 'Relation Validity', 
    'style_score': 'Style Score',
    'object_num': 'Object Number',
    'total_attrs': 'Total Attributes'
}

# 为不同MLLM定义颜色
colors = {
    'llava-1_6': '#1f77b4',      # 蓝色
    'llava-ov': '#ff7f0e',       # 橙色  
    'qwen2_5-vl': '#2ca02c'      # 绿色
}

# 透明度
alpha_kde = 0.45
alpha_hist = 0.18

def load_scores(file_base, results_tag):
    """加载scores文件"""
    scores_file = f"{file_base}/{results_tag}_scores.json"
    if os.path.exists(scores_file):
        with open(scores_file, "r", encoding='utf-8') as f:
            return json.load(f)
    return None

def create_plot_directory(file_base):
    """创建综合plot目录"""
    plot_dir = f"{file_base}/comparison_plots"
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def extract_dimension_data(scores_data, dimension):
    """提取指定维度的所有数据"""
    data = []
    for image_key, metrics in scores_data.items():
        if dimension in metrics:
            data.append(metrics[dimension])
    return data

def plot_dimension_comparison(file_base, dimension, dimension_label):
    """为指定数据集和维度画多MLLM对比图"""
    plt.figure(figsize=(12, 8))
    
    # 收集所有可用的数据
    all_data = {}
    for results_tag in results_tags:
        scores_data = load_scores(file_base, results_tag)
        if scores_data is not None:
            data = extract_dimension_data(scores_data, dimension)
            if data:
                all_data[results_tag] = data
    
    if not all_data:
        print(f"No data found for {dimension} in {file_base}")
        return
    
    # 根据数据类型选择不同的图表类型
    if dimension in ['semantic_coverage', 'relation_validity', 'style_score']:
        # 对于连续数据，使用直方图对比
        for tag, data in all_data.items():
            # 可选：背景直方图
            sns.histplot(data, bins=30, color=colors[tag], alpha=alpha_hist, stat='density', element='step', linewidth=1)
            # 平滑分布曲线
            if len(set(data)) > 1:
                sns.kdeplot(data, fill=True, color=colors[tag], alpha=alpha_kde, linewidth=2.5, label=tag)
            else:
                # 只有一个值时画竖线
                plt.axvline(data[0], color=colors[tag], linestyle='--', alpha=0.7, label=tag)
    else:
        # 对于整数数据，使用条形图对比
        all_values = set()
        for data in all_data.values():
            all_values.update(data)
        all_values = sorted(list(all_values))
        bar_width = 0.22
        num_models = len(all_data)
        for i, (tag, data) in enumerate(all_data.items()):
            unique_values, counts = np.unique(data, return_counts=True)
            full_counts = np.zeros(len(all_values))
            for val, count in zip(unique_values, counts):
                idx = all_values.index(val)
                full_counts[idx] = count
            positions = np.arange(len(all_values)) + i * bar_width
            plt.bar(positions, full_counts/len(data), bar_width, alpha=alpha_kde+0.1, color=colors[tag], label=tag, edgecolor='black', linewidth=0.7)
        plt.xticks(np.arange(len(all_values)) + bar_width * (num_models - 1) / 2, all_values)
        plt.ylabel('Frequency', fontsize=14)
    
    # 创建图例
    legend_elements = []
    for tag in all_data.keys():
        legend_elements.append(plt.Line2D([0], [0], color=colors[tag], lw=3, label=tag))
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=13, frameon=True)
    plt.xlabel(f'{dimension_label}', fontsize=15)
    plt.ylabel('Density' if dimension in ['semantic_coverage', 'relation_validity', 'style_score'] else 'Frequency', fontsize=15)
    
    # 创建标题
    dataset_name = file_base.replace("/", "_")
    plt.title(f'{dataset_name} - {dimension_label} Distribution Comparison', 
              fontsize=17, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    plot_dir = create_plot_directory(file_base)
    filename = f"{dataset_name}_{dimension}_comparison.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {filepath}")

def plot_all_comparisons():
    """为所有数据集和维度画对比图"""
    total_plots = 0
    
    for file_base in results_bases:
        print(f"\nProcessing dataset: {file_base}")
        
        # 检查该数据集下是否有任何scores文件
        has_data = False
        for results_tag in results_tags:
            if load_scores(file_base, results_tag) is not None:
                has_data = True
                break
        
        if not has_data:
            print(f"No scores files found in {file_base}")
            continue
        
        # 为每个维度画对比图
        for dimension, dimension_label in dimensions.items():
            plot_dimension_comparison(file_base, dimension, dimension_label)
            total_plots += 1
    
    print(f"\n总共生成了 {total_plots} 张对比图")

def generate_summary_statistics():
    """生成汇总统计信息"""
    all_stats = defaultdict(lambda: defaultdict(dict))
    
    for file_base in results_bases:
        for results_tag in results_tags:
            scores_data = load_scores(file_base, results_tag)
            
            if scores_data is None:
                continue
            
            dataset_name = file_base.replace("/", "_")
            
            for dimension in dimensions.keys():
                data = extract_dimension_data(scores_data, dimension)
                if data:
                    all_stats[dimension][f"{dataset_name}_{results_tag}"] = {
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'median': float(np.median(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data)),
                        'count': int(len(data))
                    }
    
    # 保存统计摘要
    summary_file = "scores_summary_statistics.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计摘要已保存到: {summary_file}")

if __name__ == "__main__":
    print("Generating beautiful comparison distribution plots...")
    plot_all_comparisons()

    # print("\nGenerating summary statistics...")
    # generate_summary_statistics()
    print("\nAll tasks completed!") 