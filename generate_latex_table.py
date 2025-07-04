import json
import os

STATS = ['mean', 'std', 'count']
METRICS = [
    'semantic_coverage',
    'relation_validity', 
    'style_score',
    'object_num',
    'total_attrs'
]

SUMMARY_FILE = 'scores_summary_statistics.json'
OUTPUT_TEX = 'scores_big_table.tex'

def parse_name(name: str):
    """Return sd_base, lora, mllm given composite key."""
    parts = name.split('_')
    sd_base = parts[0]  # sd_xl or sd1-5
    # last token is mllm
    mllm = parts[-1]
    if len(parts) == 3:
        lora = 'results'
    else:
        # join parts[2:-1]
        lora = '_'.join(parts[2:-1])
    return sd_base, lora, mllm

def format_val(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

def generate_latex_table():
    print(f"Looking for {SUMMARY_FILE}")
    if not os.path.exists(SUMMARY_FILE):
        print(f"{SUMMARY_FILE} not found")
        return
    
    print(f"Loading data from {SUMMARY_FILE}")
    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} metrics")
    
    # Collect all combinations
    all_keys = set()
    for metric_data in data.values():
        all_keys.update(metric_data.keys())
    
    # Parse and sort
    rows = []
    for key in all_keys:
        sd_base, lora, mllm = parse_name(key)
        row_data = {}
        for metric in METRICS:
            if key in data[metric]:
                row_data[metric] = {
                    stat: format_val(data[metric][key][stat]) 
                    for stat in STATS
                }
            else:
                row_data[metric] = {stat: '-' for stat in STATS}
        rows.append((sd_base, lora, mllm, row_data))
    
    # Sort by sd_base, lora, mllm
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    
    print(f"Generated {len(rows)} rows")
    
    # Generate LaTeX
    latex_lines = []
    
    # Document setup
    latex_lines.append("\\documentclass{article}")
    latex_lines.append("\\usepackage[landscape]{geometry}")
    latex_lines.append("\\usepackage{booktabs}")
    latex_lines.append("\\usepackage{multirow}")
    latex_lines.append("\\usepackage{array}")
    latex_lines.append("\\begin{document}")
    latex_lines.append("")
    
    # Table setup - calculate number of columns
    num_stat_cols = len(METRICS) * len(STATS)  # 5 * 3 = 15
    total_cols = 3 + num_stat_cols  # SD-Base, LoRA, MLLM + 15 stat columns
    
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\tiny")  # Make font smaller to fit
    latex_lines.append(f"\\begin{{tabular}}{{|l|l|l|{'c|' * num_stat_cols}}}")
    latex_lines.append("\\hline")
    
    # First header row (Metrics)
    header1 = ["\\multirow{2}{*}{SD-Base}", "\\multirow{2}{*}{LoRA}", "\\multirow{2}{*}{MLLM}"]
    for metric in METRICS:
        metric_name = metric.replace('_', '\\_')
        header1.append(f"\\multicolumn{{{len(STATS)}}}{{c|}}{{{metric_name}}}")
    latex_lines.append(" & ".join(header1) + " \\\\")
    latex_lines.append("\\cline{4-" + str(total_cols) + "}")
    
    # Second header row (Statistics)
    header2 = ["", "", ""]  # Empty cells for the multirow entries
    for metric in METRICS:
        for stat in STATS:
            header2.append(stat)
    latex_lines.append(" & ".join(header2) + " \\\\")
    latex_lines.append("\\hline")
    
    # Data rows
    for sd_base, lora, mllm, row_data in rows:
        row_cells = [sd_base, lora, mllm]
        for metric in METRICS:
            for stat in STATS:
                row_cells.append(row_data[metric][stat])
        latex_lines.append(" & ".join(row_cells) + " \\\\")
        latex_lines.append("\\hline")
    
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Comprehensive Scores Statistics Table}")
    latex_lines.append("\\label{tab:scores_statistics}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")
    latex_lines.append("\\end{document}")
    
    # Write to file
    print(f"Writing to {OUTPUT_TEX}")
    with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table generated: {OUTPUT_TEX}")

if __name__ == "__main__":
    generate_latex_table() 