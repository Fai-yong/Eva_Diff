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
OUTPUT_MD = 'scores_big_table.md'

COMBINED_HEADERS = []
for metric in METRICS:
    for stat in STATS:
        COMBINED_HEADERS.append(f"{metric}_{stat}")


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


def main():
    if not os.path.exists(SUMMARY_FILE):
        print(f"{SUMMARY_FILE} not found")
        return
    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # Collect rows
    rows = []  # list of (sd_base, lora, mllm, {metric:{stat:value}})
    for metric in METRICS:
        metric_dict = summary.get(metric, {})
        for name, stats in metric_dict.items():
            sd_base, lora, mllm = parse_name(name)
            # find existing row
            key = (sd_base, lora, mllm)
            row = next((r for r in rows if r[:3] == list(key)), None)
            if row is None:
                data = {m: {s: '' for s in STATS} for m in METRICS}
                row = [sd_base, lora, mllm, data]
                rows.append(row)
            else:
                data = row[3]
            for stat in STATS:
                val = stats.get(stat, '')
                data[metric][stat] = format_val(val)

    # Sort rows
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    # Build markdown
    md_lines = []
    header = ["SD-Base", "LoRA", "MLLM"] + COMBINED_HEADERS
    md_lines.append('| ' + ' | '.join(header) + ' |')
    md_lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    # Data rows
    for sd_base, lora, mllm, data in rows:
        row_cells = [sd_base, lora, mllm]
        for metric in METRICS:
            for stat in STATS:
                row_cells.append(data[metric][stat])
        md_lines.append('| ' + ' | '.join(row_cells) + ' |')

    # Write to file
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Markdown table written to {OUTPUT_MD}")

if __name__ == '__main__':
    main() 