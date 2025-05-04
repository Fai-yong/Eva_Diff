# import sd

# # prompts = sd.get_prompts("/usr1/home/s124mdg44_01/diffusion/eva_diff/dataset/prompt/prompts.txt")
# # 示例使用流程
# prompt = "brutalist architecture by le corbusier, abandoned buildings, cyberpunk colour palette"

# # 1. 解析Prompt
# parsed = eval_func.parse_prompt(prompt)
# print("解析结果:", parsed)

# # 2. 生成评估模板
# template = eval_func.generate_eval_template(parsed)
# print("评估模板:\n", template)

# # 3. 检测文化引用
# cultural_refs = eval_func.detect_cultural_references(prompt)
# print("文化引用:", cultural_refs)

# # 4. 假设LLM输出结果
# llm_output = """
# {
#   "objects": [
#     {"name": "brutalist architecture", "present": true},
#     {"name": "abandoned buildings", "present": true}
#   ],
#   "style_consistency": {"score": 4, "evidence": "匹配粗野主义风格但色彩偏未来感"},
#   "relations": []
# }
# """

# # 5. 计算指标
# metrics = eval_func.calculate_hybrid_metrics("demo_image.jpg", llm_output, prompt)
# print("评估指标:", metrics)

#---------------------------------------Extract Object, Attributes, and Relations---------------------------------------

import extract_sym_func as esf
from tqdm import tqdm
import json
# 示例使用流程
# prompt = "brutalist architecture by le corbusier, abandoned buildings, cyberpunk colour palette"

def get_prompts(file_path):
    with open(file_path, "r") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    return prompts

prompts = get_prompts("dataset/prompt/prompts.txt")
template_list = []
# extract objects, attributes, and relations form prompts with tqdm
for index, prompt in enumerate(tqdm(prompts)):
    # 1. 解析Prompt
    parsed = esf.parse_prompt(prompt)
    # print(json.dumps(parsed, indent=2)+"\n")
    # print("解析结果:", parsed)

    # 2. 生成评估模板
    template = esf.generate_eval_template(parsed)
    # print("评估模板:\n", template)
    template_item = {
        "index": index,
        "prompt": prompt,
        "llava_template": template
    }
    template_list.append(template_item)

# save as json

with open("llava_template.json", "w") as f:
    json.dump(template_list, f, indent=2)
