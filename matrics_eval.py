import eval_func
import os
import json
from tqdm import tqdm

# load dimentional_eval_results.json and use matrics_eval
# metrics = eval_func.calculate_hybrid_metrics(image_path, llm_json_output, prompt_text, clip_model)

# load dimentional_eval_results.json
results_file = "dimentional_eval_results.json"
with open(results_file, "r") as f:
    results = json.load(f)

# load prompts.txt
prompt_file = "/usr1/home/s124mdg44_01/diffusion/eva_diff/dataset/prompt/prompts.txt"
prompts = eval_func.get_prompts(prompt_file)

# load clip model
from sentence_transformers import SentenceTransformer
clip_model = SentenceTransformer('clip-ViT-B-32')


all_metrics = []
image_dir = "/usr1/home/s124mdg44_01/diffusion/eva_diff/sd_2-1/img_output/untuned"

# evaluate each response
for key, value in tqdm(results.items()):
    index = int(key.split(".")[0])
    original_prompt = prompts[index]
    llava_output = value
    image_path = os.path.join(image_dir, key)

    metrics = eval_func.calculate_hybrid_metrics(image_path, llava_output, original_prompt, clip_model)
    all_metrics.append(metrics)

    print(json.dumps(metrics, indent=2))
    break

# map(lambda x: print(json.dumps(x, indent=2)), all_metrics[:5])

# # save metrics to json
# with open("metrics_eval_results.json", "w") as f:
#     json.dump(all_metrics, f, indent=2)
