import eval_func
import os
import json
from tqdm import tqdm

# load dimentional_eval_results.json and use matrics_eval
# metrics = eval_func.calculate_hybrid_metrics(image_path, llm_json_output, prompt_text, clip_model)

# load dimentional_eval_results.json
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
image_dirs = [
    "sd_xl/img_output/untuned",
    "sd_xl/img_output/tuned/lcm-lora",
    "sd_xl/img_output/tuned/slider_extrem",
    "sd_xl/img_output/tuned/slider_ultra",
    "sd1-5/img_output/untuned",
    "sd1-5/img_output/tuned/hands",
    "sd1-5/img_output/tuned/impasto",
    "sd1-5/img_output/tuned/retro"
]
results_tags = [
    "llava-1_6",
    "llava-ov",
    "qwen2_5-vl"
]

# e.g.: 'llava-1_6_eval_results.json' for llava-1_6
def get_results(file_base, results_tag):
    results_file = f"{file_base}/{results_tag}_eval_results.json"
    with open(results_file, "r") as f:
        results = json.load(f)
    return results

def save_scores(file_base, all_metrics, results_tag):
    scores_file = f"{file_base}/{results_tag}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(all_metrics, f, indent=2)


# load prompts.txt
prompt_file = "/usr1/home/s124mdg44_01/diffusion/eva_diff/dataset/prompt/prompts.txt"

def matrics_eval(file_base, results_tag, image_dir, prompt_file):
    prompts = eval_func.get_prompts(prompt_file)

    # load clip model
    # from sentence_transformers import SentenceTransformer
    # clip_model = SentenceTransformer('clip-ViT-B-32')

    # load results
    results = get_results(file_base=file_base, results_tag=results_tag)

    all_metrics = {}

    # evaluate each response
    for key, value in tqdm(results.items()):
        index = int(key.split(".")[0])
        original_prompt = prompts[index]
        image_path = os.path.join(image_dir, key)

        # if eval_data['objects'] is None, print eval_data and exit
        if value is None:
            continue
        
        # try and except
        try:
            metrics = eval_func.calculate_hybrid_metrics(image_path, value, original_prompt)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"Error evaluating {key}: {tag}, {file_base}")
            print(e)
            exit()

        # metrics = eval_func.calculate_hybrid_metrics(image_path, value, original_prompt)
        # all_metrics[key] = metrics

        # print(json.dumps(metrics, indent=2))

        # for test only
        # break

    return all_metrics

# test all json in all results_tags under all results_bases
for i in range(len(results_bases)):
    results_base = results_bases[i]
    image_dir = image_dirs[i]
    for tag in results_tags:
        all_metrics = matrics_eval(file_base=results_base, results_tag=tag, image_dir=image_dir, prompt_file=prompt_file)
        save_scores(file_base=results_base, all_metrics=all_metrics, results_tag=tag)
        print(f'all_metrics has {len(all_metrics)} items')
        print(f"Saved {tag} scores to {results_base}/{tag}_scores.json")



# map(lambda x: print(json.dumps(x, indent=2)), all_metrics[:5])

# # save metrics to json
# with open("metrics_eval_results.json", "w") as f:
#     json.dump(all_metrics, f, indent=2)
