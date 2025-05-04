from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import json
import re
from tqdm import tqdm
import eval_func
import mllm_func

device_index = 2
diffusion_base = 'sd1-5'
mllm_model = 'llava-ov'
template_file = "json/llava_template.json"

model_path = f'mllm_models/{mllm_model}'
save_path = f'{diffusion_base}/results/hands'
images_dir = f"{diffusion_base}/img_output/tuned/hands"
result_file = f'{save_path}/llava_ov_eval_results.json' # change with different mllm model

# ------------------------------- Using Llava Onevision For Conditional Generation --------------------------------
model = mllm_func.Llava(model_path=model_path).to(device_index)
processor = AutoProcessor.from_pretrained(model_path)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Set up device map for model parallelism
max_memory = {0: "12GB", 1: "22GB"}

# ------------------------------- Using Qwen For Conditional Generation --------------------------------

# model_name = "Qwen/Qwen2-VL-7B-Instruct"
# model_name = "Qwen/Qwen2-VL-2B-Instruct"
# model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
# model = mllm_func.Qwen_2vl(max_memory, model_name)


# from transformers import AutoModel
# model_name = "MAGAer13/mplug-owl-llama-7b"
# model = AutoModel.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     max_memory=max_memory,
#     device_map="auto"
# )

from transformers import AutoModelForImageTextToText, AutoProcessor
# model_name = "google/gemma-3-12b-it"
# model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

# model_name = "openbmb/MiniCPM-Llama3-V-2_5"
# model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_path)


templat_list = eval_func.get_template_list(template_file=template_file)

prompt_file = "dataset/prompt/prompts.txt"
prompts = eval_func.get_prompts(prompt_file)

all_results = {}

# open images file in directory
total_images = len([name for name in os.listdir(images_dir) if name.endswith(".png")])

# read and print image file name in order from 0 to total_images
for i in tqdm(range(total_images)):
    image_path = os.path.join(images_dir, f"{i}.png")
    image = Image.open(image_path)
    # print(f'image path: {image_path}')

    template = templat_list[i]
    conversation = eval_func.build_mllm_conversation(template)
    # print("---------------------------------")
    # print(json.dumps(conversation, indent=2))

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device_index, torch.float16)
    
    output = model.generate(**inputs, max_new_tokens=2000)

    response = processor.decode(output[0], skip_special_tokens=True)
    json_response = eval_func.get_json_resp(response)
    # print("--------------------------------------------------------------------------------")
    # print(json_response)

    # add in all_results
    all_results[f"{i}.png"] = json_response

    # print("Json of response:")
    # print(json.dumps(json.loads(response), indent=2))

    # llm_json_output = json.loads(response)
    # prompt_text = prompts[i]
    # metrics = eval_func.calculate_hybrid_metrics(image_path, llm_json_output, prompt_text, clip_model)
    # print(json.dumps(metrics, indent=2))
# 保存结果为JSON
with open(result_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Successfuly saved {result_file}")


