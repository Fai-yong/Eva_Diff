# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
# import torch
# from PIL import Image
# import requests
# import os
# import json
# import re
# from tqdm import tqdm
# import eval_func
# import mllm_func

# device_index = 2
# diffusion_base = 'sd1-5'
# mllm_model = 'llava-ov'
# template_file = "json/llava_template.json"

# model_path = f'mllm_models/{mllm_model}'
# save_path = f'{diffusion_base}/results/hands'
# images_dir = f"{diffusion_base}/img_output/tuned/hands"
# result_file = f'{save_path}/llava_ov_eval_results.json' # change with different mllm model

# # ------------------------------- Using Llava Onevision For Conditional Generation --------------------------------
# model = mllm_func.Llava(model_path=model_path).to(device_index)
# processor = AutoProcessor.from_pretrained(model_path)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# # Set up device map for model parallelism
# max_memory = {0: "12GB", 1: "22GB"}

# # ------------------------------- Using Qwen For Conditional Generation --------------------------------

# # model_name = "Qwen/Qwen2-VL-7B-Instruct"
# # model_name = "Qwen/Qwen2-VL-2B-Instruct"
# # model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
# # model = mllm_func.Qwen_2vl(max_memory, model_name)


# # from transformers import AutoModel
# # model_name = "MAGAer13/mplug-owl-llama-7b"
# # model = AutoModel.from_pretrained(
# #     model_name,
# #     torch_dtype=torch.float16,
# #     low_cpu_mem_usage=True,
# #     max_memory=max_memory,
# #     device_map="auto"
# # )

# from transformers import AutoModelForImageTextToText, AutoProcessor
# # model_name = "google/gemma-3-12b-it"
# # model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

# # model_name = "openbmb/MiniCPM-Llama3-V-2_5"
# # model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# processor = AutoProcessor.from_pretrained(model_path)


# templat_list = eval_func.get_template_list(template_file=template_file)

# prompt_file = "dataset/prompt/prompts.txt"
# prompts = eval_func.get_prompts(prompt_file)

# all_results = {}

# # open images file in directory
# total_images = len([name for name in os.listdir(images_dir) if name.endswith(".png")])

# # read and print image file name in order from 0 to total_images
# for i in tqdm(range(total_images)):
#     image_path = os.path.join(images_dir, f"{i}.png")
#     image = Image.open(image_path)
#     # print(f'image path: {image_path}')

#     template = templat_list[i]
#     conversation = eval_func.build_mllm_conversation(template)
#     # print("---------------------------------")
#     # print(json.dumps(conversation, indent=2))

#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device_index, torch.float16)
    
#     output = model.generate(**inputs, max_new_tokens=2000)

#     response = processor.decode(output[0], skip_special_tokens=True)
#     json_response = eval_func.get_json_resp(response)
#     # print("--------------------------------------------------------------------------------")
#     # print(json_response)

#     # add in all_results
#     all_results[f"{i}.png"] = json_response

#     # print("Json of response:")
#     # print(json.dumps(json.loads(response), indent=2))

#     # llm_json_output = json.loads(response)
#     # prompt_text = prompts[i]
#     # metrics = eval_func.calculate_hybrid_metrics(image_path, llm_json_output, prompt_text, clip_model)
#     # print(json.dumps(metrics, indent=2))
# # 保存结果为JSON
# with open(result_file, "w") as f:
#     json.dump(all_results, f, indent=2)

# print(f"Successfuly saved {result_file}")

import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor
import eval_func
import mllm_func


# ---------- 工具函数 ----------
def load_result_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_result_json(json_path, result_data):
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

def get_error_images(result_data):
    return [img_name for img_name, val in result_data.items() if val is None]


# ---------- VLM生成函数 ----------
def generate_response_for_image(image_path, template, model, processor, device_index):
    image = Image.open(image_path)
    conversation = eval_func.build_mllm_conversation(template)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device_index, torch.float16)
    output = model.generate(**inputs, max_new_tokens=4000)
    response = processor.decode(output[0], skip_special_tokens=True)

    json_response = eval_func.get_json_resp(response)
    return json_response


# ---------- 重试并替换结果 ----------
def regenerate_missing_results(result_json_path, images_dir, templat_list, model, processor, device_index):
    result_data = load_result_json(result_json_path)
    bad_image_keys = get_error_images(result_data)

    error_dict = {"total": 0}

    for image_key in tqdm(bad_image_keys, desc="Regenerating images", unit="image"):
        error_dict["total"] += 1
        
        retries = 0
        success = False
        while not success:
            image_path = os.path.join(images_dir, image_key)
            i = int(image_key.split(".")[0])
            template = templat_list[i]

            json_response = generate_response_for_image(
                image_path=image_path,
                template=template,
                model=model,
                processor=processor,
                device_index=device_index
            )
            if json_response == None:
                retries += 1

            else: 
                result_data[image_key] = json_response
                retries += 1
                success = True

        error_dict[image_key] = retries

    save_result_json(result_json_path, result_data)
    
    return error_dict

import argparse
# ---------- 主函数 ----------
def main():

    parser = argparse.ArgumentParser(description="MLLM 图像评估脚本")
    parser.add_argument("--device_index", type=int, default=2, help="GPU设备索引")
    parser.add_argument("--diffusion_base", type=str, default='sd1-5', help="扩散模型基础路径")
    parser.add_argument("--mllm_model", type=str, default='llava-ov', help="多模态大语言模型名称")
    parser.add_argument('--images_dir', type=str, default='sd1-5/img_output/tuned/hands', help='图像目录')
    parser.add_argument("--save_path", type=str, default='sd1-5/results', help="结果保存路径")

    args = parser.parse_args()

    # ---- 参数设置 ----
    device_index = args.device_index
    diffusion_base = args.diffusion_base
    mllm_model = args.mllm_model
    images_dir = args.images_dir
    save_path = args.save_path
    template_file = "json/llava_template.json"

    model_path = f'mllm_models/{mllm_model}'
    result_file = f'{save_path}/{mllm_model}_eval_results.json'

    # ---- 结果文件 ----
    print(f"Loading result file: {result_file}")

    # ---- 模型加载 ----
    model = mllm_func.Llava(model_path=model_path).to(device_index)
    processor = AutoProcessor.from_pretrained(model_path)

    # ---- 模板加载 ----
    templat_list = eval_func.get_template_list(template_file=template_file)

    # ---- 执行重生成逻辑 ----
    error_dict = regenerate_missing_results(
        result_json_path=result_file,
        images_dir=images_dir,
        templat_list=templat_list,
        model=model,
        processor=processor,
        device_index=device_index
    )

    # ---- 记录错误情况 ----
    error_summary_file = os.path.join(save_path, f"{mllm_model}_error_summary.json")
    with open(error_summary_file, "w") as f:
        json.dump(error_dict, f, indent=2)

    print(f"Finished. Final error count: {error_dict['total']}")
    print(f"Saved result: {result_file}")
    print(f"Saved error summary: {error_summary_file}")


if __name__ == "__main__":
    main()

