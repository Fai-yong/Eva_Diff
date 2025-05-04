import argparse
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import os
import json
from tqdm import tqdm
import eval_func
import mllm_func

def parse_args():
    parser = argparse.ArgumentParser(description="MLLM 图像评估脚本")
    
    # 必需参数
    parser.add_argument('--prompt_file', type=str, help='包含评测prompt的文本文件路径')
    parser.add_argument('--images_dir', type=str, required=True,
                      help='待评估图像所在的目录路径')
    
    # 可选参数
    parser.add_argument('--mllm_model', type=str, default='llava-ov',
                      choices=['llava-ov','llava-1_6', 'qwen2_5-vl', 'qwen2-vl'],
                      help='选择使用的MLLM模型类型（默认：llava-ov）')
    parser.add_argument('--model_path', type=str, 
                      default='mllm_models/llava-ov',
                      help='MLLM模型权重路径（默认根据模型类型自动生成）')
    parser.add_argument('--template_file', type=str, default="json/llava_template.json",
                      help='对话模板的JSON文件路径')
    parser.add_argument('--device_index', type=int, default=0,
                      help='使用的CUDA设备编号（例如0,1,2...）')
    parser.add_argument('--result_dir', type=str, default="results",
                      help='评估结果保存目录（默认：results）')
    parser.add_argument('--max_new_tokens', type=int, default=2000,
                      help='模型生成的最大token数（默认：2000）')
    parser.add_argument('--fp16', action='store_true',
                      help='是否使用FP16精度（默认关闭）')
    
    return parser.parse_args()

def load_model(args):
    """根据参数加载模型和处理器"""
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    if args.mllm_model == 'llava-ov' or args.mllm_model == 'llava-1_6':
        model = mllm_func.Llava(model_path=f'mllm_models/{args.mllm_model}').to(args.device_index)
        processor = AutoProcessor.from_pretrained(f'mllm_models/{args.mllm_model}')
    elif args.mllm_model == 'qwen2_5-vl' :
        max_memory = {args.device_index: "23GB"}
        
        model = model = mllm_func.Qwen2_5vl(max_memory, model_path=f'mllm_models/{args.mllm_model}')
        processor = AutoProcessor.from_pretrained(f'mllm_models/{args.mllm_model}')
    # 其他模型的加载逻辑...
    elif args.mllm_model == 'qwen2-vl':
        max_memory = {args.device_index: "23GB"}
        
        model = mllm_func.Qwen_2vl(max_memory, model_path=f'mllm_models/{args.mllm_model}')
        processor = AutoProcessor.from_pretrained(f'mllm_models/{args.mllm_model}')
    else:
        raise ValueError(f"Unsupported model type: {args.mllm_model}")

    
    return model, processor

def main():
    args = parse_args()
    
    os.makedirs(args.result_dir, exist_ok=True)
    result_file = os.path.join(args.result_dir, f"{args.mllm_model}_eval_results.json")

    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, processor = load_model(args)
    model.eval()
    
    template_list = eval_func.get_template_list(args.template_file)
    # prompts = eval_func.get_prompts(args.prompt_file)
    
    # 处理图像
    all_results = {}
    image_files = sorted([f for f in os.listdir(args.images_dir) if f.endswith('.png')])
    
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="评估进度"):
            img_path = os.path.join(args.images_dir, img_file)
            image = Image.open(img_path)
            
            # 获取对应模板
            idx = int(os.path.splitext(img_file)[0])
            conversation = eval_func.build_mllm_conversation(template_list[idx])
            
            # 生成输入
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(args.device_index, torch.float16 if args.fp16 else torch.float32)
            
            # 模型推理
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # 保存结果
            all_results[img_file] = eval_func.get_json_resp(response)


    # 保存结果
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"result save to ：{result_file}")

if __name__ == "__main__":
    main()