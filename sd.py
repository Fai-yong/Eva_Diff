

# print(torch.__version__)  # 查看torch当前版本号
# print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
# print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用

# ----------------------------批量生成图片------------------------------

# load prompts from /usr1/home/s124mdg44_01/diffusion/eva_diff/dataset/prompt/prompts.txt

# with open("/usr1/home/s124mdg44_01/diffusion/eva_diff/dataset/prompt/prompts.txt", "r") as f:
#     prompts = f.readlines()
#     prompts = [prompt.strip() for prompt in prompts]


# from diffusers import DiffusionPipeline
# import torch
# from tqdm import tqdm

# model_type = "stabilityai/stable-diffusion-xl-base-1.0"

# # pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
# pipeline = DiffusionPipeline.from_pretrained(model_type, revision="fp16", torch_dtype=torch.float16)
# pipeline.to("cuda:3")

# save_path = '/usr1/home/s124mdg44_01/diffusion/eva_diff/sd_xl/img_output/untuned'

# # save img
# for i, prompt in tqdm(enumerate(prompts)):
#     pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(f"{save_path}/{i}.png")

# pipeline("A man on the sofa", num_inference_steps=50, guidance_scale=7.5).images[0].save("dog_bucket_0.png")

# ================================================================================

# ------------------------------- 试用 tunned model--------------------------------
# from diffusers import DiffusionPipeline, UNet2DConditionModel
# from transformers import CLIPTextModel
# import torch

# unet = UNet2DConditionModel.from_pretrained("/usr1/home/s124mdg44_01/diffusion/tunning/model_output/unet") # path to unet

# # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("/usr1/home/s124mdg44_01/diffusion/tunning/model_output/text_encoder") # path to text_encoder

# pipeline = DiffusionPipeline.from_pretrained(
#     "stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
# ).to("cuda")

# image = pipeline("A man on the sofa", num_inference_steps=50, guidance_scale=7.5).images[0]
# image.save("dog-bucket.png")

from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
from eval_func import get_prompts

def generate_untuned(prompts, 
                    model_path="stabilityai/stable-diffusion-xl-base-1.0", 
                    save_path='sd_xl/img_output/untuned',
                    device="cuda:3",
                    test = False):
        # Load the model
    pipeline = DiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to(device)
    pipeline.enable_xformers_memory_efficient_attention()
    
    if test:
        p_index = 3
        images = pipeline(prompts[p_index], num_inference_steps=50, guidance_scale=7.5).images
        print(f"Total images generated: {len(images)}")
        images[0].save(f'test_{p_index}.png')
        print(f'image saved test_{p_index}.png')


    # Generate and save images
    else:
        for i, prompt in tqdm(enumerate(prompts)):
            # Generate image from the prompt
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            # Save the image
            image.save(f"{save_path}/{i}.png")
import os
from diffusers import DiffusionPipeline
def generate_lora_tuned(prompts,
                        model_path,
                        save_path,
                        lora_path,
                        device="cuda:3",
                        test = True):
    # Load the model
    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(model_path,
                                            torch_dtype=torch.float16, 
                                            safety_checker=None,
                                            variant="fp16", 
                                            use_safetensors=True
                                            ).to(device)
    pipe.load_lora_weights(lora_path)
    pipe.enable_xformers_memory_efficient_attention()
    # Use xformers 
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    #     print("xformers enabled")
    # except Exception:
    #     pass 
    if test:
        p_index = 5
        # batch_size = 16
        # print(prompts[p_index: p_index + batch_size])
        # Generate image from the prompt
        images = pipe(prompts[p_index], num_inference_steps=50, guidance_scale=7.5).images
        print(f"Total images generated: {len(images)}")
        images[0].save(f'test_{p_index}.png')
        print(f'image saved test_{p_index}.png')

    else:
        for i, prompt in tqdm(enumerate(prompts)):
            # Generate image from the prompt
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            # Save the image
            image.save(f"{save_path}/{i}.png")

        # enable_xformers_memory_efficient_attention


def generate_tuned(prompts, 
                   model_path,
                   save_path,
                   model_type="stable-diffusion-v1-5/stable-diffusion-v1-5",                    
                   device="cuda:3",
                   test = False):
    unet = UNet2DConditionModel.from_pretrained(f'{model_path}/unet')
    text_encoder = CLIPTextModel.from_pretrained(f'{model_path}/text_encoder')

    pipeline = DiffusionPipeline.from_pretrained(
        model_type, unet=unet, text_encoder=text_encoder, dtype=torch.float32
    ).to(device)

    if test:
        p_index = 3
        image = pipeline(prompts[p_index], num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(f'test_{p_index}.png')
    else:
        for i, prompt in tqdm(enumerate(prompts)):
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image.save(f'{save_path}/{i}.png')
            print(f"Image {i} saved successfully")



import argparse

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Stable Diffusion generate")
    
    # 添加命令行参数
    parser.add_argument('--prompts_path', type=str, default="dataset/prompt/prompts.txt", help='Prompt txt  path')
    parser.add_argument('--save_path', type=str, required=True, help='image save path')
    parser.add_argument('--model_path', type=str, default="raw_model/sd_1.5", help='base diffusion model path')
    parser.add_argument('--lora_mode', action='store_true', help='using lora  or not')
    parser.add_argument('--device', type=str, default="cuda:1", help='device to run or accelerate, cuda:0 or cpu')
    parser.add_argument('--lora_path', type=str, default="sd1-5/lora/impasto", help='LoRA adapter path')
    parser.add_argument('--test', action='store_true', help='using test or not')
    
    # 解析参数
    args = parser.parse_args()
    
    # 获取 prompts
    prompts = get_prompts(args.prompts_path)

    lora_mode = args.lora_mode
    if lora_mode:
    # 调用生成函数
        generate_lora_tuned(
            prompts=prompts,
            lora_path=args.lora_path,
            model_path=args.model_path,
            save_path=args.save_path,
            device=args.device,
            test=args.test
        )
    else:
        generate_untuned(
            prompts=prompts,
            model_path=args.model_path,
            save_path=args.save_path,
            device=args.device,
            test=args.test
        )


if __name__ == "__main__":
    main()