from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch

def Llava(model_path):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    return model

def Qwen_2vl(max_memory, model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map= "auto",
        max_memory=max_memory
    )
    return model


def Qwen2_5vl(max_memory, model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map= "auto",
        max_memory=max_memory
    )
    return model

