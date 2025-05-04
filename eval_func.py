import torch
from PIL import Image
import re
import os
import json

# # 预定义的问题模板
# eval_prompts = {
#     "object": 
#         """Role: You are a meticulous image analyzer. Follow these steps:
#         1. List all objects mentioned in the original prompt: [PROMPT].
#         2. For each object, check its presence in the image.
#         3. If present, describe its attributes (color, size, position) and relationships to other objects.
#         4. Output in JSON format:
#         {
#         "objects": [
#             {
#             "name": "object_name",
#             "present": true/false,
#             "color": "string",
#             "size_relative": "larger/smaller/same as [other object]",
#             "position_relation": "e.g., on top of, next to"
#             }
#         ]
#         }""",
#     "background": [
#         "Describe the background scene.",
#         "Are there any impossible elements?"
#     ],
#     "style": [
#         "What is the artistic style?",
#         "Does it match the prompt requirement?"
#     ],
#     "consistency": [
#         "Are objects logically related? List contradictions if any."
#     ],
#     "alignment": [
#         "Rate image-prompt alignment (1-5). Explain your rating."
#     ]
# }

def get_prompts(file_path):
    with open(file_path, "r") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    return prompts

def get_template_list(template_file):

    templat_list = []
    with open(template_file, "r") as f:
        mllm_prompts = json.load(f)
        for prompt in mllm_prompts:
            templat_list.append(prompt["llava_template"])

    return templat_list

def process_image(image_path, eval_prompts, model, processor, device):
    """处理单张图片并返回结构化结果"""
    image = Image.open(image_path)
    results = {}
    
    # 分别处理每个类别
    for category, prompts in eval_prompts.items():
        category_results = {}
        question_counter = 1
        
        for prompt in prompts:
            # 清理回答中的模板残留
            raw_answer = ask_model(image, prompt, model, processor, device)
            clean_answer = clean_response(raw_answer, prompt)
            
            # 确保每个问题都存在 response
            category_results[f"{category}_q{question_counter}"] = {
                "question": prompt,
                "response": clean_answer if clean_answer else "No relevant information."
            }
            
            question_counter += 1
        
        results[category] = category_results

        print(f"Category '{category}' processed successfully.")

    return results

def clean_response(text, prompt):
    """
    Eliminate unnecessary information from the model response
    """

    # Remove user/assistant tags
    text = re.sub(r"user\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"assistant\s*\n", "", text, flags=re.IGNORECASE)

    # Eliminate the prompt from the response
    text = text.replace(prompt, "").strip()
    return text

# 转换模板为对话格式
def build_mllm_conversation(template):
    # use llava_template to generate conversation
    conversation = []
    conversation.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": template["instructions"]}
        ]
    })
    for q in template["questions"]:
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": q }
            ]
        })


    # 构造对话历史，其中在问题中明确要求输出 JSON 格式，并给出格式模板
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "user"},  # 如果需要包含图像信息
                {"type": "text", "text": (
                    "Please analyze this image and output the results in the following JSON format:\n"
                    "Do not include any additional explanations or redundant text. Make sure the JSON output is valid:\n"
                    f"output_format: {json.dumps(template['response_example'], indent=4)}"
                )}
            ]
        }
    )
        
    return conversation

def ask_model(image, prompt, model, processor, device):
    """
    Ask a question to the model and return the answer
    """
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_template = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        images=image,
        text=prompt_template,
        return_tensors="pt"
    ).to(device, torch.float16)
    
    # Adjust the parameters as needed
    output = model.module.generate(
        **inputs,
        max_new_tokens=200,  # 增加 token 数量
        temperature=0.7,     # 控制随机性
        do_sample=True
    )
    raw_answer = processor.decode(output[0], skip_special_tokens=True)
    return raw_answer

from PIL import Image
import torch



def calculate_hybrid_metrics(image_path, llm_json_output, original_prompt, clip_model):
    """
    增强版混合指标计算
    新增指标：
    - 属性匹配度 (Attribute Match)
    - 关系有效性 (Relation Validity)
    - 逻辑矛盾检测 (Logical Contradictions)
    """
    # 计算CLIP文本-图像相似度
    image = Image.open(image_path)
    image_emb = clip_model.encode(image)
    text_emb = clip_model.encode(original_prompt)
    
    # 转换为 PyTorch Tensor
    image_emb = torch.tensor(image_emb, dtype=torch.float32)
    text_emb = torch.tensor(text_emb, dtype=torch.float32)

    clip_score = torch.nn.functional.cosine_similarity(image_emb, text_emb, dim=0).item()
    
    # 解析LLM输出
    eval_data = llm_json_output
    
    # ================== 1. 语义覆盖度增强 ================== 
    total_elements = 0
    matched_elements = 0
    
    for obj in eval_data["objects"]:
        # 对象存在性
        if obj["present"]:
            matched_elements += 1
            # 属性匹配检查
            if "attributes" in obj:
                total_elements += len(obj["attributes"])
                # 假设原始prompt包含属性（需实现属性提取逻辑）
                matched_attrs = sum(1 for attr_value in obj["attributes"].values() if attr_value != "unknown")
                matched_elements += matched_attrs
        total_elements += 1  # 对象存在性本身算1个元素
    
    semantic_coverage = matched_elements / total_elements if total_elements > 0 else 0

    # ================== 2. 关系有效性 ==================
    valid_relations = sum(1 for rel in eval_data["relations"] if rel["valid"])
    relation_validity = valid_relations / len(eval_data["relations"]) if eval_data["relations"] else 1.0

    # ================== 3. 风格一致性 ==================
    style_score = eval_data["style_consistency"]["score"] / 5  # 归一化到0-1
    hybrid_style_score = 0.6 * style_score + 0.4 * clip_score

    # ================== 4. 逻辑矛盾检测 ==================
    contradiction_penalty = 0
    # 检测常见矛盾模式（可扩展）
    for obj in eval_data["objects"]:
        attrs = obj.get("attributes", {})
        # 示例检测：光照方向矛盾
        if "light source" in attrs:
            if ("sun" in attrs["light source"]) and ("shadow_direction" in attrs):
                if attrs.get("shadow_direction") == "toward light source":
                    contradiction_penalty += 0.2
    logical_consistency = max(0, 1 - contradiction_penalty)

    # ================== 综合得分 ==================
    composite_score = (
        0.4 * semantic_coverage +
        0.3 * relation_validity +
        0.2 * hybrid_style_score +
        0.1 * logical_consistency
    )

    return {
        "clip_score": round(clip_score, 4),
        "semantic_coverage": round(semantic_coverage, 4),
        "relation_validity": round(relation_validity, 4),
        "hybrid_style_score": round(hybrid_style_score, 4),
        "logical_consistency": round(logical_consistency, 4),
        "composite_score": round(composite_score, 4)
    }


CULTURAL_LEXICON = {
    "isaac asimov's foundation": {"type": "literary_work", "tags": ["sci-fi", "library"]},
    "borderlands": {"type": "video_game", "tags": ["cell-shaded", "post-apocalyptic"]}
}

# from sentence_transformers import SentenceTransformer, util

# # 初始化文本相似度模型
# similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# def llava_response_to_json(raw_response):
#     """
#     将LLaVA原始响应转换为结构化JSON
#     技术参考: 
#     - 正则表达式解析 (https://docs.python.org/3/library/re.html)
#     - 语义相似度计算 (https://www.sbert.net)
#     """
#     # 模块1: 分割问题与回答
#     questions = re.findall(r'\[INST\](.*?)\[/INST\]', raw_response, re.DOTALL)
#     answer = raw_response.split('[/INST]')[-1].strip()
    
#     # 模块2: 问题分类与清洗
#     structured_data = {"objects": [], "styles": [], "actions": []}
#     for q in questions:
#         q_clean = q.strip()
#         if not q_clean:
#             continue
        
#         # 识别问题类型 (对象/风格/动作)
#         query_type = "objects"
#         if "style" in q_clean.lower():
#             query_type = "styles"
#         elif "action" in q_clean.lower():
#             query_type = "actions"
        
#         # 提取查询关键词
#         key_phrases = re.findall(r'(le corbusier|impasto brush strokes|wearing|...)', q_clean, re.IGNORECASE)
#         if not key_phrases:
#             continue
        
#         # 模块3: 语义匹配计算
#         query_embed = similarity_model.encode(q_clean)
#         answer_embed = similarity_model.encode(answer)
#         sim_score = float(util.cos_sim(query_embed, answer_embed)[0][0])
        
#         # 构建结构化条目
#         entry = {
#             "query": key_phrases[0],
#             "present": sim_score > 0.3,  # 可调节阈值
#             "evidence": extract_evidence(answer, key_phrases),
#             "score": round(sim_score, 2)
#         }
#         structured_data[query_type].append(entry)
    
#     return json.dumps(structured_data, indent=2)

def extract_evidence(text, keywords):
    """
    基于关键词抽取证据句子
    """
    sentences = re.split(r'[.!?]', text)
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in keywords):
            return sent.strip()
    return "No explicit mention found"


import spacy

def detect_cultural_references(prompt_text):
    """
    文化/虚构实体检测
    技术参考: 
    - 知识图谱构建方法 (Auer et al., 2007) 
    - 实体链接技术 (https://github.com/dice-group/FOX)
    """
    detected = []
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(prompt_text)
    
    # 滑动窗口检测多词实体
    for i in range(len(doc)):
        for j in range(i+1, min(i+5, len(doc))):
            phrase = doc[i:j].text.lower()
            if phrase in CULTURAL_LEXICON:
                detected.append({
                    "phrase": phrase,
                    "metadata": CULTURAL_LEXICON[phrase]
                })
    
    return detected

def get_json_resp(response):
    """
    Extract JSON response from the model output
    """
    match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL) 
    if match:
        json_str = match.group(1)
        try:
            parsed_output = json.loads(json_str)
            return parsed_output
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
    else:
        print("未匹配到 ```json 代码块")
        return None