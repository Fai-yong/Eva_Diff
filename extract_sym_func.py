
# dynamically generate dimentional-semantic-teplate-prompt for LLava
# import spacy
# import json
# def extract_dynamic_dimensions(prompt):

#     # Load the spaCy model
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(prompt)
    
#     # 提取对象层
#     objects = []
#     for ent in doc.ents:
#         # 提取命名实体作为对象
#         objects.append({
#             "name": ent.text,
#             "type": ent.label_
#         })

#     # 提取属性和关系（如“穿着”，“设计师是”）
#     attributes = {}
#     relations = []
#     for token in doc:
#         # 检测动作关系（如动词和其主语/宾语的关系）
#         if token.dep_ in ['nsubj', 'dobj', 'prep']:
#             if token.dep_ == 'nsubj':
#                 subject = token.text
#                 verb = token.head.text
#                 object_ = token.head.rights[0].text if len(token.head.rights) > 0 else None
#                 relations.append({
#                     "subject": subject,
#                     "verb": verb,
#                     "object": object_
#                 })
            
#             # 额外的属性提取（如“详细的”，“优雅的”）
#             if token.dep_ == 'amod':
#                 attributes[token.head.text] = token.text
#     print(relations)

#     # 提取风格信息（根据描述的艺术风格）
#     style = {}
#     for token in doc:
#         if token.text.lower() in ["victorian", "hyper detailed", "concept art", "intricate", "elegant", "dark"]:
#             style[token.text] = True
    
#     # 提取背景信息（如“背景”，“在后面”）
#     background = {}
#     for token in doc:
#         if token.dep_ == 'prep' and token.head.dep_ in ['pobj', 'dobj']:
#             background["location"] = token.head.text
    
#     return objects, attributes, relations, style, background

# # 动态生成自适应模板
# def generate_dynamic_template(objects, attributes, relations, style, background):
#     template = {}

#     # 动态生成对象层模板
#     if objects:
#         template["objects_layer"] = {"objects": []}
#         for obj in objects:
#             template["objects_layer"]["objects"].append({
#                 "name": obj["name"],
#                 "type": obj["type"]
#             })
    
#     # 动态生成属性层模板
#     if attributes:
#         template["attributes_layer"] = attributes
    
#     # 动态生成关系层模板
#     if relations:
#         template["relations_layer"] = {"relations": relations}
    
#     # 动态生成风格层模板
#     if style:
#         template["style_layer"] = style
    
#     # 动态生成背景层模板
#     if background:
#         template["background_layer"] = background
    
#     return json.dumps(template, indent=4)

import spacy
from spacy.matcher import Matcher
from allennlp.predictors import Predictor
import json

# 初始化模型
nlp = spacy.load("en_core_web_lg")
matcher = Matcher(nlp.vocab)
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# 预定义艺术运动词库 (示例)
PREDEFINED_ART_MOVEMENTS = {"renaissance", "cyberpunk", "vaporwave", "brutalist"}

def parse_prompt(prompt_text):
    """
    Multi-dimensional semantic parser
    Technical reference:
    - SpaCy Dependency resolution (https://spacy.io)
    - AllenNLP semantic role labeling (https://github.com/allenai/allennlp)
    """
    doc = nlp(prompt_text)
    elements = {"objects": [], "attributes": {}, "relations": [], "styles": {"artists": [], "movements": [], "mediums": [], "aesthetics": []}}
    
    # Rule 1: Compound object recognition (e.g. "anthropomorphic anthro male fox fursona")
    compound_pattern = [{"POS": "ADJ"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "+"}]
    matcher.add("COMPOUND_OBJECT", [compound_pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        elements["objects"].append({"text": span.text, "type": "compound"})
    
    # Rule 2: Artist/style references (e.g. "by Gustav Klimt")
    artist_pattern = [{"LOWER": "by"}, {"POS": "PROPN", "OP": "+"}]
    
    matcher.add("ARTIST_REF", [artist_pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        elements["styles"]["artists"].append(doc[start+1:end].text)
    
    # Semantic role annotation extracts complex relationships (e.g. "fighting the horrors")
    srl_result = srl_predictor.predict(sentence=prompt_text)
    for verb_info in srl_result["verbs"]:
        elements["relations"].append({
            "verb": verb_info["verb"],
            "args": verb_info["description"]
        })
    
    return elements

def generate_eval_template(parsed_data):
    """
    Generate an evaluation question template based on the analysis results
    Technical reference:
    TIFA Evaluation Framework (https://tifa-benchmark.github.io) (CVPR 2023)
    - Chain - of - Thought Prompting (https://arxiv.org/abs/2201.11903)
    """
    questions = []
    
    # Generate object-related questions
    for obj in parsed_data["objects"]:
        obj_text = obj["text"]
        questions.append(f"Is there a {obj_text} in the image? Describe its key attributes.(Used to fill 'objects')")
    
    # Generate style-related questions
    if parsed_data["styles"]["artists"]:
        artists = ", ".join(parsed_data["styles"]["artists"])
        questions.append(f"Does the image reflect the style of {artists}? Provide specific evidence.(Used to fill 'style_consistency')")
    
    # Generate relation-related questions
    for rel in parsed_data["relations"]:
        questions.append(f"Does the action '{rel['verb']}' appear in the image? Describe how it's represented.(Used to fill 'relations')")
    
    # Generate additional questions based on the prompt
    # Final prompt template
    template = {
        "instructions": (
            "You are a vision-language reasoning model. Your task is to analyze the image based on the following questions. "
            "After answering internally, summarize your analysis by outputting a structured JSON result following the format shown in 'response_example'. "
            "Only output the final JSON. Do not include any extra explanations or commentary."
        ),
        "questions": questions,
        "response_example": {
        "objects": [
            {
            "name": "sample_object_1",
            "present": True,
            "attributes": {
                "attribute_1": True,
                "attribute_2": False
            }
            },
            {
            "name": "sample_object_2",
            "present": False,
            "attributes": {}
            }
        ],
        "style_consistency": {
            "score": 3,
            "evidence": "Some visual elements loosely follow the referenced art style, such as atmospheric lighting and intricate details, but overall consistency is moderate."
        },
        "relations": [
            {
            "verb": "sample_action_1",
            "valid": True
            },
            {
            "verb": "sample_action_2",
            "valid": False
            }
        ]
        }
    }
    return template



