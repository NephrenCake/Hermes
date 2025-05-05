import json
from typing import List, Dict
from pathlib import Path

def get_few_shot_examples() -> List[Dict[str, str]]:
    """从prompts目录加载few-shot示例"""
    examples_path = Path("/home/bxdu/sc1/data/gsm8k/prompts/interactive_examples.json")
    with open(examples_path, "r") as f:
        examples_data = json.load(f)
    
    # 解析交互式示例中的问题-解决方案对
    examples = []
    q_prefix = "Question 1:"
    a_prefix = "Answer 1.1:"
    
    raw_text = examples_data["input"]
    segments = raw_text.split("\n\n")
    
    for seg in segments:
        if q_prefix in seg and a_prefix in seg:
            parts = seg.split("\n")
            question = parts[0].replace(q_prefix, "").strip()
            solution = "\n".join([p for p in parts if p.startswith("Answer")])
            examples.append({
                "question": question,
                "solution": solution
            })
    return examples[:2]  # 取前两个作为few-shot示例

def load_gsm8k_dataset(split: str = "test") -> List[Dict[str, str]]:
    """加载GSM8k数据集"""
    dataset_path = Path(f"/home/bxdu/sc1/data/gsm8k/{split}.jsonl")
    problems = []
    
    with open(dataset_path, "r") as f:
        index=0
        for line in f:
            index=index+1
            data = json.loads(line)
            
            # 解析答案中的计算步骤
            solution_steps = []
            answer_lines = data["answer"].split("\n")
            for line in answer_lines:
                if "<<" in line and ">>" in line:
                    step = line.split("=")[-1].split("<")[0].strip()
                    solution_steps.append(step)
            
            # 构建solution格式
            formatted_solution = "Let's solve this step by step:\n"
            for i, step in enumerate(solution_steps, 1):
                formatted_solution += f"{i}) {step}\n"
            formatted_solution += f"\nTherefore, the answer is {data['answer'].split('####')[-1].strip()}."
            
            if(index<30):
                problems.append({
                    "question": data["question"],
                    "answer": data["answer"].split("####")[-1].strip(),
                    "solution": formatted_solution
                })
    return problems

# 保持原有接口兼容性
def load_addsum_examples() -> List[Dict[str, str]]:
    return load_gsm8k_dataset()