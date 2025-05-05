import json
from pathlib import Path
from experiment import ChainOfThoughtExperiment
from analysis import DynamicAnalyzer

# ================== 配置参数 ==================
DATA_DIR = "/home/bxdu/sc1/data/gsm8k"
TEST_FILE = "test.jsonl"
SAMPLE_SIZE = 1  # 测试数据量
MODEL_PATH = "/home/bxdu/tongyi/model"

# ================== 数据加载 ==================
def load_test_samples():
    """加载少量测试数据"""
    samples = []
    with open(Path(DATA_DIR) / TEST_FILE, "r") as f:
        for _ in range(SAMPLE_SIZE):
            line = f.readline()
            data = json.loads(line)
            samples.append({
                "question": data["question"],
                "answer": data["answer"].split("####")[-1].strip(),
                "raw_answer": data["answer"]
            })
    return samples

# ================== 流程验证 ==================
def test_full_pipeline():
    print("=== 开始测试流程 ===")
    
    # # 1. 初始化实验环境
    # print("\n1. 初始化推理引擎...")
    experiment = ChainOfThoughtExperiment(model_path=MODEL_PATH)
    
    # 2. 加载few-shot示例
    print("\n2. 验证prompt加载:")
    print(f"加载few-shot示例数量: {len(experiment.few_shot_examples)}")
    print("示例问题:", experiment.few_shot_examples)
    
    # 3. 加载测试数据
    print("\n3. 加载测试数据集:")
    test_data = load_test_samples()
    print(f"成功加载 {len(test_data)} 条测试数据")
    print("首条测试问题:", test_data[0]["question"])
    
    # 4. 生成CoT提示
    print("\n4. 验证提示生成:")
    sample_question = test_data[0]["question"]
    prompt = experiment.generate_cot_prompt(sample_question)
    # print(prompt)
    # print("生成提示片段:\n" + prompt + "...\n")
    
    # result=experiment.generate_reasoning_paths(sample_question,num_samples=1)
    # print(result)
    
    # 5. 执行推理验证
    print("\n5. 执行推理验证:")
    for i, problem in enumerate(test_data, 1):
        print(f"\n处理问题 {i}/{len(test_data)}")
        print("问题:", problem["question"])
        
        # 生成推理路径
        paths = experiment.generate_reasoning_paths(problem["question"], num_samples=1)
        #print("生成推理路径:", paths[0][:50] + "...")
        print("path为:",paths)
        
        # 验证答案
        result = experiment.evaluate_question(
            question=problem["question"],
            correct_answer=problem["answer"]
        )
        print(result)
        print("预测答案:", result["standard_answer"])
        print("正确答案:", problem["answer"])
        print("是否正确:", result["standard_correct"])
        
        analyzer=DynamicAnalyzer()
        dynamics = analyzer.analyze_all(test_data, result)
        
        print("\n2. Structure Dynamics:")
        print(f"Average Steps: {dynamics['structure_dynamics']['avg_steps']:.1f} ± {dynamics['structure_dynamics']['step_std']:.1f}")
        print(f"Average Operators: {dynamics['structure_dynamics']['avg_operators']:.1f}")
        

if __name__ == "__main__":
    test_full_pipeline()
    print("\n=== 测试完成 ===")