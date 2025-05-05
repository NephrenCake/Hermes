from typing import List, Dict
import numpy as np
from tqdm import tqdm
from experiment import ChainOfThoughtExperiment
from analysis import DynamicAnalyzer  # 新增导入
from problems_data import load_gsm8k_dataset

def run_experiments(
    model_path: str,
    num_samples: int = 5,
    temperatures: List[float] = [0.7],
    tensor_parallel_size: int = 1
) -> Dict:
    problems = load_gsm8k_dataset()

    experiment = ChainOfThoughtExperiment(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size
    )
    model_results = {}
    for temp in temperatures:
        print(f"\nRunning experiments with temperature {temp}")
        temp_results = {
            "details": [],
            "standard": [],
            "self_consistency": []
        }
        with tqdm(total=len(problems), desc=f"Processing temp={temp}") as pbar:
            for problem in problems:
                eval_result = experiment.evaluate_question(
                    question=problem["question"],
                    correct_answer=problem["answer"],
                    num_samples=num_samples
                )

                temp_results["details"].append(eval_result)
                temp_results["standard"].append(eval_result["standard_correct"])
                temp_results["self_consistency"].append(eval_result["sc_correct"])
                pbar.update(1)
        model_results[temp] = temp_results
    
    return model_results, problems  # 同时返回问题和结果

if __name__ == "__main__":
    model_path = "/home/bxdu/tongyi/model/xxx"
    
    # 运行实验并获取结果和问题数据
    model_data, problems = run_experiments(
        model_path=model_path,
        num_samples=5,
        temperatures=[0.7],
        tensor_parallel_size=1
    )
    
    # 性能结果输出
    print("\n=== Final Results ===")
    for temp, temp_data in model_data.items():
        std_acc = np.mean(temp_data["standard"])
        sc_acc = np.mean(temp_data["self_consistency"])
        print(f"Temperature: {temp}")
        print(f"Standard CoT Accuracy: {std_acc:.2%}")
        print(f"Self-Consistency Accuracy: {sc_acc:.2%}")
        print(f"Improvement: {sc_acc - std_acc:+.2%}")
    
    # 动态性分析
    # analyzer = DynamicAnalyzer()
    # dynamics = analyzer.analyze_all(problems, results)
    
    
    # print("\n=== Dynamic Analysis ===")
    # print(f"1. Volume Dynamics (Total: {dynamics['volume_dynamics']['total_problems']})")
    # for diff, count in dynamics['volume_dynamics']['difficulty_distribution'].items():
    #     print(f"- {diff.capitalize()}: {count} ({(count/len(problems))*100:.1f}%)")
    
    # print("\n2. Structure Dynamics:")
    # print(f"Average Steps: {dynamics['structure_dynamics']['avg_steps']:.1f} ± {dynamics['structure_dynamics']['step_std']:.1f}")
    # print(f"Average Operators: {dynamics['structure_dynamics']['avg_operators']:.1f}")
    
    # print("\n3. Node Dynamics:")
    # print(f"Average Tokens/Path: {dynamics['node_dynamics']['avg_tokens']:.1f} ± {dynamics['node_dynamics']['token_std']:.1f}")
    # print(f"Average Unique Tokens: {dynamics['node_dynamics']['avg_unique_tokens']:.1f}")
