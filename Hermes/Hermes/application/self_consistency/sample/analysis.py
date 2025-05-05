import re
import numpy as np
from typing import List, Dict, Any

class DynamicAnalyzer:
    def __init__(self):
        self.difficulty_cache = {}
    
    def analyze_all(self, problems: List[Dict], results: Dict) -> Dict:
        return {
            "volume_dynamics": self._analyze_volume(problems),
            "structure_dynamics": self._analyze_structure(results),
            "node_dynamics": self._analyze_nodes(results)
        }
    
    def _analyze_volume(self, problems: List[Dict]) -> Dict:
        difficulty_dist = {}
        for p in problems:
            diff = self._classify_difficulty(p["solution"])
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        return {
            "total_problems": len(problems),
            "difficulty_distribution": difficulty_dist
        }
    
    def _classify_difficulty(self, solution: str) -> str:
        if solution in self.difficulty_cache:
            return self.difficulty_cache[solution]
            
        steps = len(re.findall(r"\n\d+\)", solution))
        operators = len(set(re.findall(r"[+\-*/×÷]", solution)))
        
        if steps >= 5 or "tripling" in solution:
            diff = "expert"
        elif operators > 2 or "tax" in solution.lower():
            diff = "advanced"
        elif steps > 2:
            diff = "intermediate"
        else:
            diff = "basic"
            
        self.difficulty_cache[solution] = diff
        return diff
    
    def _analyze_structure(self, results: Dict) -> Dict:
        step_data = []
        operator_data = []
        
        for model_data in results.values():
            for temp_data in model_data.values():
                for detail in temp_data["details"]:
                    # 分析标准路径
                    std_steps = len(re.findall(r"\n\d+\)", detail["standard_path"])) + 1
                    step_data.append(std_steps)
                    std_ops = len(set(re.findall(r"[+\-*/×÷]", detail["standard_path"])))
                    operator_data.append(std_ops)
                    
                    # 分析自洽路径
                    for path in detail["sc_paths"]:
                        sc_steps = len(re.findall(r"\n\d+\)", path)) + 1
                        step_data.append(sc_steps)
                        sc_ops = len(set(re.findall(r"[+\-*/×÷]", path)))
                        operator_data.append(sc_ops)
        
        return {
            "avg_steps": np.mean(step_data),
            "step_std": np.std(step_data),
            "avg_operators": np.mean(operator_data)
        }
    
    def _analyze_nodes(self, results: Dict) -> Dict:
        token_lengths = []
        unique_tokens = []
        
        for model_data in results.values():
            for temp_data in model_data.values():
                for detail in temp_data["details"]:
                    # 分析所有路径
                    for path in [detail["standard_path"]] + detail["sc_paths"]:
                        tokens = re.findall(r"\b\w+\b", path)
                        token_lengths.append(len(tokens))
                        unique_tokens.append(len(set(tokens)))
        
        return {
            "avg_tokens": np.mean(token_lengths),
            "token_std": np.std(token_lengths),
            "avg_unique_tokens": np.mean(unique_tokens)
        }

# import re
# import numpy as np
# from typing import List, Dict, Any

# class DynamicAnalyzer:
#     def __init__(self):
#         self.difficulty_cache = {}
#         self.token_pattern = re.compile(r"\b[\w\.\$%]+\b")  # 增强token匹配模式

#     def analyze_all(self, problems: List[Dict], results: Dict) -> Dict:
#         return {
#             "volume_dynamics": self._analyze_volume(problems),
#             "structure_dynamics": self._analyze_structure(results),
#             "node_dynamics": self._analyze_nodes(results),
#             "format_compliance": self._analyze_format_compliance(results)  # 新增格式合规分析
#         }
    
#     def _analyze_volume(self, problems: List[Dict]) -> Dict:
#         difficulty_dist = {}
#         for p in problems:
#             # 使用标准solution字段替代原始问题数据
#             diff = self._classify_difficulty(p["solution"])
#             difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
#         return {
#             "total_problems": len(problems),
#             "difficulty_distribution": difficulty_dist
#         }
    
#     def _classify_difficulty(self, solution: str) -> str:
#         if solution in self.difficulty_cache:
#             return self.difficulty_cache[solution]
            
#         # 适配新的步骤计数方式（已严格格式化的步骤）
#         steps = len(re.findall(r"^\d+\)", solution, re.MULTILINE))
#         operators = len(set(re.findall(r"[+\-*/×÷=><≤≥]", solution)))  # 扩展运算符集合
        
#         # 调整难度分类标准
#         if steps >= 5 or any(kw in solution.lower() for kw in ["tripling", "ratio", "compound"]):
#             diff = "expert"
#         elif operators > 2 or any(kw in solution.lower() for kw in ["tax", "discount", "interest"]):
#             diff = "advanced"
#         elif steps > 2 or operators > 1:
#             diff = "intermediate"
#         else:
#             diff = "basic"
            
#         self.difficulty_cache[solution] = diff
#         return diff
    
#     def _analyze_structure(self, results: Dict) -> Dict:
#         step_data = []
#         operator_data = []
        
#         for model_data in results.values():
#             for temp_data in model_data.values():
#                 for detail in temp_data["details"]:
#                     # 统一分析标准路径和自洽路径
#                     for path in [detail["standard_path"]] + detail["sc_paths"]:
#                         # 使用多行模式精确匹配步骤
#                         steps = len(re.findall(r"^\d+\)", path, re.MULTILINE))
#                         step_data.append(steps)
                        
#                         # 扩展运算符检测
#                         ops = set(re.findall(r"[+\-*/×÷=><≤≥]", path))
#                         operator_data.append(len(ops))
        
#         return {
#             "avg_steps": np.mean(step_data),
#             "step_std": np.std(step_data),
#             "max_steps": np.max(step_data) if step_data else 0,
#             "min_steps": np.min(step_data) if step_data else 0,
#             "avg_operators": np.mean(operator_data)
#         }
    
#     def _analyze_nodes(self, results: Dict) -> Dict:
#         token_lengths = []
#         unique_tokens = []
        
#         for model_data in results.values():
#             for temp_data in model_data.values():
#                 for detail in temp_data["details"]:
#                     for path in [detail["standard_path"]] + detail["sc_paths"]:
#                         # 改进的token识别（包含数字和特殊符号）
#                         tokens = self.token_pattern.findall(path)
#                         token_lengths.append(len(tokens))
#                         unique_tokens.append(len(set(tokens)))
        
#         return {
#             "avg_tokens": np.mean(token_lengths),
#             "token_std": np.std(token_lengths),
#             "avg_unique_tokens": np.mean(unique_tokens),
#             "max_tokens": np.max(token_lengths) if token_lengths else 0
#         }

#     def _analyze_format_compliance(self, results: Dict) -> Dict:
#         """新增格式合规性分析"""
#         compliance_stats = {
#             "valid_steps": 0,
#             "invalid_steps": 0,
#             "has_conclusion": 0
#         }
        
#         for model_data in results.values():
#             for temp_data in model_data.values():
#                 for detail in temp_data["details"]:
#                     for path in [detail["standard_path"]] + detail["sc_paths"]:
#                         # 步骤格式验证
#                         steps = re.findall(r"^(\d+)\)", path, re.MULTILINE)
#                         if len(steps) == 0:
#                             compliance_stats["invalid_steps"] += 1
#                             continue
                            
#                         # 检查步骤连续性
#                         expected = 1
#                         valid_sequence = True
#                         for s in steps:
#                             if int(s[0]) != expected:
#                                 valid_sequence = False
#                                 break
#                             expected += 1
                            
#                         if valid_sequence:
#                             compliance_stats["valid_steps"] += 1
#                         else:
#                             compliance_stats["invalid_steps"] += 1
                            
#                         # 结论检查
#                         if re.search(r"^Therefore", path, re.MULTILINE | re.IGNORECASE):
#                             compliance_stats["has_conclusion"] += 1
                            
#         total = compliance_stats["valid_steps"] + compliance_stats["invalid_steps"]
#         return {
#             "compliance_rate": compliance_stats["valid_steps"] / total if total > 0 else 0,
#             "conclusion_rate": compliance_stats["has_conclusion"] / total if total > 0 else 0
#         }
