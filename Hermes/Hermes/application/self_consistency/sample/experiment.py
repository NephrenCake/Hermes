from typing import List, Dict
from vllm import LLM, SamplingParams


class ChainOfThoughtExperiment:
    def __init__(self,
                 model_name: str = "llama",
                 model_path: str = "/home/bxdu/tongyi/model",
                 tensor_parallel_size: int = 1):

        # 初始化模型路径
        self.model_path = model_path if model_path else model_name
        print(f"Loading model from: {self.model_path}")

        # 配置vLLM引擎
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.6,
            max_model_len=4096,
            dtype="float16",
            trust_remote_code=True
        )

        # 生成参数配置
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            n=1
        )

        # 加载示例并预处理
        self.few_shot_examples = self._load_processed_examples()

    def _load_processed_examples(self) -> List[Dict[str, str]]:
        """加载并预处理few-shot示例"""
        from problems import get_few_shot_examples
        raw_examples = get_few_shot_examples()
        return [{
            "question": ex["question"],
            "solution": self._format_solution(ex["solution"])
        } for ex in raw_examples]

    def _format_solution(self, raw_solution: str) -> str:
        """标准化解决方案格式"""
        formatted = []
        for line in raw_solution.strip().split("\n"):
            line = line.strip()
            # 保留符合规范的步骤行
            if any(line.startswith(f"{i})") for i in range(1, 10)):
                formatted.append(line)
            # 规范最终结论行
            elif line.lower().startswith("therefore"):
                formatted.append("\n" + line.capitalize())
        return "\n".join(formatted)

    def generate_cot_prompt(self, question: str) -> str:
        """生成严格格式化的CoT提示"""
        format_rules = (
            "Solve the math problem step by step following these rules:\n"
            "1. Start each step with number and ')', e.g. '1)'\n"
            "2. Use exactly one step per line\n"
            "3. End with 'Therefore...' conclusion\n\n"
            "Correct Format Examples:\n"
        )

        prompt = format_rules

        # 添加规范化示例
        for idx, ex in enumerate(self.few_shot_examples, 1):
            prompt += f"\n--- Example {idx} ---\n"
            prompt += f"Problem: {ex['question']}\n"
            prompt += f"Solution:\n{ex['solution']}\n"

        # 添加待解决问题
        prompt += (
            f"\n--- Your Task ---\n"
            f"Problem: {question}\n"
            f"Solution:\n"
            f"1)"  # 强制起始格式
        )
        return prompt

    async def generate_reasoning_paths(self,
                                 question: str,
                                 num_samples: int = 5,
                                 temperature: float = 0.7) -> List[str]:
        """生成推理路径"""
        prompt = self.generate_cot_prompt(question)

        # 更新生成参数
        self.sampling_params.temperature = temperature
        self.sampling_params.n = num_samples

        # 批量生成并后处理
        outputs = self.llm.generate([prompt], self.sampling_params)  # todo
        return [self._postprocess_generation(o.outputs[0].text)
                for o in outputs]

    def _postprocess_generation(self, text: str) -> str:
        """后处理生成的文本"""
        processed = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # 保留有效步骤
            if any(line.startswith(f"{i})") for i in range(1, 10)):
                processed.append(line)
            # 捕获最终答案
            elif line.lower().startswith("therefore"):
                processed.append("\n" + line)
                break  # 提前终止
        return "\n".join(processed).strip()

    def evaluate_question(self,
                          question: str,
                          correct_answer: str,
                          num_samples: int = 1) -> Dict:
        """评估问题解决情况"""
        from .evaluation import (
            extract_final_answer,
            majority_vote,
            check_answer_correctness
        )

        # 标准CoT路径
        standard_path = self.generate_reasoning_paths(question, 1)[0]
        standard_answer = extract_final_answer(standard_path)

        # 自洽方法
        sc_paths = self.generate_reasoning_paths(question, num_samples)
        # sc_paths = [self.generate_reasoning_paths(question, 1)
        #             for _ in range(num_samples)]
        # asyncio.gather(sc_paths)
        sc_answers = [extract_final_answer(p) for p in sc_paths]
        sc_answer = majority_vote(sc_answers)

        return {
            'standard_correct': check_answer_correctness(standard_answer, correct_answer),
            'sc_correct': check_answer_correctness(sc_answer, correct_answer),
            'standard_path': standard_path,
            'sc_paths': sc_paths,
            'standard_answer': standard_answer,
            'sc_answer': sc_answer
        }
