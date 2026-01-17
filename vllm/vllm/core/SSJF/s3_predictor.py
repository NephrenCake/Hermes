import os
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class S3Predictor:
    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(cur_dir, "./distilbert-base-uncased")
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(cur_dir, "./checkpoint-4922"),
            num_labels=20,
            ignore_mismatched_sizes=True
        )
        self.bucket_size = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt: str) -> int:
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("输入必须是非空字符串")

        input_data = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"  # 直接返回tensor，避免手动转换
        ).to(self.device)

        predicted_class_id = self.model(
            **input_data
        ).logits.argmax(dim=1).item()

        return predicted_class_id * self.bucket_size

    @torch.no_grad()
    def predict_batch(self, prompts: List[str]) -> List[int]:
        if not isinstance(prompts, list):
            raise ValueError("批量预测输入必须是字符串列表")
        if len(prompts) == 0:
            return []
        if not all(isinstance(p, str) for p in prompts):
            raise ValueError("列表中的所有元素必须是字符串类型")

        inputs = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        predicted_class_id = self.model(
            **inputs
        ).logits.argmax(dim=1).cpu().tolist()

        return [
            pid * self.bucket_size
            for pid in predicted_class_id
        ]


if __name__ == '__main__':
    predictor = S3Predictor()

    # 1. 单条预测（原有功能）
    single_prompt = "Hello, I am a student. I will continue speaking."
    single_result = predictor.predict(single_prompt)
    print(f"单条预测结果: {single_result}")

    # 2. 批量预测（新增功能）
    batch_prompts = [
        # "Hello, I am a student. I will continue speaking.",
        # "This is another test sentence for batch prediction.",
        "A third example to demonstrate batch processing."
    ]
    batch_results = predictor.predict_batch(batch_prompts)
    print(f"\n批量预测结果: {batch_results}")