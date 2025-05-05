from problems_data import load_addsum_examples,load_gsm8k_dataset,get_few_shot_examples
# 测试数据加载
test_problems = load_gsm8k_dataset()
print(f"Loaded {len(test_problems)} problems")
print("Sample problem:")
print(test_problems[1]["question"])
print("\nSolution:")
print(test_problems[1]["solution"])

# 测试few-shot示例
few_shots = get_few_shot_examples()
print("\nFew-shot examples:")
for ex in few_shots:
    print(f"Q: {ex['question']}")
    print(f"A: {ex['solution'][:50]}...")