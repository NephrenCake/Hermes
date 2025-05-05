from typing import List, Dict
import random

"""
Ideally, test it on benchmarks like gsm8k, reclor, and so on.
These examples are just for educational purposes
"""

def get_few_shot_examples() -> List[Dict[str, str]]:
    return [
        {
            "question": "Joan found 70 seashells on the beach. She gave Sam some of her seashells. She has 27 seashells. How many seashells did she give to Sam?",
            "solution": """Let's solve this step by step:
1) Joan started with 70 seashells
2) She now has 27 seashells
3) To find how many she gave away: 70 - 27 = 43 seashells

Therefore, Joan gave Sam 43 seashells."""
        }
    ]

# def load_addsum_examples() -> List[Dict[str, str]]:
#     return [
#         {
#             "question": "There are 7 cards on the table. James took 3 cards. How many cards are still on the table?",
#             "answer": "4",
#             "solution": """1) Initially, there are 7 cards on the table
# 2) James took 3 cards away
# 3) To find remaining cards: 7 - 3 = 4

# Therefore, there are 4 cards still on the table."""
#         },
#         {
#             "question": "John has 5 marbles. He found 3 more marbles at the park and his friend gave him 4 marbles. How many marbles does John have now?",
#             "answer": "12",
#             "solution": """1) Initially, John has 5 marbles
# 2) He found 3 more marbles: 5 + 3 = 8 marbles
# 3) His friend gave him 4 more: 8 + 4 = 12 marbles

# Therefore, John has 12 marbles now."""
#         },
#         {
#             "question": "Sara had 8 dollars. She spent 3 dollars on lunch and then earned 5 dollars helping her neighbor. How many dollars does Sara have now?",
#             "answer": "10",
#             "solution": """1) Sara started with 8 dollars
# 2) After spending 3 dollars: 8 - 3 = 5 dollars
# 3) After earning 5 more: 5 + 5 = 10 dollars

# Therefore, Sara has 10 dollars now."""
#         }
#     ]

def load_addsum_examples() -> List[Dict[str, str]]:
    examples = []
    
    # 基础层：简单加减法（保持原有难度）
    examples.extend([
        {
            "question": "There are 7 cards on the table. James took 3 cards. How many cards are still on the table?",
            "answer": "4",
            "solution": """1) Initial cards: 7
2) Cards taken: 3
3) Remaining: 7 - 3 = 4"""
        },
        {
            "question": "John has 5 marbles. He found 3 more marbles and received 4 from a friend. Total?",
            "answer": "12",
            "solution": """1) Initial: 5
2) Found: 5 + 3 = 8
3) Received: 8 + 4 = 12"""
        }
    ])
    
    # 进阶层：引入乘除和复杂运算
    for _ in range(3):
        a, b, c = random.randint(10, 50), random.randint(5, 20), random.randint(1, 5)
        examples.append({
            "question": f"A store had {a} apples. Sold {b} in the morning and {c} in the afternoon. Then received a shipment tripling remaining stock. Total apples now?",
            "answer": str(3 * (a - b - c)),
            "solution": f"""1) Initial: {a}
2) Morning sales: {a} - {b} = {a-b}
3) Afternoon sales: {a-b} - {c} = {a-b-c}
4) Shipment: 3 × {a-b-c} = {3*(a-b-c)}"""
        })
    
    # 挑战层：混合运算和分数
    money_values = [(18.5, 4.99, 3), (25.0, 7.5, 2)]
    for initial, price, quantity in money_values:
        tax_rate = random.choice([0.08, 0.1])
        total = (price * quantity + initial) * (1 + tax_rate)
        examples.append({
            "question": f"Sarah had ${initial}. Bought {quantity} items at ${price} each with {tax_rate*100}% tax. How much money left?",
            "answer": f"${total - (price * quantity * (1 + tax_rate)) :.2f}",
            "solution": f"""1) Initial: ${initial}
2) Item cost: {quantity} × ${price} = ${price*quantity}
3) Tax: {tax_rate*100}% of ${price*quantity} = ${price*quantity*tax_rate:.2f}
4) Total spent: ${price*quantity} + ${price*quantity*tax_rate:.2f} = ${price*quantity*(1+tax_rate):.2f}
5) Remaining: ${initial} - ${price*quantity*(1+tax_rate):.2f} = ${initial - price*quantity*(1+tax_rate):.2f}"""
        })
    
    # 专家层：多步骤逻辑陷阱
    expert_problems = [
        {
            "question": "A clock gains 3 minutes every hour. After being set at 12:00, what's the actual time when it shows 3:00 PM?",
            "answer": "2:48 PM",
            "solution": """1) Elapsed shown time: 3 hours = 180 minutes
2) Actual time passed: 180 × (60/63) = 171.43 minutes
3) 171.43 minutes = 2 hours 51.43 minutes
4) Actual time: 12:00 + 2h51m26s ≈ 2:51 PM"""
        },
        {
            "question": "If 3 workers build 2 walls in 4 days, how many days for 7 workers to build 5 walls?",
            "answer": "4.29 days",
            "solution": """1) Work rate: (3 workers) → 2 walls / 4 days = 0.5 walls/day
2) Per worker rate: 0.5 / 3 ≈ 0.1667 walls/worker/day
3) Required work: 5 walls
4) Daily output with 7 workers: 7 × 0.1667 ≈ 1.1667 walls/day
5) Days needed: 5 / 1.1667 ≈ 4.29 days"""
        }
    ]
    examples.extend(expert_problems)
    
    # 随机生成复杂场景
    for _ in range(2):
        base = random.randint(10, 20)
        exponent = random.randint(2, 3)
        modifier = random.choice([1.5, 2, 3])
        examples.append({
            "question": f"A bacterial culture triples every {base} hours. Starting with 100 colonies, how many after {exponent*base} hours?",
            "answer": str(100 * (3 ** exponent)),
            "solution": f"""1) Tripling cycle: every {base} hours
2) Number of cycles: {exponent*base} / {base} = {exponent}
3) Final amount: 100 × 3^{exponent} = {100*(3**exponent)}"""
        })
    
    return examples

