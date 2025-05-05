import json
import os
import numpy as np
from transformers import PreTrainedTokenizerBase, LlamaTokenizerFast
from typing import List, Tuple, Dict
import random
import time

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Dict]:

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 4]
    promptss = [[data["conversations"][2*i]["value"] for i in range(len(data["conversations"])//2)] for data in dataset]
    answerss = [[data["conversations"][2*i+1]["value"] for i in range(len(data["conversations"])//2)] for data in dataset]

    promptss_ids = [tokenizer(prompts).input_ids for prompts in promptss]
    answerss_len = [[len(answer) for answer in tokenizer(answers).input_ids] for answers in answerss]

    # Filter out too long sequences.
    filtered_dataset: List[List[Tuple[str, str, int, int]]] = []
    for i in range(len(promptss)):
        session_len = 0
        session_list = []
        for j in range(len(promptss[i])):
            prompt_len = len(promptss_ids[i][j])
            output_len = answerss_len[i][j]
            if prompt_len < 4 or output_len < 4:
                break
            session_len += prompt_len
            session_len += output_len
            if session_len > 4096:
                break
            session_list.append((promptss[i][j], answerss[i][j], prompt_len, output_len))
        if len(session_list) >= 2:
            filtered_dataset.append(session_list)

    # Sample the requests.
    num_requests = min(num_requests, len(filtered_dataset))
    sampled_requests = random.sample(filtered_dataset, num_requests)

    dataset = []
    for requests in sampled_requests:
        data = {}
        data['num_turns'] = len(requests)
        data['conversations'] = []
        for conversation in requests:
            data['conversations'].append({"value": conversation[0],
                                          "length": conversation[2]})
            data['conversations'].append({"value": conversation[1],
                                          "length": conversation[3]})
        dataset.append(data)
    print(f"sampled {len(dataset)} datas from {dataset_path}")
    return dataset


def save_data(
        sampled_data: List[Dict],
        save_path: str,
        ) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'SHAREGPT_DATA.json'), 'w') as f:
        json.dump(sampled_data, f, indent=4)
    print(f"save sampled data in {save_path}")


def main():
    random.seed(0)

    # sample and save 
    dataset_path = "./Datasets/multiturn_conversations/ShareGPT_V3_unfiltered_cleaned_split.json"
    tokenizer_path = "/home/zgan/Models/Llama-2-7b-chat-hf/"
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    sampled_data = sample_requests(dataset_path, 1000, tokenizer)
    save_path = "./Datasets/multiturn_conversations"
    save_data(sampled_data, save_path)

    # load and sample
    # dataset_path = "./Datasets/multiturn_conversations/SHAREGPT_DATA.json"
    # with open(dataset_path, 'r') as f:
    #     data = json.load(f)
    # print(data[1])
    

if __name__ == "__main__":
    main()