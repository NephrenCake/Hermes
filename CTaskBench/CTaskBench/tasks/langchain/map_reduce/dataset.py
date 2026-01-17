from typing import List, Dict
import json
import os
import pathlib

from CTaskBench.utils.base.dataset import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class LangchainMapReduceDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        data_path = os.path.join(curr_dir_name, "../../../../Datasets/langchain/map_reduce/")
        with open(os.path.join(data_path, 'MAP_REDUCE_new.json')) as f:
            data = json.load(f)
        return data


if __name__ == '__main__':
    dataset = LangchainMapReduceDataset()
    print(len(dataset))
    for i, d in enumerate(dataset):
        # print(d.keys())
        # print(d["generate_summary"]["usage"]["total_tokens"])
        # print(d["collapse_summaries"]["usage"][0]["total_tokens"] if d["collapse_summaries"]["usage"] else [])
        # print(d["generate_final_summary"]["usage"]["total_tokens"])

        v = max((d["generate_summary"]["usage"]["total_tokens"] +
                 (d["collapse_summaries"]["usage"][0]["total_tokens"] if d["collapse_summaries"]["usage"] else []) +
                 [d["generate_final_summary"]["usage"]["total_tokens"]]))
        if v > 4000:
            print(i, d["generate_summary"]["input"])
