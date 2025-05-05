from typing import List, Dict
import json
import os 
import pathlib

from Hermes.application.type import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class CodeFeedbackDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        data_path = os.path.join(curr_dir_name, "../../../Datasets/code_feedback/")
        # dataset_name = "HumanEval"
        dataset_name = "HumanEval_with_multiturns"
        # dataset_name = "20single_25multi"
        with open(os.path.join(data_path,f'{dataset_name}.json')) as f:
            data = json.load(f)
        # data.pop(21)
        # data.pop(17)
        # data.pop(16)
        # data.pop(15)
        # data.pop(8)
        # data.pop(2)  # todo 遇到越界就换 llama3 或者打开注释
        return data
    