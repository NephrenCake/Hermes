from typing import List, Dict
import json
import os 
import pathlib

from CTaskBench.utils.base.dataset import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class HuggingGPTDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        data_path = os.path.join(curr_dir_name, "../../../Datasets/hugginggpt/")
        dataset_name = "CHAIN_SIMPLE"
        with open(os.path.join(data_path,f'{dataset_name}.json')) as f:
            data = json.load(f)
        return data