from typing import List, Dict
import json
import os 
import pathlib

from Hermes.application.type import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class LangchainMapReduceDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        data_path = os.path.join(curr_dir_name, "../../../../Datasets/langchain/map_reduce/")
        with open(os.path.join(data_path,'MAP_REDUCE_new.json')) as f:
            data = json.load(f)
        return data