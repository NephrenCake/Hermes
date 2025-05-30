from typing import List, Dict
import json
import os 
import pathlib

from Hermes.application.type import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class ReactFeverDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        data_path = os.path.join(curr_dir_name, "../../../../Datasets/react/fever/")
        with open(os.path.join(data_path,'FEVER.json')) as f:
            data = json.load(f)
        for i in range(len(data)):
            data[i]['idx'] = i
        return data