from typing import List, Dict
import json
import os 
import pathlib
import csv

from CTaskBench.utils.base.dataset import BaseDataset

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


class GotDocMergeDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_data()

    def load_data(self) -> List[List]:
        data_path = os.path.join(curr_dir_name, "../../../../Datasets/got/doc_merge/")
        data = []
        with open(os.path.join(data_path,'documents.csv')) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row[0] = int(row[0])
                data.append(row)
        return data[0:50]