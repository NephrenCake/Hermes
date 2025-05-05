from typing import List, Dict
import json
import os 
import pathlib
import csv

from Hermes.application.type import BaseDataset

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
        # data.pop(21)
        # data.pop(7)  # todo 遇到越界就换 llama3 或者打开注释，此处错误case未完全探明
        return data[0:50]