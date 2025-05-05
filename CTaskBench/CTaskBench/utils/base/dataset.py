

class BaseDataset:
    def __init__(self) -> None:
        self.data = None

    def load_data(self):
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)