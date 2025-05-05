import numpy as np

from CTaskBench.utils.base.dataset import BaseDataset
from CTaskBench.tasks.factool.code.dataset import FactoolCodeDataset
from CTaskBench.tasks.factool.kbqa.dataset import FactoolKbqaDataset
from CTaskBench.tasks.factool.math.dataset import FactoolMathDataset
from CTaskBench.tasks.react.fever.dataset import ReactFeverDataset
from CTaskBench.tasks.react.alfw.dataset import ReactAlfwDataset
from CTaskBench.tasks.got.doc_merge.dataset import GotDocMergeDataset
# from CTaskBench.tasks.got.key_count.dataset import GotKeyCountDataset
# from CTaskBench.tasks.got.set_intersection.dataset import GotSetIntersectionDataset
# from CTaskBench.tasks.got.sort.dataset import GotSortDataset
from CTaskBench.tasks.multiturn_conversations.dataset import MultiTurnConversationDataset
from CTaskBench.tasks.langchain.map_reduce.dataset import LangchainMapReduceDataset 
from CTaskBench.tasks.code_feedback.dataset import CodeFeedbackDataset
from CTaskBench.tasks.hugginggpt.dataset import HuggingGPTDataset
from CTaskBench.logger import init_logger

logger = init_logger(__name__)

Dataset_Dict = {
    'factool_code': FactoolCodeDataset,
    'factool_kbqa': FactoolKbqaDataset,
    'factool_math': FactoolMathDataset,
    'react_fever' : ReactFeverDataset,
    'react_alfw' : ReactAlfwDataset,
    'got_docmerge': GotDocMergeDataset,
    # 'got_keycount': GotKeyCountDataset,
    # 'got_setintersection': GotSetIntersectionDataset,
    # 'got_sort': GotSortDataset,
    'multiturn_conversations': MultiTurnConversationDataset,
    'langchain_mapreduce': LangchainMapReduceDataset,
    'code_feedback': CodeFeedbackDataset,
    'hugginggpt': HuggingGPTDataset,
}


class DataLoader:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.dataset = self._load_data()
        print(f"Loaded {self.task_name} dataset with {len(self.dataset)} samples.")

    def _load_data(self) -> BaseDataset:
        if self.task_name in Dataset_Dict:
            return Dataset_Dict[self.task_name]()
        else:
            raise NameError("Invalid task name '%s'", self.task_name)
        
    def sample_data(self):
        index = np.random.randint(len(self.dataset))
        return self.dataset[index]
    
    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
        
