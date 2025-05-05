import numpy as np

from Hermes.application.type import BaseDataset
from Hermes.application.factool.code.dataset import FactoolCodeDataset
from Hermes.application.factool.kbqa.dataset import FactoolKbqaDataset
from Hermes.application.factool.math.dataset import FactoolMathDataset
from Hermes.application.react.fever.dataset import ReactFeverDataset
from Hermes.application.react.alfw.dataset import ReactAlfwDataset
from Hermes.application.got.doc_merge.dataset import GotDocMergeDataset
# from Hermes.application.got.key_count.dataset import GotKeyCountDataset
# from Hermes.application.got.set_intersection.dataset import GotSetIntersectionDataset
# from Hermes.application.got.sort.dataset import GotSortDataset
# from Hermes.application.multiturn_conversations.dataset import MultiTurnConversationDataset
from Hermes.application.langchain.map_reduce.dataset import LangchainMapReduceDataset
from Hermes.application.code_feedback.dataset import CodeFeedbackDataset
from Hermes.application.hugginggpt.dataset import HuggingGPTDataset
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)

Dataset_Dict = {
    'factool_code': FactoolCodeDataset,
    'factool_kbqa': FactoolKbqaDataset,
    'factool_math': FactoolMathDataset,
    'react_fever': ReactFeverDataset,
    'react_alfw': ReactAlfwDataset,
    'got_docmerge': GotDocMergeDataset,
    # 'got_keycount': GotKeyCountDataset,
    # 'got_setintersection': GotSetIntersectionDataset,
    # 'got_sort': GotSortDataset,
    # 'multiturn_conversations': MultiTurnConversationDataset,
    'langchain_mapreduce': LangchainMapReduceDataset,
    'code_feedback': CodeFeedbackDataset,
    'hugginggpt': HuggingGPTDataset,
}


class DataLoader:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.dataset = self._load_data()
        logger.info(f"Loaded {self.task_name} dataset with {len(self.dataset)} samples.")

    def _load_data(self) -> BaseDataset:
        if self.task_name in Dataset_Dict:
            return Dataset_Dict[self.task_name]()
        else:
            raise NameError("Invalid task name '%s'", self.task_name)

    def sample_idx(self):
        return np.random.randint(len(self.dataset))

    def sample_data(self):
        return self.dataset[self.sample_idx()]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
