from Hermes.application.type import BaseTask
from Hermes.application.factool.code.task import FactoolCodeTask
from Hermes.application.factool.kbqa.task import FactoolKbqaTask
from Hermes.application.factool.math.task import FactoolMathTask
from Hermes.application.react.fever.task import ReactFeverTask
from Hermes.application.react.alfw.task import ReactAlfwTask
from Hermes.application.got.doc_merge.task import GotDocMergeTask
# from Hermes.application.got.key_count.task import GotKeyCountTask
# from Hermes.application.got.set_intersection.task import GotSetIntersectionTask
# from Hermes.application.got.sort.task import GotSortTask
# from Hermes.application.multiturn_conversations.task import MultiTurnConversationTask
from Hermes.application.langchain.map_reduce.task import LangchainMapReduceTask
from Hermes.application.code_feedback.task import CodeFeedbackTask
from Hermes.application.hugginggpt.task import HuggingGPTTask

Task_Dict = {
    'factool_code': FactoolCodeTask,
    'factool_kbqa': FactoolKbqaTask,
    'factool_math': FactoolMathTask,
    'react_fever': ReactFeverTask,
    'react_alfw': ReactAlfwTask,
    'got_docmerge': GotDocMergeTask,
    # 'got_keycount': GotKeyCountTask,
    # 'got_setintersection': GotSetIntersectionTask,
    # 'got_sort': GotSortTask,
    # 'multiturn_conversations': MultiTurnConversationTask,
    'langchain_mapreduce': LangchainMapReduceTask,
    'code_feedback': CodeFeedbackTask,
    'hugginggpt': HuggingGPTTask,
}


class TaskRunner:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    def create_task(self, **args) -> BaseTask:
        if self.task_name in Task_Dict:
            # print(self.task_name)
            return Task_Dict[self.task_name](**args)
        else:
            raise NameError("Invalid task name '%s'", self.task_name)

    async def launch_task(self, **args):
        task = self.create_task(**args)
        return await task.run()
