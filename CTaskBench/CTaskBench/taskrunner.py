from CTaskBench.utils.base.task import BaseTask
from CTaskBench.tasks.factool.code.task import FactoolCodeTask
from CTaskBench.tasks.factool.kbqa.task import FactoolKbqaTask
from CTaskBench.tasks.factool.math.task import FactoolMathTask
from CTaskBench.tasks.react.fever.task import ReactFeverTask
from CTaskBench.tasks.react.alfw.task import ReactAlfwTask
from CTaskBench.tasks.got.doc_merge.task import GotDocMergeTask
# from CTaskBench.tasks.got.key_count.task import GotKeyCountTask
# from CTaskBench.tasks.got.set_intersection.task import GotSetIntersectionTask
# from CTaskBench.tasks.got.sort.task import GotSortTask
from CTaskBench.tasks.multiturn_conversations.task import MultiTurnConversationTask
from CTaskBench.tasks.langchain.map_reduce.task import LangchainMapReduceTask
from CTaskBench.tasks.code_feedback.task import CodeFeedbackTask
from CTaskBench.tasks.hugginggpt.task import HuggingGPTTask


Task_Dict = {
    'factool_code': FactoolCodeTask,
    'factool_kbqa': FactoolKbqaTask,
    'factool_math': FactoolMathTask,
    'react_fever' : ReactFeverTask,
    'react_alfw' : ReactAlfwTask,
    'got_docmerge': GotDocMergeTask,
    # 'got_keycount': GotKeyCountTask,
    # 'got_setintersection': GotSetIntersectionTask,
    # 'got_sort': GotSortTask,
    'multiturn_conversations': MultiTurnConversationTask,
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

    async def launch_task(
        self,
        **args,):
        task = self.create_task(**args)
        return await task.run()
