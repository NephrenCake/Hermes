# Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling

## Tasks Supported
- Factool
    - Code
    - KBQA
- Multi-Turn Conversations

## Install 
```bash
# install dependencies
pip install -r requirements.txt
# install
pip install -e .
```

## Quickstart
Refer to the files in `./example`, for example:
```
python example/factool_code.py
```

## Structure
`Hermes/engine.py`: Including test enigne. 
- Currently we support `OpenLoopEngine`, `CloseLoopEngine` and `SerialEngine`.
- `HybridEngine` will be supported in the future, which can launch multiple tasks in a single evaluation.

`Hermes/taskrunner.py`: Responsible for launching the same type of task.

`Hermes/utils/base/task.py`: `BaseTask` is the base class of task.

`Hermes/utils/dataset.py`: `BaseDataset` is the base class of dataset.

`Hermes/time_recorder.py`: Each test will return a `BenchTimeRecorder`, which contains the jct of every task and every request in a task. It can calculate averge jct. And it will support to plot the execution workflow of a task in the future.



## Add New Task
1. Encapsulate the task into a `BaseTask` class (like `FactoolCodeTask` in `Hermes/tasks/factool/code/task.py`). Each time the task is executed, creating an object of that class and executing `.run()` method.
2. Encapsulate the dataset into a `BaseDataset` class (like `FactoolCodeDataset` in `Hermes/tasks/factool/code/dataset.py`). Implement `.load_data` method and rewrite `.sample_data` method if needed.
3. Add `time_recorder.start_task(task_id)` and `time_recorder.finish_task(task_id)` at the start and the end of .run() method.
4. Add `time_recorder.start_request(task_id, request_id)` and `time_recorder.end_request(task_id, request_id)` at the start at the end of each dag node you want to record time.
5. Registering task class in `Hermes/taskrunner.py` like `'factool_code'`.
6. Registering dataset in `Hermes/dataloader.py` like `'factool_code'`.