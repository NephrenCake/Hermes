import random

import math
import os

import numpy as np
import yaml
import ast
from typing import List, Dict
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio
import time

# from transformers import AutoTokenizer
# from CTaskBench.time_recorder import BenchTimeRecorder
# from langchain_openai import ChatOpenAI
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from CTaskBench.logger import init_logger
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.const import prefill_time, decode_time

logger = init_logger(__name__)


class LangchainMapReduceTask(BaseTask):
    def __init__(
        self,
        **base_args,
    ):
        super().__init__(**base_args)
        
        self.request_cnt = 0
        self.gs_cnt = 0
        self.cs_cnt = 0
        self.sc_cnt = 0

        # self.tokenizer = AutoTokenizer.from_pretrained("/home/zgan/Models/Llama-2-7b-chat-hf")

        # os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # os.environ["LANGCHAIN_API_KEY"] = "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"

        # self.llm = ChatOpenAI(temperature=0, 
        #                 model_name=self.model_name,
        #                 api_key="sk-3792751bf6634f20bd8925701c4ae64e",
        #                 base_url="http://localhost:8000/v1",
        #                 )

        # map_prompt = ChatPromptTemplate.from_messages(
        #     [("system", "Write a concise summary of the following:\\n\\n{context}")]
        # )
        # self.map_chain = map_prompt | self.llm | StrOutputParser()

        # reduce_template = """
        # The following is a set of summaries:
        # {docs}
        # Take these and distill it into a final, consolidated summary
        # of the main themes.
        # """
        # reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
        # self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()

        # split_docs = text_splitter.split_documents(docs)
        # print(f"Generated {len(split_docs)} documents.")

        self.token_max = 1000

        gs_parallelism = len(self.data["generate_summary"]["input"])
        gs_input_len = self.data["generate_summary"]["usage"]["prompt_tokens"]
        gs_output_len = self.data["generate_summary"]["usage"]["completion_tokens"]
        hint = {
            "generate_summary": {"parallelism":gs_parallelism,
                                  "length": [(gs_input_len, gs_output_len)]},
                                 }
        num_cs = len(self.data["collapse_summaries"]["input"])
        for i in range(num_cs):
            cs_parallelism = len(self.data[f"collapse_summaries"]["input"][i])
            cs_input_len = self.data["collapse_summaries"]["usage"][i]["prompt_tokens"]
            cs_output_len = self.data["collapse_summaries"]["usage"][i]["completion_tokens"]
            hint[f"collapse_summaries_{i}"] = {"parallelism":cs_parallelism,
                                                                "length": [(cs_input_len, cs_output_len)]}
        gfs_input_len = self.data["generate_final_summary"]["usage"]["prompt_tokens"]
        gfs_output_len = self.data["generate_final_summary"]["usage"]["completion_tokens"]
        hint["generate_final_summary"] = {"parallelism": 1,
                                                           "length": [(gfs_input_len, gfs_output_len)]}
        self.hint = hint

        # generate_summary
        _input_len = np.sum([input_len for input_len, output_len in
                                    hint["generate_summary"]["length"]]) * hint["generate_summary"]["parallelism"]
        # _output_len = np.max([output_len for input_len, output_len in
        #                              hint["generate_summary"]["length"]])
        _output_len = np.sum([output_len for input_len, output_len in
                                     hint["generate_summary"]["length"]])
        # collapse_summaries
        cs_cnt = 0
        while f"collapse_summaries_{cs_cnt}" in hint:
            stage_name = f"collapse_summaries_{cs_cnt}"
            _input_len += np.sum([input_len for input_len, output_len in
                                        hint[stage_name]["length"]]) * hint[stage_name]["parallelism"]
            # _output_len += np.max([output_len for input_len, output_len in
            #                              hint[stage_name]["length"]])
            _output_len += np.sum([output_len for input_len, output_len in
                                         hint[stage_name]["length"]])
            cs_cnt += 1

        # generate_final_summary
        _input_len += hint["generate_final_summary"]["length"][0][0] * hint["generate_final_summary"]["parallelism"]
        # _output_len += hint["generate_final_summary"]["length"][0][1] * hint["generate_final_summary"]["parallelism"]
        _output_len += hint["generate_final_summary"]["length"][0][1]

        jct = (
            _input_len * prefill_time +
            _output_len * decode_time +
            0.3 * 2
        )
        from CTaskBench.platform.llm.pdgraph import APPLICATION
        app_name = self.task_id.split("--")[0]
        predictor = APPLICATION[app_name].predictor
        v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 5 * 0.7
        print(f"app_name: {app_name} standard jct {v} oracle jct {jct}")
        self.slo = self.slo * v if self.slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct, self.slo)


    async def launch_openai_request(
        self, 
        messages, 
        request_id,
        output_tokens,
        stage_name = None,
    ) -> ChatCompletion:
        await asyncio.sleep(random.uniform(6, 300) / 1000)
        # print(f"launch_openai_request: {stage_name}")
        # last request of a stage
        new_extra_body = {}
        new_extra_body = {"request_id": request_id,}
        new_extra_body.update(self.extra_body)
        coinference_info_dict = {
            "stage_name": stage_name,
            "hint": self.hint,
            "slo": self.slo,
            "tpt": self.tpt,
        }

        new_extra_body.update({"coinference_info_dict": coinference_info_dict})
        self.request_cnt += 1

        self.time_recorder.start_request(self.task_id, request_id)
        response = await self.openai_client.chat.completions.create(
            model=self.config['model_name'],
            messages=messages,
            max_tokens=output_tokens,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            timeout=self.config['timeout'],
            extra_body=new_extra_body)
        self.time_recorder.end_request(self.task_id, request_id)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response

    # def length_function(self, documents: List[Document]) -> int:
    #     """Get number of tokens for input contents."""
    #     return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)


    # This will be the overall state of the main graph.
    # It will contain the input document contents, corresponding
    # summaries, and a final summary.
    class OverallState(TypedDict):
        # Notice here we use the operator.add
        # This is because we want combine all the summaries we generate
        # from individual nodes back into one list - this is essentially
        # the "reduce" part
        contents: List[str]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str


    # This will be the state of the node that we will "map" all
    # documents to in order to generate summaries
    class SummaryState(TypedDict):
        content: str


    # Here we generate a summary, given a document

    async def generate_summary(self, state: SummaryState):
        request_id = self.task_id + "--" + "gs" + str(self.gs_cnt)
        self.gs_cnt += 1
        start = time.time()
        # print(request_id)
        # response = await self.map_chain.ainvoke(state["content"])
        messages = [
            {"role": "user", "content": "Write a concise summary of the following:\\n\\n{}".format(state["content"]) },
        ]
        # print(messages)
        r = await self.launch_openai_request(messages,
                                             request_id,
                                             self.data['generate_summary']['usage']["completion_tokens"][self.gs_cnt-1],
                                             stage_name="generate_summary")
        response = self.data['generate_summary']['output'][self.gs_cnt-1]
        # print('ok')
        # print(response+'\n\n')
        end = time.time()
        # info['generate_summary']['input'].append(state["content"])
        # info['generate_summary']['output'].append(response)
        # info['generate_summary']['duration'].append(end - start )
        return {"summaries": [response]}


    # Here we define the logic to map out over the documents
    # We will use this an edge in the graph
    def map_summaries(self, state: OverallState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]


    def collect_summaries(self, state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }


    # Add node to collapse summaries
    async def collapse_summaries(self, state: OverallState):
        # doc_lists = split_list_of_docs(
        #     state["collapsed_summaries"], self.length_function, self.token_max
        # )
        # print(doc_lists)
        doc_lists = self.data['collapse_summaries']['input'][self.cs_cnt]
        start = time.time()
        results = []
        inputs = []
        outputs = []
        c_cnt = 0
        tasks = []
        for doc_list in doc_lists:
            # doc_list = [doc.page_content for doc in doc_list]
            request_id = self.task_id + "--" + "cs" + str(self.cs_cnt)+'_'+str(c_cnt)
            c_cnt += 1
            # print(request_id)
            # results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))
            reduce_template = """
            The following is a set of summaries:
            {docs}
            Take these and distill it into a final, consolidated summary
            of the main themes.
            """
            messages = [
                {'role': 'user', 'content': reduce_template.format(docs=doc_list)}
            ]
            # print(messages)
            task = self.launch_openai_request(
                messages,
                request_id,
                self.data['collapse_summaries']['usage'][self.cs_cnt]["completion_tokens"][c_cnt-1],
                stage_name="collapse_summaries"
            )
            # print('ok')
            tasks.append(task)
        await asyncio.gather(*tasks)
        results = self.data['collapse_summaries']['output'][self.cs_cnt]
        results = [Document(result) for result in results]
        self.cs_cnt += 1
        end = time.time()
        # print(results)
        # info['collapse_summaries']['input'].append(inputs)
        # info['collapse_summaries']['output'].append([res.page_content for res in results])
        # info['collapse_summaries']['duration'].append(end - start) 
        return {"collapsed_summaries": results}


    # This represents a conditional edge in the graph that determines
    # if we should collapse the summaries or not
    def should_collapse(self, 
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        request_id = self.task_id + "--" + 'sc' + '--' +str(self.sc_cnt)
        self.sc_cnt += 1
        self.time_recorder.start_request(self.task_id, request_id=request_id)
        # num_tokens = self.length_function(state["collapsed_summaries"])
        self.time_recorder.end_request(self.task_id, request_id=request_id)
        if len(self.data['collapse_summaries']['input']) > self.cs_cnt:
            return "collapse_summaries"
        else:
            return "generate_final_summary"


    # Here we will generate the final summary
    async def generate_final_summary(self, state: OverallState):
        # info['generate_final_summary']['input'].append([a.page_content for a in state["collapsed_summaries"]])
        start = time.time()
        request_id = self.task_id + "--" + "gfs"
        # response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        reduce_template = """
        The following is a set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary
        of the main themes.
        """
        messages = [
                {'role': 'user', 'content': reduce_template.format(docs = state["collapsed_summaries"])}
            ]
        await self.launch_openai_request(messages,
                                         request_id,
                                         self.data['generate_final_summary']["usage"]["completion_tokens"],
                                         stage_name="generate_final_summary")
        response = self.data['generate_final_summary']['output'][0]
        end = time.time()
        # info['generate_final_summary']['output'].append(response)
        # info['generate_final_summary']['duration'].append(end - start)
        return {"final_summary": response}


# Construct the graph
# Nodes:
    async def run(self):
        self.time_recorder.start_task(self.task_id)
        split_docs = [Document(doc) for doc in self.data['generate_summary']['input']]
        graph = StateGraph(self.OverallState)
        graph.add_node("generate_summary", self.generate_summary)  # same as before
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        # Edges:
        graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        graph.add_edge("generate_final_summary", END)

        app = graph.compile()

        async for step in app.astream(
            {"contents": [doc.page_content for doc in split_docs]},
            {"recursion_limit": 10},
        ):
            list(step.keys())
            # print(step)
            # info[step.keys()[0]] = step[step.keys()[0]][step.keys()[0]]
        self.time_recorder.finish_task(self.task_id)
        # self.time_recorder.save_to_file('/workspace/CTaskBench/Datasets/langchain/map_reduce/try.json')
    

# if __name__=="__main__":
#     with open('/workspace/langchain/data_m.json', 'r') as f:
#         datas = json.load(f)
#     print(len(datas))
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1000, chunk_overlap=0
# )
#     for data in tqdm(datas):
#         print(len(data))
#         docs = [Document(data)]
#         # print(length_function(docs))
#         split_docs = text_splitter.split_documents(docs)
#         print(f"Generated {len(split_docs)} documents.")
#         # for split_doc in split_docs:
#         #     print(length_function([split_doc]))
#         info = {}
#         info['duration'] = 0
#         info['generate_summary'] = {}
#         info['generate_summary']['input'] = []
#         info["generate_summary"]['output'] = []
#         info["generate_summary"]['duration'] = []
#         info['collect_summaries'] = []
#         info['collapse_summaries'] = {}
#         info['collapse_summaries']['input'] = []
#         info["collapse_summaries"]['output'] = []
#         info["collapse_summaries"]['duration'] = []
#         info['generate_final_summary'] = {} 
#         info['generate_final_summary']['input'] = []
#         info["generate_final_summary"]['output'] = []
#         info["generate_final_summary"]['duration'] = []

#         try:
#             start = time.time()
#             asyncio.run(run(split_docs))
#             end = time.time()
#             info['duration'] = end - start 
#             record.append(info)
#             with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/MAP_REDUCE.json','w') as f:
#                 json.dump(record, f, indent = 4)
#         except:
#             continue    
#     async def run(self):

#         return responses

