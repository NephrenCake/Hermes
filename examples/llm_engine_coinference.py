import argparse
from typing import List, Tuple
import time

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ({"prompt_token_ids": [42]*1000},
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--1--1"),
        ({"prompt_token_ids": [42]*1000},
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--2--1"),
        ({"prompt_token_ids": [42]*1000},
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--1--2"),
        ({"prompt_token_ids": [42]*1000},
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--1--3"),
        ({"prompt_token_ids": [42]*1000},
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--1--4"),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--2--2"),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--2--3"),
        ("It is only with the heart that one can see rightly",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=6, ignore_eos=True),
         "factool_code--2--4"),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""

    step_id = 0
    request_arrival_step = [0, 4, 10, 11, 12, 30, 31, 32]
    while test_prompts or engine.has_unfinished_requests():
        if step_id in request_arrival_step and test_prompts:
            prompt, sampling_params, request_id = test_prompts.pop(0)
            engine.add_request(request_id, prompt, sampling_params)
            
        t1 = time.time()
        engine.step()
        t2 = time.time()
        print(f"step {step_id} cost {(t2-t1)*1000:.2f} ms")
        step_id += 1

def test_swap_out(engine: LLMEngine, 
                  num_blocks: int):
    engine.test_swap_out(num_blocks)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.model = "/home/zgan/Models/opt-125m"
    # engine_args.model = "/home/zgan/Models/Llama-2-7b-chat-hf"
    # engine_args.tensor_parallel_size = 2
    engine_args.tokenizer = engine_args.model
    engine_args.gpu_memory_utilization = 0.2
    engine_args.max_num_seqs = 1
    engine_args.coinference_scheduler = True
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
