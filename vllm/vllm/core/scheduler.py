import enum
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _requeset_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _requeset_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            return

        self._requeset_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            self._requeset_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            return

        self._requeset_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            self._requeset_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class LogicalBudget:
    max_num_seqs: int
    max_num_blocks: int
    _num_curr_seqs: int = 0
    _num_used_blocks: int = 0
    budget_full: bool = False

    def can_schedule(self, num_new_seqs: int, num_new_blocks: int):
        assert num_new_seqs != 0
        assert num_new_blocks != 0
        return (self.num_curr_seqs + num_new_seqs <= self.max_num_seqs and
                self.num_used_blocks + num_new_blocks <= self.max_num_blocks)

    def add_num_seqs(self, num_new_seqs: int):
        self._num_curr_seqs += num_new_seqs

    def add_num_used_blocks(self, num_new_blocks: int):
        self._num_used_blocks += num_new_blocks

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_used_blocks(self):
        return self._num_used_blocks


@dataclass
class ScheduledSeqGroupOutputs:
    ignored_seq_groups: Set[SequenceGroup]
    prefill_queue: Set[SequenceGroup]
    decode_queue: Set[SequenceGroup]
    swapin_queue: Set[SequenceGroup]
    swapout_queue: Set[SequenceGroup]
    swapped_queue: Set[SequenceGroup]
    waiting_queue: Set[SequenceGroup]
    curr_loras: Set[LoRARequest]

@dataclass
class TokenBudget:
    token_budget: int
    _num_batched_tokens: int = 0

    def can_schedule(self, num_new_tokens: int):
        assert num_new_tokens != 0
        return self.num_batched_tokens + num_new_tokens <= self.token_budget

    def add_num_batched_tokens(self, num_batched_tokens: int):
        self._num_batched_tokens += num_batched_tokens

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerDecodeOutputs:
    seq_groups: List[ScheduledSequenceGroup]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_batched_tokens: int

    @classmethod
    def create_empty(cls) -> "SchedulerDecodeOutputs":
        return SchedulerDecodeOutputs(
            seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_batched_tokens=0,
        )


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: Iterable[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # blocks to save.
    blocks_to_save: List[Tuple[int, int]]
    # blocks to load.
    blocks_to_load: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int
    advised_lora: Set[LoRARequest]

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        # assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy
                and not self.blocks_to_save and not self.blocks_to_load)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    def swap_info_empty(self) -> bool:
        return (not self.blocks_to_swap_in and not self.blocks_to_swap_out
                and not self.blocks_to_copy
                and not self.blocks_to_save and not self.blocks_to_load)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


# @dataclass
# class SchedulerPrefillOutputs:
#     """The requests that are scheduled from a waiting queue.
#
#     Could contain a fresh prefill requests or preempted requests that need
#     to be recomputed from scratch.
#     """
#     # Selected sequences for prefill.
#     seq_groups: List[SequenceGroup]
#     # Ignored sequence groups.
#     ignored_seq_groups: List[SequenceGroup]
#     num_lookahead_slots: int
#
#     @classmethod
#     def create_empty(cls) -> "SchedulerPrefillOutputs":
#         return SchedulerPrefillOutputs(
#             seq_groups=[],
#             ignored_seq_groups=[],
#             num_lookahead_slots=0,
#         )


@dataclass
class SchedulerPrefillOutputs:
    seq_groups: List[ScheduledSequenceGroup]
    num_batched_tokens: int

    def is_empty(self) -> bool:
        return len(self.seq_groups) == 0


class TimeRecorder:
    def __init__(
            self,
            record_window_size: int,
            default_time_per_token: float = 0.1,
            ema_alpha: float = 0.9,
            token_type: str = "prefill"
    ) -> None:
        ''' ms '''
        self.time_per_token = default_time_per_token
        self.time_recorder = []
        self.num_tokens_recorder = []
        self.record_window_size = record_window_size
        self.ema_alpha = ema_alpha
        self.token_type = token_type

    def update(
            self,
            num_tokens: int,
            dur: float
    ):
        self.time_recorder.append(dur)
        self.num_tokens_recorder.append(num_tokens)
        if len(self.num_tokens_recorder) >= self.record_window_size:
            time_per_token = sum(self.time_recorder) / sum(self.num_tokens_recorder)
            self.time_recorder = []
            self.num_tokens_recorder = []
            self.time_per_token = self.time_per_token * self.ema_alpha + time_per_token * (1 - self.ema_alpha)

            print(f"[TimeRecorder] {self.token_type} time_per_token: {self.time_per_token:.4f} ms")
        # with open(os.path.join(os.path.dirname(__file__), f"{self.token_type}.json"), "w") as f:
        #     json.dump({
        #         "time_per_token": self.time_per_token,
        #     }, f)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"

        # Create the block space manager.
        from vllm.core.block_manager_v1 import BlockSpaceManagerV1
        self.block_manager = BlockSpaceManagerV1(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            num_disk_blocks=self.cache_config.num_disk_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()

        self.finished: Deque[SequenceGroup] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        self.non_preempt = scheduler_config.non_preempt
        self.lora_requests: Dict[str, LoRARequest] = {}

        self.prefill_recorder = TimeRecorder(record_window_size=100, default_time_per_token=0.33, token_type="prefill")
        self.decode_recorder = TimeRecorder(record_window_size=1000, default_time_per_token=3.16, token_type="decode")

        self.used_prefix_block = 0
        self.all_prefill_block = 0
        self.next_log_prefix = 0


    def get_request_info(self):
        request_info = {}
        for sg in self.waiting:
            request_info[sg.request_id] = {
                "input_len": len(sg.prompt_token_ids),
                "output_len": sum(seq.get_output_len() for seq in sg.seqs_dict.values()),
                "state": "waiting"
            }
        for sg in self.running:
            request_info[sg.request_id] = {
                "input_len": len(sg.prompt_token_ids),
                "output_len": sum(seq.get_output_len() for seq in sg.seqs_dict.values()),
                "state": "running"
            }
        for sg in self.swapped:
            request_info[sg.request_id] = {
                "input_len": len(sg.prompt_token_ids),
                "output_len": sum(seq.get_output_len() for seq in sg.seqs_dict.values()),
                "state": "swapped"
            }
        for sg in self.finished:
            request_info[sg.request_id] = {
                "input_len": len(sg.prompt_token_ids),
                "output_len": sum(seq.get_output_len() for seq in sg.seqs_dict.values()),
                "state": "finished"
            }
        self.finished.clear()
        return request_info

    def update_request_priority(self, priorities: Dict[str, float]):
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id not in priorities:
                    # logger.info(f"{seq_group.request_id} not in {priorities.keys()}")
                    continue
                seq_group.priority = priorities[seq_group.request_id]
                # logger.info(f"Update request {seq_group.request_id} priority to {seq_group.priority}")

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            self.finished.extend(aborted_groups)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            running_queue: The queue that contains running requests (i.e.,
                decodes). The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            policy: The sorting policy to sort running_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            A tuple of remaining running queue (should be always 0) after
            scheduling and SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            policy: The sorting policy to sort swapped_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining swapped_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        infeasible_seq_groups: List[SequenceGroup] = []

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            alloc_status = self.block_manager.can_swap_in(seq_group)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras, enable_chunking=False)

        fcfs_policy = PolicyFactory.get_policy(policy_name="global")
        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            remaining_running, running_scheduled = self._schedule_running(
                self.running,
                budget,
                curr_loras,
                fcfs_policy,
                enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.swapped, budget, curr_loras, fcfs_policy)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            blocks_to_save=None,
            blocks_to_load=None,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self):
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # Decoding should be always scheduled first by fcfs.
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            fcfs_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, fcfs_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_save=[],
            blocks_to_load=[],
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        # if self.scheduler_config.chunked_prefill_enabled:
        #     return self._schedule_chunked_prefill()
        # else:
        #     return self._schedule_default()
        return self._schedule_hermes()

    def _schedule_hermes(self):
        # split coinference to running_queue and waiting_queue
        schedule = self.schedule_seq_groups()

        # prepare for eviction and prefetch
        swap_mapping = self.block_manager.cache_swap_mapping.clear()

        # swap out all seq_group in waiting_queue
        blocks_to_swap_out = self.schedule_swapout(schedule.swapout_queue)

        # prefill if there is seq_group need to prefill in running_queue
        schedule_prefill_outputs = self.schedule_prefill(schedule.prefill_queue)

        # decode and swap in
        schedule_decode_outputs = SchedulerDecodeOutputs.create_empty()
        if schedule_prefill_outputs.is_empty():
            schedule_decode_outputs = self.schedule_decode(schedule.decode_queue,
                                                           schedule.swapin_queue)

        # lora
        advised_lora = schedule.curr_loras  # default vllm lora policy
        if self.lora_enabled:  # took over by top scheduler
            lora_preference = []  # todo 由 top scheduler 给出 [lora id]
            for name in lora_preference:
                lora = self.lora_requests[name]
                if len(advised_lora) == self.lora_config.max_cpu_loras:
                    break
                if lora in advised_lora:
                    continue
                advised_lora.add(lora)
                logger.info(
                    f"[LoRA Debug] > "
                    f"advised_lora: {[i.lora_int_id for i in advised_lora]}, "
                )
        # kv
        if self.cache_config.enable_prefix_caching:  # took over by top scheduler
            kv_preference = []  # todo 由 top scheduler 给出 [app id]
            self.block_manager.gpu_allocator.evictor.set_priority(kv_preference)
            self.block_manager.cpu_allocator.evictor.set_priority(kv_preference)
            self.block_manager.disk_allocator.evictor.set_priority(kv_preference)
            # gpu
            gpu_block = self.block_manager.gpu_allocator.get_num_free_blocks()
            used_gpu_blocks = 0
            for i in kv_preference:
                if used_gpu_blocks + len(self.block_manager.gpu_allocator.may_prefetch(i)) > gpu_block:
                    break
                self.block_manager.gpu_allocator.prefetch_coinf(i)
                used_gpu_blocks += self.block_manager.gpu_allocator.evictor.num_blocks_of(i)
            # cpu
            cpu_blocks = self.block_manager.num_total_cpu_blocks * 0.30
            used_cpu_blocks = 0
            for i in kv_preference:
                if used_cpu_blocks + len(self.block_manager.cpu_allocator.may_prefetch(i)) > cpu_blocks:
                    break
                self.block_manager.cpu_allocator.prefetch_coinf(i)  # try prefetch from disk to cpu
                used_cpu_blocks += self.block_manager.cpu_allocator.evictor.num_blocks_of(i)
        self.block_manager.clean_to_watermark(0.5)

        return SchedulerOutputs(
            scheduled_seq_groups=(schedule_prefill_outputs.seq_groups +
                                  schedule_decode_outputs.seq_groups),
            num_prefill_groups=len(schedule_prefill_outputs.seq_groups),
            num_batched_tokens=(schedule_prefill_outputs.num_batched_tokens +
                                schedule_decode_outputs.num_batched_tokens),
            blocks_to_swap_in=(schedule_decode_outputs.blocks_to_swap_in + swap_mapping.cpu2gpu),
            blocks_to_swap_out=(blocks_to_swap_out + swap_mapping.gpu2cpu),
            blocks_to_copy=schedule_decode_outputs.blocks_to_copy,
            blocks_to_save=swap_mapping.cpu2disk,
            blocks_to_load=swap_mapping.disk2cpu,
            ignored_seq_groups=list(schedule.ignored_seq_groups),
            num_lookahead_slots=0,
            running_queue_size=0,
            preempted=len(schedule.swapout_queue),
            advised_lora=advised_lora,
        )

    def try_schedule(self, seq_group, logicalbudget, schedule: ScheduledSeqGroupOutputs):
        if (
                seq_group in schedule.ignored_seq_groups or
                seq_group in schedule.prefill_queue or
                seq_group in schedule.decode_queue or
                seq_group in schedule.swapin_queue or
                seq_group in schedule.swapout_queue or
                seq_group in schedule.swapped_queue or
                seq_group in schedule.waiting_queue
        ):
            return

        has_lora_slot = True
        if self.lora_enabled:
            assert self.lora_config is not None
            new_lora_num = len(schedule.curr_loras) + (seq_group.lora_request not in schedule.curr_loras)
            if new_lora_num > self.lora_config.max_loras:
                has_lora_slot = False

        if logicalbudget.budget_full or not has_lora_slot:
            if seq_group.is_running():
                schedule.swapout_queue.add(seq_group)
            elif seq_group.is_swapped():
                schedule.swapped_queue.add(seq_group)
            else:
                schedule.waiting_queue.add(seq_group)
            return

        if seq_group.is_prefill():
            # prompt too long
            prompt_limit = self.scheduler_config.max_model_len
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = waiting_seqs[0].get_num_new_tokens()
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                schedule.ignored_seq_groups.add(seq_group)
                return

            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_blocks = len(seq_group.get_seqs(status=SequenceStatus.WAITING)[0].logical_token_blocks)
            num_watermark_blocks = self.block_manager.watermark_blocks

            # never can allocate
            if self.block_manager.num_total_gpu_blocks - num_new_blocks < self.block_manager.watermark_blocks:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                schedule.ignored_seq_groups.add(seq_group)
                return
        else:
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_blocks = len(self.block_manager._get_physical_blocks(seq_group))
            num_new_blocks += seq_group.num_seqs(status=SequenceStatus.RUNNING)
            num_new_blocks += seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_watermark_blocks = 0

        if logicalbudget.can_schedule(num_new_seqs, num_new_blocks + num_watermark_blocks):
            schedule.curr_loras.add(seq_group.lora_request)
            logicalbudget.add_num_seqs(num_new_seqs)
            logicalbudget.add_num_used_blocks(num_new_blocks)
            if seq_group.is_prefill():
                schedule.prefill_queue.add(seq_group)
            elif seq_group.is_swapped():
                schedule.swapin_queue.add(seq_group)
            else:
                schedule.decode_queue.add(seq_group)
        else:
            logicalbudget.budget_full = True
            if seq_group.is_running():
                schedule.swapout_queue.add(seq_group)
            elif seq_group.is_swapped():
                schedule.swapped_queue.add(seq_group)
            else:
                schedule.waiting_queue.add(seq_group)

    def schedule_seq_groups(self) -> ScheduledSeqGroupOutputs:
        logicalbudget = LogicalBudget(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_blocks=self.cache_config.num_gpu_blocks,
        )
        schedule = ScheduledSeqGroupOutputs(ignored_seq_groups=set(),
                                            prefill_queue=set(),
                                            decode_queue=set(),
                                            swapin_queue=set(),
                                            swapout_queue=set(),
                                            swapped_queue=set(),
                                            waiting_queue=set(),
                                            curr_loras=set())

        if self.non_preempt:  # follow default preempt logic of vllm
            for seq_group in self.running + self.swapped:
                if seq_group.is_finished():
                    continue
                self.try_schedule(seq_group, logicalbudget, schedule)

        seq_groups = self.running + self.swapped + self.waiting
        seq_groups = PolicyFactory.get_policy(
            policy_name="global"
        ).sort_by_priority(time.time(), seq_groups)  # track and sort all requests
        for seq_group in seq_groups:
            self.try_schedule(seq_group, logicalbudget, schedule)

        self.waiting = deque(schedule.ignored_seq_groups | schedule.waiting_queue)
        self.running = deque(schedule.prefill_queue | schedule.decode_queue | schedule.swapin_queue)
        self.swapped = deque(schedule.swapout_queue | schedule.swapped_queue)
        # print(f"[Schedule Debug] running requests: {[sg.request_id for sg in self.running]}")

        return schedule

    def schedule_swapout(self, swapout_queue: Set[SequenceGroup]) -> List[Tuple[int, int]]:
        blocks_to_swap_out: List[Tuple[int, int]] = []

        for seq_group in swapout_queue:
            preemption_mode = {
                "swap": PreemptionMode.SWAP,
                "recompute": PreemptionMode.RECOMPUTE,
            }["swap"]
            self._preempt(seq_group,
                          blocks_to_swap_out,
                          preemption_mode)

        return blocks_to_swap_out

    def schedule_prefill(self, prefill_queue: Set[SequenceGroup]) -> SchedulerPrefillOutputs:
        tokenbudget = TokenBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
        )
        seq_groups: List[ScheduledSequenceGroup] = []
        for seq_group in prefill_queue:
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            num_new_tokens = waiting_seqs[0].get_num_new_tokens()
            if not tokenbudget.can_schedule(num_new_tokens=num_new_tokens):
                break
            self._allocate_and_set_running(seq_group)
            seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group,
                                                     token_chunk_size=num_new_tokens))
            tokenbudget.add_num_batched_tokens(num_new_tokens)
            break  # force to prefill only one sequence group to prevent illegal memory access error

        return SchedulerPrefillOutputs(seq_groups=seq_groups,
                                       num_batched_tokens=tokenbudget.num_batched_tokens)

    def schedule_decode(
            self,
            decode_queue: Set[SequenceGroup],
            swapin_queue: Set[SequenceGroup],
    ) -> SchedulerDecodeOutputs:
        tokenbudget = TokenBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
        )
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        for seq_group in decode_queue:
            num_new_tokens = seq_group.num_seqs(status=SequenceStatus.RUNNING)
            assert tokenbudget.can_schedule(num_new_tokens), "token budget full"
            tokenbudget.add_num_batched_tokens(num_new_tokens)
            self._append_slots(seq_group, blocks_to_copy)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group, token_chunk_size=1))

        for seq_group in swapin_queue:
            num_new_tokens = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            assert tokenbudget.can_schedule(num_new_tokens), "token budget full"
            tokenbudget.add_num_batched_tokens(num_new_tokens)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group, token_chunk_size=1))

        return SchedulerDecodeOutputs(seq_groups=seq_groups,
                                      blocks_to_swap_in=blocks_to_swap_in,
                                      blocks_to_copy=blocks_to_copy,
                                      num_batched_tokens=tokenbudget.num_batched_tokens)

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens() <
                        seqs[0].data.get_len()):
                    do_sample = False

                self.used_prefix_block += len(common_computed_block_nums)
                self.all_prefill_block += len(block_tables[seqs[0].seq_id])
                # if len(common_computed_block_nums) > 0:
                #     logger.info(
                #         f"[KVC Debug] > {seq_group.request_id} prefill with prefix: "
                #         f"{len(common_computed_block_nums)}/{len(block_tables[seqs[0].seq_id])}"
                #     )

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        if self.cache_config.enable_prefix_caching and self.all_prefill_block > self.next_log_prefix * 1000:
            self.next_log_prefix = self.all_prefill_block // 1000 + 1
            if scheduler_outputs.num_prefill_groups != 0:
                logger.info(f"[KVC Debug] > prefix utilization: "
                            f"{self.used_prefix_block} / {self.all_prefill_block} = "
                            f"{self.used_prefix_block / self.all_prefill_block:.2f}")
                g_hit, g_miss, g_total = self.block_manager.gpu_allocator.export_statistics()
                c_hit, c_miss, c_total = self.block_manager.cpu_allocator.export_statistics()
                d_hit, d_miss, d_total = self.block_manager.disk_allocator.export_statistics()
                logger.info(f"[KVC Debug] > "
                            f"GPU: {g_hit} / {g_total} = {(g_hit / g_total) if g_total else 0.:.2f}, "
                            f"CPU: {c_hit} / {g_total} = {(c_hit / g_total) if g_total else 0.:.2f}, "
                            f"Disk: {d_hit} / {g_total} = {(d_hit / g_total) if g_total else 0.:.2f}")

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.finished.extend([seq_group for seq_group in self.running
                              if seq_group.is_finished()])
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns 0 if the new token cannot be computed due to token budget.
        """
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens

    def record_step_time(self, num_tokens: int, dur: float, is_prefill: bool):
        if is_prefill:
            self.prefill_recorder.update(num_tokens, dur)
        else:
            self.decode_recorder.update(1, dur)
