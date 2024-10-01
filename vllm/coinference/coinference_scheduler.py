import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import CoInferencePolicy, PolicyFactory
from vllm.core.scheduler import (SchedulerOutputs, ScheduledSequenceGroup,
                                 PreemptionMode)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.coinference.coinference import CoInference, FinishType
from vllm.coinference.coinference_creator import create_coinference
from vllm.utils import Counter

logger = init_logger(__name__)


@dataclass
class SplitSeqGroupOutputs:
    ignored_seq_groups: List[SequenceGroup]
    prefill_queue: List[SequenceGroup]
    decode_queue: List[SequenceGroup]
    swapin_queue: List[SequenceGroup]
    swapout_queue: List[SequenceGroup]
    swapped_queue: List[SequenceGroup]
    waiting_queue: List[SequenceGroup]
    curr_loras: Set[LoRARequest]


@dataclass
class SchedulerPrefillOutputs:
    seq_groups: List[ScheduledSequenceGroup]
    num_batched_tokens: int

    def is_empty(self) -> bool:
        return len(self.seq_groups) == 0


@dataclass
class SchedulerSwapOutOutputs:
    blocks_to_swap_out: List[Tuple[int, int]]


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
class LogicalBudget:
    max_num_seqs: int
    max_num_blocks: int
    _num_curr_seqs: int = 0
    _num_used_blocks: int = 0

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


class TimeRecorder:
    def __init__(
            self,
            record_window_size: int,
            default_time_per_token: float = 0.1,
            ema_alpha: float = 0.7,
    ) -> None:
        ''' ms '''
        self.time_per_token = default_time_per_token
        self.time_recorder = []
        self.num_tokens_recorder = []
        self.record_window_size = record_window_size
        self.ema_alpha = ema_alpha

    def update(
            self,
            num_tokens: int,
            time: float
    ):
        self.time_recorder.append(time)
        self.num_tokens_recorder.append(num_tokens)
        if len(self.num_tokens_recorder) >= self.record_window_size:
            time_per_token = sum(self.time_recorder) / sum(self.num_tokens_recorder)
            self.time_recorder = []
            self.num_tokens_recorder = []
            self.time_per_token = self.time_per_token * self.ema_alpha + time_per_token * (1 - self.ema_alpha)


class CoInferenceScheduler:
    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config

        version = "v1"
        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            num_disk_blocks=self.cache_config.num_disk_blocks)

        self.coinferences_dict: Dict[str, CoInference] = {}
        self.coinferences_queue: List[CoInference] = []

        self.num_running_seq_groups = 0
        self.num_waiting_seq_groups = 0
        self.num_swapped_seq_groups = 0

        self.proactive_reservation = scheduler_config.proactive_reservation
        self.scheduling_policy = scheduler_config.scheduling_policy

        self.prefill_recorder = TimeRecorder(record_window_size=10, default_time_per_token=0.1)
        self.decode_recorder = TimeRecorder(record_window_size=50, default_time_per_token=3)

        self.used_prefix_block = 0
        self.all_prefill_block = 0
        self.next_log_prefix = 0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    # schedule main function
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
                if self.cache_config.enable_prefix_caching and self.all_prefill_block > self.next_log_prefix * 1000:
                    self.next_log_prefix = self.all_prefill_block // 1000 + 1
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

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_default()

    def get_lora_preference(self, schedule_prefill_outputs, schedule_decode_outputs, split_outputs, mode="Hermes"):
        # scheduled_loras contains prefill/decode loras, curr_loras contains prefill+decode loras
        scheduled_loras = {i.seq_group.lora_request for i in
                           schedule_prefill_outputs.seq_groups + schedule_decode_outputs.seq_groups}
        # logger.info(f"scheduled_loras: {scheduled_loras}, curr_loras: {split_outputs.curr_loras}")
        lora_preference = list(scheduled_loras) + list(split_outputs.curr_loras - scheduled_loras)
        advised_lora = set(lora_preference)
        for coinf in self.coinferences_queue:
            if len(advised_lora) == self.lora_config.max_cpu_loras:
                break

            if mode == "Hermes":
                is_in_queue = coinf.finish_status == FinishType.UnFinished
                if coinf.finish_status == FinishType.StageFinished:
                    detected_gap = time.time() - coinf.finish_time
                    prob1 = coinf.predictor.get_next_stage_gap_cdf(
                        coinf.stages[-1].stage_name, detected_gap + 0.3)  # consider the transferring time
                    prob2 = coinf.predictor.get_next_stage_gap_cdf(
                        coinf.stages[-1].stage_name, detected_gap - 0.3)  # extra slack time
                    is_in_queue |= 0.1 <= prob1 and prob2 <= 0.9
                    # logger.info(f"Next Stage GAP CDF: {prob2:.2f}~{prob1:.2f}, detected_gap: {detected_gap * 1000}")
            elif mode == "EPWQ":  # Evict/Prefetch on Waiting Queue
                is_in_queue = coinf.finish_status == FinishType.UnFinished
            else:
                raise

            if not is_in_queue:
                continue

            lr = coinf.stages[-1].parallel_requests[0].lora_request
            if lr in advised_lora:
                continue
            advised_lora.add(lr)
            lora_preference.append(lr)
        # logger.info(f"lora_preference: {[lr.lora_int_id for lr in lora_preference]}, "
        #             f"advised_lora: {[lr.lora_int_id for lr in advised_lora]}")

        return lora_preference, advised_lora

    def _schedule_default(self) -> SchedulerOutputs:
        now = time.time()

        t_sort = time.time()
        policy: CoInferencePolicy = PolicyFactory.get_policy(policy_name=self.scheduling_policy)
        if self.scheduling_policy in ["Hermes", "Idealized-SRJF", "Mean-SRJF"]:
            for coinf in self.coinferences_dict.values():
                coinf.estimate_remaining_time(self.prefill_recorder.time_per_token,
                                              self.decode_recorder.time_per_token,
                                              use_mean=self.scheduling_policy == "Mean-SRJF")
        self.coinferences_queue = policy.sort_by_priority(now, self.coinferences_queue)
        t_sort = (time.time() - t_sort) * 1000

        # split coinference to running_queue and waiting_queue
        split_outputs = self.split_seq_groups(self.coinferences_queue)

        self.block_manager.cache_swap_mapping.clear()

        # swap out all seq_group in waiting_queue
        t_preempt = time.time()
        schedule_swapout_outputs = self.schedule_swapout(split_outputs.swapout_queue)
        t_preempt = (time.time() - t_preempt) * 1000

        # prefill if there is seq_group need to prefill in running_queue
        t_prefill = time.time()
        schedule_prefill_outputs = self.schedule_prefill(split_outputs.prefill_queue)
        t_prefill = (time.time() - t_prefill) * 1000
        # if t_prefill > 10:
        #     logger.info(f"t_prefill: {t_prefill:.2f}")

        # decode and swap in
        t_decode = time.time()
        schedule_decode_outputs = SchedulerDecodeOutputs.create_empty()
        if schedule_prefill_outputs.is_empty():
            schedule_decode_outputs = self.schedule_decode(split_outputs.decode_queue,
                                                           split_outputs.swapin_queue)
        t_decode = (time.time() - t_decode) * 1000

        t_lora = time.time()
        # get lora preference
        advised_lora = set()
        if self.lora_enabled and self.scheduler_config.lora_policy in ["Hermes", "EPWQ"]:
            lora_preference, advised_lora = self.get_lora_preference(schedule_prefill_outputs,
                                                                     schedule_decode_outputs,
                                                                     split_outputs,
                                                                     mode=self.scheduler_config.lora_policy)
            # lora_preference_, advised_lora_ = self.get_lora_preference(schedule_prefill_outputs,
            #                                                            schedule_decode_outputs,
            #                                                            split_outputs,
            #                                                            mode="EPWQ")
            # logger.info(
            #     f"[LoRA Debug] > "
            #     f"Hermes: {[i.lora_int_id for i in lora_preference[:10]]}, "
            #     # f"advised_lora: {[i.lora_int_id for i in advised_lora]}, "
            #     f"WaitingQueue: {[i.lora_int_id for i in lora_preference_[:10]]}, "
            # )
        t_lora = (time.time() - t_lora) * 1000

        # logger.info(f"t_sort: {t_sort:.2f}, t_preempt: {t_preempt:.2f}, t_prefill: {t_prefill:.2f}, "
        #             f"t_decode: {t_decode:.2f}, t_lora: {t_lora:.2f}, total = {time.time() - now:.2f}")

        # update num of seq_groups in different status
        self.num_waiting_seq_groups = (len(split_outputs.waiting_queue) + len(split_outputs.prefill_queue)
                                       - len(schedule_prefill_outputs.seq_groups))
        self.num_running_seq_groups = len(split_outputs.decode_queue) + len(schedule_prefill_outputs.seq_groups)
        if schedule_prefill_outputs.is_empty():
            self.num_running_seq_groups += len(split_outputs.swapin_queue)
        self.num_swapped_seq_groups = len(split_outputs.swapped_queue) + len(split_outputs.swapout_queue)

        # logger.info(f"\tcoinferences: {self.coinferences_dict}")
        # logger.info(f"\tprefill_queue: {split_outputs.prefill_queue}")
        # logger.info(f"\tdecode_queue: {split_outputs.decode_queue}")
        # logger.info(f"\tswapin_queue: {split_outputs.swapin_queue}")
        # logger.info(f"\tswapout_queue: {split_outputs.swapout_queue}")
        # logger.info(f"\tswapped_queue: {split_outputs.swapped_queue}")
        # logger.info(f"\twaiting_queue: {split_outputs.waiting_queue}")
        self.block_manager.clean_to_watermark()
        swap_mapping = self.block_manager.get_cache_swap_mapping()
        schedule_outputs = SchedulerOutputs(
            scheduled_seq_groups=(schedule_prefill_outputs.seq_groups +
                                  schedule_decode_outputs.seq_groups),
            num_prefill_groups=len(schedule_prefill_outputs.seq_groups),
            num_batched_tokens=(schedule_prefill_outputs.num_batched_tokens +
                                schedule_decode_outputs.num_batched_tokens),
            blocks_to_swap_in=(schedule_decode_outputs.blocks_to_swap_in + swap_mapping.cpu2gpu),
            blocks_to_swap_out=(schedule_swapout_outputs.blocks_to_swap_out + swap_mapping.gpu2cpu),
            blocks_to_copy=schedule_decode_outputs.blocks_to_copy,
            blocks_to_save=swap_mapping.cpu2disk,
            blocks_to_load=swap_mapping.disk2cpu,
            ignored_seq_groups=split_outputs.ignored_seq_groups,
            num_lookahead_slots=0,
            running_queue_size=0,
            preempted=len(split_outputs.swapout_queue),
            advised_lora=advised_lora,
        )
        return schedule_outputs

    def split_seq_groups(self, coinferences_queue: List[CoInference]) -> SplitSeqGroupOutputs:
        logicalbudget = LogicalBudget(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_blocks=self.cache_config.num_gpu_blocks,
        )

        ignored_seq_groups: List[SequenceGroup] = []
        prefill_queue: List[SequenceGroup] = []
        decode_queue: List[SequenceGroup] = []
        swapin_queue: List[SequenceGroup] = []
        swapout_queue: List[SequenceGroup] = []
        swapped_queue: List[SequenceGroup] = []
        waiting_queue: List[SequenceGroup] = []

        budget_full = False
        curr_loras = set()
        for coinf in coinferences_queue:
            if coinf.current_stage_id == len(coinf.stages):
                continue
            stage = coinf.current_stage
            seq_groups = [seq_group for seq_group in stage.parallel_requests if not seq_group.is_finished()]
            for seq_group in seq_groups:
                has_lora_slot = True
                if self.lora_enabled:
                    assert self.lora_config is not None
                    new_lora_num = len(curr_loras) + (seq_group.lora_request not in curr_loras)
                    if new_lora_num > self.lora_config.max_loras:
                        has_lora_slot = False

                if budget_full or not has_lora_slot:
                    if seq_group.is_running():
                        swapout_queue.append(seq_group)
                    elif seq_group.is_swapped():
                        swapped_queue.append(seq_group)
                    else:
                        waiting_queue.append(seq_group)
                    continue

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
                        ignored_seq_groups.append(seq_group)
                        continue

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
                        ignored_seq_groups.append(seq_group)
                        continue
                else:
                    num_new_seqs = seq_group.get_max_num_running_seqs()
                    num_new_blocks = len(self.block_manager._get_physical_blocks(seq_group))
                    num_new_blocks += seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    num_new_blocks += seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                    num_watermark_blocks = 0

                if logicalbudget.can_schedule(num_new_seqs, num_new_blocks + num_watermark_blocks):
                    curr_loras.add(seq_group.lora_request)
                    logicalbudget.add_num_seqs(num_new_seqs)
                    logicalbudget.add_num_used_blocks(num_new_blocks)
                    if seq_group.is_prefill():
                        prefill_queue.append(seq_group)
                    elif seq_group.is_swapped():
                        swapin_queue.append(seq_group)
                    else:
                        decode_queue.append(seq_group)
                else:
                    budget_full = True
                    if seq_group.is_running():
                        swapout_queue.append(seq_group)
                    elif seq_group.is_swapped():
                        swapped_queue.append(seq_group)
                    else:
                        waiting_queue.append(seq_group)

            # # proactive_reservation
            # if self.proactive_reservation and not budget_full:
            #     # logger.info(f"{coinf.coinf_id} next stage will arrival after {(stage.predicted_stage_arrival_time - now)*1000:.2f} ms")
            #     if not stage.all_arrival and stage.should_reserve(now):
            #         for reserve_seq_group in stage.reserve_seq_groups:
            #             num_new_seqs = reserve_seq_group.num_new_seqs
            #             # NOTE (zgan): there will reserve one block less as rounding down.
            #             # But I think it doesn't matter.
            #             num_new_blocks = reserve_seq_group.num_prompt_tokens//self.block_manager.block_size
            #             num_watermark_blocks = self.block_manager.watermark_blocks
            #             if logicalbudget.can_schedule(num_new_seqs, num_new_blocks+num_watermark_blocks):
            #                 logicalbudget.add_num_seqs(num_new_seqs)
            #                 logicalbudget.add_num_used_blocks(num_new_blocks)
            #             else:
            #                 budget_full = True
            #                 break

        return SplitSeqGroupOutputs(ignored_seq_groups=ignored_seq_groups,
                                    prefill_queue=prefill_queue,
                                    decode_queue=decode_queue,
                                    swapin_queue=swapin_queue,
                                    swapout_queue=swapout_queue,
                                    swapped_queue=swapped_queue,
                                    waiting_queue=waiting_queue,
                                    curr_loras=curr_loras)

    def schedule_swapout(self, swapout_queue: List[SequenceGroup]) -> SchedulerSwapOutOutputs:
        blocks_to_swap_out: List[Tuple[int, int]] = []

        for seq_group in swapout_queue:
            preemption_mode = {
                "swap": PreemptionMode.SWAP,
                "recompute": PreemptionMode.RECOMPUTE,
            }[self.cache_config.preemption_mode]
            self._preempt(seq_group,
                          blocks_to_swap_out,
                          preemption_mode)

        return SchedulerSwapOutOutputs(blocks_to_swap_out=blocks_to_swap_out)

    def schedule_prefill(self, prefill_queue: List[SequenceGroup]) -> SchedulerPrefillOutputs:
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
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            tokenbudget.add_num_batched_tokens(num_new_tokens)

        return SchedulerPrefillOutputs(seq_groups=seq_groups,
                                       num_batched_tokens=tokenbudget.num_batched_tokens)

    def schedule_decode(
            self,
            decode_queue: List[SequenceGroup],
            swapin_queue: List[SequenceGroup],
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

    # other function
    def add_seq_group(
            self,
            seq_group: SequenceGroup,
            coinference_info_dict: Optional[Dict],
    ) -> int:
        self.free_finished_seq_groups()

        coinference_info_dict = coinference_info_dict if coinference_info_dict is not None else {}
        hint = coinference_info_dict.get("hint", None) if self.scheduling_policy == "Idealized-SRJF" else None
        stage_name = coinference_info_dict.get("stage_name", None)

        if self.scheduling_policy == "Request-Level-FIFO":
            seq_group.app_name = None
            seq_group.coinf_id = seq_group.request_id

        # logger.info(f"Add seq_group {seq_group.request_id}, coinference_info_dict: {coinference_info_dict}")
        if seq_group.coinf_id not in self.coinferences_dict:
            new_coinf = create_coinference(seq_group.app_name,
                                           seq_group.coinf_id,
                                           seq_group.metrics.arrival_time,
                                           hint)
            self.coinferences_dict[seq_group.coinf_id] = new_coinf
            self.coinferences_queue.append(new_coinf)

            # logger.info(f"Add coinference {seq_group.coinf_id} to scheduler")

        self.coinferences_dict[seq_group.coinf_id].add_req(seq_group, stage_name, self.scheduling_policy == "Mean-SRJF")

        for seq in seq_group.seqs_dict.values():
            seq.coinf_id = seq_group.coinf_id

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]):
        # TODO: implement
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        aborted_groups: List[SequenceGroup] = []
        for request_id in request_ids:
            if '--' in request_id:
                splited_id = request_id.split('--')
                coinf_id = f"{splited_id[0]}--{splited_id[1]}"
            else:
                coinf_id = request_id
            if coinf_id not in self.coinferences_dict:
                continue
            coinf = self.coinferences_dict.pop(coinf_id)
            self.coinferences_queue.remove(coinf)
            aborted_groups += coinf.current_stage.parallel_requests
        for aborted_group in aborted_groups:
            for seq in aborted_group.get_seqs():
                if seq.is_finished():
                    continue
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.free_seq(seq)

    def get_num_unfinished_seq_groups(self) -> int:
        return sum([coinf.get_num_unfinished_seq_groups() for coinf in self.coinferences_dict.values()])

    def has_unfinished_seqs(self) -> bool:
        return self.get_num_unfinished_seq_groups() != 0

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        # NOTE: this function actually free finished coinferences
        now = time.time()
        finished_coinf = []
        for coinf_id, coinf in self.coinferences_dict.items():
            if coinf.is_finished(now):
                finished_coinf.append(coinf_id)

        for coinf_id in finished_coinf:
            coinf = self.coinferences_dict.pop(coinf_id)
            if self.scheduling_policy in ["Hermes", "Mean-SRJF"]:
                coinf.update_online_profiling()
            self.coinferences_queue.remove(coinf)

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
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, 0)
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

    def record_step_time(self, num_tokens: int, time: float, is_prefill: bool):
        if is_prefill:
            self.prefill_recorder.update(num_tokens, time)
        else:
            self.decode_recorder.update(num_tokens, time)
