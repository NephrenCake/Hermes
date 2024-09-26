from typing import Dict, List, Tuple
from collections import deque, OrderedDict

from vllm.block import PhysicalTokenBlock, DiskBlock


class DiskBlockManager:
    def __init__(
            self,
            num_blocks: int,
    ) -> None:
        self.total_num_blocks = num_blocks
        self.free_blocks = deque([i for i in range(num_blocks)])
        # {block_hash: (block_id, num_hashed_tokens)}
        self.block_table: OrderedDict[int, DiskBlock] = OrderedDict()
        self.default_free_num = int(self.total_num_blocks * 0.1)

    def save_to_disk(
            self,
            block: PhysicalTokenBlock,
    ) -> int:
        if len(self.free_blocks) <= 0:
            self.free_some_blocks()
        assert block.block_hash not in self.block_table
        block_id = self.free_blocks.popleft()
        self.block_table[block.block_hash] = DiskBlock(block_id,
                                                       block.block_hash,
                                                       block.num_hashed_tokens,
                                                       block.computed)
        return block_id

    def load_from_disk(
            self,
            block_hash: int,
    ) -> DiskBlock:
        assert block_hash in self.block_table
        disk_block = self.block_table.pop(block_hash)
        self.free_blocks.append(disk_block.block_number)
        return disk_block

    def contains_block(self, block_hash: int):
        return block_hash in self.block_table

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def free_some_blocks(self, num_blocks_need_free: int = None) -> None:
        if num_blocks_need_free == None:
            num_blocks_need_free = self.default_free_num

        assert num_blocks_need_free <= len(self.block_table)
        blocks_to_evict = list(self.block_table.keys())[:num_blocks_need_free]
        for block_id in blocks_to_evict:
            disk_block = self.block_table.pop(block_id)
            self.free_blocks.append(disk_block.block_number)
