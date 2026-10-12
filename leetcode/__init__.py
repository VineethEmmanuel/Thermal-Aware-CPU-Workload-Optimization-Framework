"""
LeetCode-Style Algorithmic Problem Solvers & DAA Pattern Implementations.

Maps core algorithms of the Thermal-Aware CPU Workload Optimization Framework
to standard competitive programming and interview patterns:
- 0/1 & Fractional Knapsack (LeetCode 416, 474, 494)
- Matrix Chain Multiplication / Interval DP (LeetCode 312, 1000)
- Advanced Priority Queues & Heaps (LeetCode 23, 218, 295)
"""

from leetcode.knapsack_solver import (
    can_partition_subset_sum,
    fractional_knapsack,
    knapsack_01_memoized,
    knapsack_01_space_optimized,
    knapsack_01_tabulation,
)
from leetcode.matrix_chain import (
    burst_balloons,
    matrix_chain_multiplication,
)
from leetcode.priority_queues import (
    IndexedMinHeap,
    merge_k_sorted_lists,
)

__all__ = [
    "knapsack_01_tabulation",
    "knapsack_01_memoized",
    "knapsack_01_space_optimized",
    "can_partition_subset_sum",
    "fractional_knapsack",
    "matrix_chain_multiplication",
    "burst_balloons",
    "IndexedMinHeap",
    "merge_k_sorted_lists",
]
