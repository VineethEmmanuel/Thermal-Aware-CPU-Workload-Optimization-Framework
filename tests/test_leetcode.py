"""
Unit tests for leetcode/ algorithmic problem solvers and pattern implementations.
"""

import unittest

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


class TestLeetCodeSolvers(unittest.TestCase):
    """Test suite for leetcode pattern implementations."""

    def test_01_knapsack_tabulation_and_memoization(self):
        """Tests 0/1 knapsack on textbook weights and values."""
        weights = [10, 20, 30]
        values = [60.0, 100.0, 120.0]
        capacity = 50
        
        max_val, indices = knapsack_01_tabulation(weights, values, capacity)
        self.assertEqual(max_val, 220.0)
        self.assertEqual(indices, [1, 2])
        
        opt_val = knapsack_01_space_optimized(weights, values, capacity)
        self.assertEqual(opt_val, 220.0)
        
        memo_val = knapsack_01_memoized(weights, values, capacity)
        self.assertEqual(memo_val, 220.0)

    def test_leetcode_416_partition_equal_subset_sum(self):
        """Tests LeetCode 416 Subset Sum decision pattern."""
        self.assertTrue(can_partition_subset_sum([1, 5, 11, 5]))  # [1, 5, 5] and [11]
        self.assertFalse(can_partition_subset_sum([1, 2, 3, 5]))

    def test_fractional_knapsack(self):
        """Tests continuous fractional knapsack greedy allocation."""
        weights = [10.0, 20.0, 30.0]
        values = [60.0, 100.0, 120.0]
        capacity = 50.0
        
        val, allocs = fractional_knapsack(weights, values, capacity)
        # 60 + 100 + (20/30)*120 = 240
        self.assertAlmostEqual(val, 240.0)
        self.assertEqual(len(allocs), 3)

    def test_matrix_chain_multiplication_with_counter(self):
        """Tests MCM DP and verifies operation counter."""
        dims = [10, 100, 5, 50]
        cost, order, ops = matrix_chain_multiplication(dims)
        self.assertEqual(cost, 7500)
        self.assertEqual(order, "((A1 x A2) x A3)")
        self.assertGreater(ops, 0)

    def test_leetcode_312_burst_balloons(self):
        """Tests LeetCode 312 Interval DP pattern."""
        nums = [3, 1, 5, 8]
        # Max coins is 167
        self.assertEqual(burst_balloons(nums), 167)

    def test_indexed_min_heap_decrease_key(self):
        """Tests IndexedMinHeap insert, extract_min, and decrease_key."""
        heap = IndexedMinHeap[str]()
        heap.insert("task_A", 50.0, "Job A")
        heap.insert("task_B", 30.0, "Job B")
        heap.insert("task_C", 70.0, "Job C")
        
        # Decrease task_C to priority 10.0
        heap.decrease_key("task_C", 10.0)
        
        p, item_id, payload = heap.extract_min()
        self.assertEqual(item_id, "task_C")
        self.assertEqual(p, 10.0)
        
        p2, item_id2, _ = heap.extract_min()
        self.assertEqual(item_id2, "task_B")
        self.assertEqual(p2, 30.0)

    def test_leetcode_23_merge_k_sorted_lists(self):
        """Tests LeetCode 23 Merge k Sorted Lists using Min-Heap."""
        lists = [
            [1, 4, 5],
            [1, 3, 4],
            [2, 6]
        ]
        merged = merge_k_sorted_lists(lists)
        self.assertEqual(merged, [1, 1, 2, 3, 4, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
