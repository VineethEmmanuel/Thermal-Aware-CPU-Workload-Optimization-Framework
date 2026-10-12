"""
Unit tests for 0/1 Knapsack and Fractional Knapsack implementations.
Covers normal, boundary, and pathological inputs.
"""

import unittest

from framework.algorithms.knapsack_01 import (
    knapsack_01_dp,
    knapsack_01_dp_space_optimized,
    knapsack_01_greedy_heuristic,
    knapsack_01_memoized,
)
from framework.algorithms.knapsack_fractional import (
    compare_01_vs_fractional,
    fractional_knapsack_greedy,
    fractional_knapsack_linear,
)
from framework.models import Task, WorkloadType


class TestKnapsackAlgorithms(unittest.TestCase):
    """Test suite for 0/1 and Fractional Knapsack solvers."""

    def setUp(self):
        # Classic textbook instance (Cormen et al.)
        self.standard_tasks = [
            Task(job_id="T1", name="Task1", workload_type=WorkloadType.COMPUTE_BOUND, value=60.0, cost=10.0),
            Task(job_id="T2", name="Task2", workload_type=WorkloadType.COMPUTE_BOUND, value=100.0, cost=20.0),
            Task(job_id="T3", name="Task3", workload_type=WorkloadType.COMPUTE_BOUND, value=120.0, cost=30.0),
        ]

    def test_knapsack_01_dp_standard(self):
        """Tests standard 0/1 knapsack where optimal is picking items 2 and 3."""
        capacity = 50
        res = knapsack_01_dp(self.standard_tasks, capacity)
        # Optimal combination: T2 (cost 20, val 100) + T3 (cost 30, val 120) = 220
        self.assertEqual(res.max_value, 220.0)
        self.assertEqual(res.total_cost, 50.0)
        selected_ids = {t.job_id for t in res.selected_tasks}
        self.assertEqual(selected_ids, {"T2", "T3"})

    def test_knapsack_01_memoized_matches_dp(self):
        """Verifies memoized top-down DP matches bottom-up tabulation."""
        capacity = 50
        res_dp = knapsack_01_dp(self.standard_tasks, capacity)
        res_memo = knapsack_01_memoized(self.standard_tasks, capacity)
        self.assertEqual(res_dp.max_value, res_memo.max_value)
        self.assertEqual(res_dp.total_cost, res_memo.total_cost)

    def test_knapsack_01_space_optimized_matches_dp(self):
        """Verifies 1D space-optimized DP matches standard 2D table optimal value."""
        capacity = 50
        res_dp = knapsack_01_dp(self.standard_tasks, capacity)
        max_val_opt, _, _ = knapsack_01_dp_space_optimized(self.standard_tasks, capacity)
        self.assertEqual(res_dp.max_value, max_val_opt)

    def test_greedy_fails_on_01_counterexample(self):
        """
        Formal DAA proof test: demonstrates that greedy heuristic yields suboptimal
        result for 0/1 knapsack on the classic counterexample.
        """
        capacity = 50
        res_dp = knapsack_01_dp(self.standard_tasks, capacity)
        res_greedy = knapsack_01_greedy_heuristic(self.standard_tasks, capacity)
        
        # Greedy picks T1 (ratio 6) and T2 (ratio 5), cannot fit T3 -> total value = 160
        self.assertEqual(res_greedy.max_value, 160.0)
        # Optimal DP achieves 220
        self.assertEqual(res_dp.max_value, 220.0)
        self.assertLess(res_greedy.max_value, res_dp.max_value)

    def test_boundary_empty_tasks(self):
        """Boundary test: empty task list."""
        res_01 = knapsack_01_dp([], 50)
        self.assertEqual(res_01.max_value, 0.0)
        self.assertEqual(res_01.selected_tasks, [])
        
        res_frac = fractional_knapsack_greedy([], 50.0)
        self.assertEqual(res_frac.max_value, 0.0)

    def test_boundary_zero_capacity(self):
        """Boundary test: capacity = 0."""
        res_01 = knapsack_01_dp(self.standard_tasks, 0)
        self.assertEqual(res_01.max_value, 0.0)
        self.assertEqual(res_01.total_cost, 0.0)
        self.assertEqual(res_01.selected_tasks, [])

    def test_boundary_capacity_smaller_than_smallest_task(self):
        """Boundary test: capacity is smaller than any task cost."""
        res = knapsack_01_dp(self.standard_tasks, 5)
        self.assertEqual(res.max_value, 0.0)
        self.assertEqual(len(res.selected_tasks), 0)

    def test_boundary_all_tasks_fit(self):
        """Boundary test: capacity exceeds sum of all task costs."""
        large_capacity = 1000
        res = knapsack_01_dp(self.standard_tasks, large_capacity)
        expected_total = sum(t.value for t in self.standard_tasks)
        self.assertEqual(res.max_value, expected_total)
        self.assertEqual(len(res.selected_tasks), len(self.standard_tasks))

    def test_fractional_knapsack_greedy_optimal(self):
        """Tests continuous fractional knapsack on standard instance."""
        capacity = 50.0
        res = fractional_knapsack_greedy(self.standard_tasks, capacity)
        # Greedy takes all of T1 (w=10, v=60, rem=40), all of T2 (w=20, v=100, rem=20),
        # and fraction (20/30) of T3 (v=120 * 2/3 = 80) -> Total = 60 + 100 + 80 = 240
        self.assertAlmostEqual(res.max_value, 240.0)
        self.assertAlmostEqual(res.total_cost_used, 50.0)

    def test_fractional_knapsack_linear_quickselect(self):
        """Tests that O(n) QuickSelect fractional knapsack achieves the exact same optimal value."""
        capacity = 50.0
        res_sort = fractional_knapsack_greedy(self.standard_tasks, capacity)
        res_linear = fractional_knapsack_linear(self.standard_tasks, capacity)
        self.assertAlmostEqual(res_sort.max_value, res_linear.max_value, places=5)

    def test_integrality_gap_bounding_property(self):
        """Validates fundamental DAA theorem: V_01 <= V_fractional for all capacities."""
        for cap in [10, 20, 30, 40, 50, 60]:
            res_01 = knapsack_01_dp(self.standard_tasks, cap)
            res_frac = fractional_knapsack_greedy(self.standard_tasks, float(cap))
            self.assertLessEqual(res_01.max_value, res_frac.max_value)

    def test_negative_capacity_exception(self):
        """Pathological input: negative capacity raises ValueError."""
        with self.assertRaises(ValueError):
            knapsack_01_dp(self.standard_tasks, -10)


if __name__ == "__main__":
    unittest.main()
