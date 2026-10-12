"""
Unit tests for Matrix Chain Multiplication (MCM) optimization.
Covers normal, boundary, and pathological inputs.
"""

import unittest

from framework.algorithms.matrix_chain import (
    matrix_chain_brute_force,
    matrix_chain_order,
    reconstruct_optimal_parenthesization,
)


class TestMatrixChainMultiplication(unittest.TestCase):
    """Test suite for MCM Dynamic Programming solver."""

    def test_single_matrix_base_case(self):
        """Boundary test: 1 matrix (dims len = 2). Cost must be 0."""
        dims = [10, 20]
        res = matrix_chain_order(dims)
        self.assertEqual(res.optimal_cost, 0)
        self.assertEqual(res.optimal_parenthesization, "A1")

    def test_two_matrices(self):
        """Boundary test: 2 matrices A1(10x20) and A2(20x30). Cost = 10*20*30 = 6000."""
        dims = [10, 20, 30]
        res = matrix_chain_order(dims)
        self.assertEqual(res.optimal_cost, 6000)
        self.assertEqual(res.optimal_parenthesization, "(A1 x A2)")

    def test_three_matrices_classic(self):
        """
        Classic textbook test: dims = [10, 100, 5, 50].
        (A1 x A2) x A3 = 10*100*5 + 10*5*50 = 5000 + 2500 = 7500.
        A1 x (A2 x A3) = 100*5*50 + 10*100*50 = 25000 + 50000 = 75000.
        Optimal is 7,500.
        """
        dims = [10, 100, 5, 50]
        res = matrix_chain_order(dims)
        self.assertEqual(res.optimal_cost, 7500)
        self.assertEqual(res.optimal_parenthesization, "((A1 x A2) x A3)")
        self.assertEqual(res.naive_right_to_left_cost, 75000)
        self.assertGreater(res.operations_saved, 0)

    def test_clrs_six_matrices(self):
        """
        Classic CLRS 6-matrix benchmark:
        Dimensions: <30, 35, 15, 5, 10, 20, 25>
        Expected optimal scalar multiplication cost: 15,125.
        Expected optimal parenthesization: ((A1 x (A2 x A3)) x ((A4 x A5) x A6))
        """
        dims = [30, 35, 15, 5, 10, 20, 25]
        res = matrix_chain_order(dims)
        self.assertEqual(res.optimal_cost, 15125)
        self.assertEqual(res.optimal_parenthesization, "((A1 x (A2 x A3)) x ((A4 x A5) x A6))")

    def test_dp_matches_brute_force_small_chains(self):
        """Validates that DP produces the exact same optimal cost as exhaustive brute-force."""
        for dims in [
            [10, 20, 30, 40],
            [5, 10, 20, 15, 30],
            [12, 8, 25, 4, 15],
        ]:
            dp_res = matrix_chain_order(dims)
            bf_cost = matrix_chain_brute_force(dims)
            self.assertEqual(dp_res.optimal_cost, bf_cost)

    def test_thermal_energy_saved_calculation(self):
        """Verifies thermal energy reduction calculation."""
        dims = [10, 100, 5, 50]
        res = matrix_chain_order(dims, joules_per_flop=1e-6, thermal_capacitance=100.0)
        self.assertGreater(res.estimated_joules_saved, 0.0)
        self.assertGreater(res.estimated_temp_reduction_c, 0.0)

    def test_invalid_dimension_exception(self):
        """Pathological input: dims with fewer than 2 elements."""
        with self.assertRaises(ValueError):
            matrix_chain_order([10])


if __name__ == "__main__":
    unittest.main()
