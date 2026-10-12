"""
Integration test for end-to-end thermal-aware CPU workload optimization simulator.
"""

import unittest

from framework.config import SimulationConfig, ThermalConfig
from framework.simulation.engine import WorkloadOptimizationSimulator


class TestSimulationIntegration(unittest.TestCase):
    """End-to-end integration tests for the CPU workload simulation framework."""

    def test_full_simulation_pipeline(self):
        """Runs the complete 5-phase optimization workflow and asserts invariants."""
        thermal_cfg = ThermalConfig(
            ambient_temp=25.0,
            throttle_temp=85.0,
            scheduling_window_sec=5.0,
            time_step_sec=0.5
        )
        sim_cfg = SimulationConfig(num_cores=2, random_seed=123)
        simulator = WorkloadOptimizationSimulator(thermal_cfg=thermal_cfg, sim_cfg=sim_cfg)
        
        report = simulator.run_simulation(verbose=False)

        # 1. Assert knapsack 0/1 admitted valid tasks
        self.assertGreater(report.knapsack_01_result.max_value, 0.0)
        self.assertLessEqual(report.knapsack_01_result.total_cost, report.knapsack_01_result.capacity)
        self.assertGreater(len(report.knapsack_01_result.selected_tasks), 0)

        # 2. Assert fractional knapsack allocated divisible tasks
        self.assertGreater(report.fractional_result.max_value, 0.0)

        # 3. Assert Matrix Chain Optimization yielded operations savings
        self.assertGreater(len(report.mcm_results), 0)
        for job_id, mcm_res in report.mcm_results:
            self.assertGreaterEqual(mcm_res.naive_left_to_right_cost, mcm_res.optimal_cost)
            self.assertGreater(len(mcm_res.optimal_parenthesization), 0)

        # 4. Assert Heap metric tracking
        self.assertGreater(report.binomial_metrics["inserts"], 0)
        self.assertGreater(report.fibonacci_metrics["inserts"], 0)

        # 5. Assert Telemetry Tree population & valid Red-Black properties
        self.assertGreater(report.telemetry_samples_count, 0)
        is_rbt_valid, msg = simulator.rbt_index.validate_properties()
        self.assertTrue(is_rbt_valid, f"RBT property failure: {msg}")
        self.assertGreater(len(simulator.btree_index), 0)

        # 6. Assert realistic thermal outcome
        for temp in report.final_core_temps:
            self.assertGreaterEqual(temp, thermal_cfg.ambient_temp)
            self.assertLess(temp, thermal_cfg.critical_temp)


if __name__ == "__main__":
    unittest.main()
