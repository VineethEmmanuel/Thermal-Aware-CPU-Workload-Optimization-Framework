#!/usr/bin/env python3
"""
Thermal-Aware CPU Workload Optimization Framework
Design and Analysis of Algorithms (DAA) - Assignment 2
Unified CLI Entry Point & Demonstration Runner
"""

import argparse
import os
import sys
import unittest

from framework.algorithms.knapsack_01 import knapsack_01_dp, knapsack_01_greedy_heuristic
from framework.algorithms.knapsack_fractional import compare_01_vs_fractional, fractional_knapsack_greedy
from framework.algorithms.matrix_chain import matrix_chain_order
from framework.analysis.complexity_evaluator import ComplexityEvaluator
from framework.analysis.visualizer import ASCIITableFormatter, PlotGenerator
from framework.config import SimulationConfig, ThermalConfig
from framework.models import Task, WorkloadType
from framework.simulation.engine import WorkloadOptimizationSimulator


def print_banner():
    banner = r"""
========================================================================================
     THERMAL-AWARE CPU WORKLOAD OPTIMIZATION FRAMEWORK (DAA ASSIGNMENT 2)
========================================================================================
Prescribed Algorithmic Components:
 [1] Indivisible Job Admission       : 0/1 Knapsack Dynamic Programming O(nW)
 [2] Divisible Resource Allocation   : Fractional Knapsack Greedy O(n log n) / O(n)
 [3] Computation Reduction           : Matrix Chain Multiplication DP O(n^3)
 [4] Dynamic Job Priority Management : Binomial Heap vs Fibonacci Heap
 [5] Historical Telemetry Indexing   : Red-Black Tree & B-Tree O(log N + k)
 [6] Scalability & Complexity Engine : Empirical Benchmarking & Pseudo-Polynomial Analysis
========================================================================================
"""
    print(banner)


def run_demonstration():
    """Executes full thermal CPU simulation walk-through."""
    print("\n[COMMAND] Executing Thermal CPU Simulation Walkthrough (--demo)...")
    sim = WorkloadOptimizationSimulator()
    report = sim.run_simulation(verbose=True)
    return report


def run_benchmarks(generate_plots: bool = False):
    """Executes scalability and complexity analysis across all DAA components."""
    print("\n" + "=" * 80)
    print(">>> EXECUTING DAA SCALABILITY & COMPLEXITY ANALYSIS SUITE")
    print("=" * 80)

    evaluator = ComplexityEvaluator()
    results_dict = {}

    # 1. Knapsack Pseudo-Polynomial vs Polynomial Analysis
    print("\n[1] Knapsack 0/1: Pseudo-Polynomial Complexity Analysis")
    records_w, records_n, discourse = evaluator.evaluate_pseudo_polynomial_knapsack()
    results_dict["knapsack_scaling_w"] = records_w
    results_dict["knapsack_scaling_n"] = records_n

    # Render Table for Scaling W (Fixed n=40)
    headers_w = ["Thermal Budget (W)", "Input Size (bits)", "Cell Updates", "Peak Mem (KB)", "Time (ms)", "Asymptotic O(nW)"]
    rows_w = [
        [r.parameter_val, f"{r.parameter_val.bit_length()} bits", f"{r.operations_count:,}",
         f"{r.peak_memory_kb:.1f}", f"{r.elapsed_ms:.3f}", f"~{40 * r.parameter_val:,}"]
        for r in records_w
    ]
    print(ASCIITableFormatter.render(headers_w, rows_w, title="0/1 Knapsack: Exponential Scaling of Numeric W (Fixed n=40)"))

    # Render Table for Scaling n (Fixed W=500)
    headers_n = ["Task Count (n)", "DP Table Cells", "Peak Mem (KB)", "Time (ms)", "Growth Ratio"]
    rows_n = [
        [r.parameter_val, f"{r.operations_count:,}", f"{r.peak_memory_kb:.1f}",
         f"{r.elapsed_ms:.3f}", f"{(r.elapsed_ms / records_n[0].elapsed_ms):.2f}x"]
        for r in records_n
    ]
    print(ASCIITableFormatter.render(headers_n, rows_n, title="0/1 Knapsack: Linear Scaling with Tasks n (Fixed W=500)"))
    print("\n" + discourse + "\n")

    # 2. 0/1 Knapsack vs Fractional Knapsack Integrality Gap
    print("\n[2] 0/1 Knapsack vs Fractional Knapsack Formulation Showdown")
    sample_tasks = [
        Task("T1", "Core_Compute_A", WorkloadType.COMPUTE_BOUND, value=60.0, cost=10.0),
        Task("T2", "Core_Compute_B", WorkloadType.COMPUTE_BOUND, value=100.0, cost=20.0),
        Task("T3", "Core_Compute_C", WorkloadType.COMPUTE_BOUND, value=120.0, cost=30.0),
        Task("T4", "Core_Compute_D", WorkloadType.COMPUTE_BOUND, value=70.0, cost=15.0),
        Task("T5", "Core_Compute_E", WorkloadType.COMPUTE_BOUND, value=90.0, cost=25.0),
    ]
    comp_report = compare_01_vs_fractional(sample_tasks, capacity=50)
    print(comp_report.analysis_text)
    
    headers_comp = ["Formulation", "Optimal Utility", "Capacity Used", "Time (ms)", "Integrality Gap (Delta)"]
    rows_comp = [
        ["0/1 Knapsack (DP)", f"{comp_report.val_01:.2f}", f"{comp_report.cost_01:.1f}/50.0", f"{comp_report.time_01_sec*1000:.3f} ms", "-"],
        ["Fractional (Greedy)", f"{comp_report.val_fractional:.2f}", f"{comp_report.cost_fractional:.1f}/50.0", f"{comp_report.time_fractional_sec*1000:.3f} ms", f"+{comp_report.duality_gap:.2f}"]
    ]
    print(ASCIITableFormatter.render(headers_comp, rows_comp, title="Integrality Relaxation Comparison (Capacity W=50)"))

    # 3. Matrix Chain Multiplication DP O(n^3) vs Catalan Brute Force
    print("\n[3] Matrix Chain Multiplication: DP O(n³) vs Brute-Force Catalan Explosion")
    mcm_data = evaluator.evaluate_mcm_dp_vs_bruteforce()
    results_dict["mcm_dp_vs_bruteforce"] = mcm_data

    headers_mcm = ["Chained Matrices (n)", "DP Time (ms)", "Brute-Force Time (ms)", "DP Speedup Factor", "Optimal Scalar Cost"]
    rows_mcm = [
        [d["n"], f"{d['dp_ms']:.4f}", f"{d['bf_ms']:.4f}", f"{d['speedup']:,.1f}x", f"{d['opt_cost']:,}"]
        for d in mcm_data
    ]
    print(ASCIITableFormatter.render(headers_mcm, rows_mcm, title="Matrix Chain Optimization: Cubic DP vs Catalan Recursion"))

    # 4. Binomial Heap vs Fibonacci Heap Bursty Decrease-Key
    print("\n[4] Dynamic Priority Heaps: Binomial vs Fibonacci Operational Profiling")
    heap_data = evaluator.evaluate_heaps_workload()
    results_dict["heap_comparison"] = heap_data

    headers_heap = ["Workload Elements (N)", "Binomial Time (ms)", "Fibonacci Time (ms)", "Binomial Comparisons", "Fibonacci Comparisons", "Cascading Cuts"]
    rows_heap = [
        [d["num_ops"], f"{d['b_ms']:.3f}", f"{d['f_ms']:.3f}", f"{d['b_comparisons']:,}",
         f"{d['f_comparisons']:,}", f"{d['f_cascading_cuts']:,}"]
        for d in heap_data
    ]
    print(ASCIITableFormatter.render(headers_heap, rows_heap, title="Dynamic Heap Benchmark (Inserts + Bursty Decrease-Keys)"))

    # 5. Historical Telemetry Indexing: Red-Black Tree vs B-Tree vs Linear Scan
    print("\n[5] Historical Telemetry Indexing: Red-Black Tree vs B-Tree vs Linear Scan")
    telemetry_data = evaluator.evaluate_telemetry_indexing()
    results_dict["telemetry_indexing"] = telemetry_data

    headers_tel = ["Records (N)", "RBT Time (ms)", "B-Tree (t=3) (ms)", "Linear Scan (ms)", "RBT Speedup", "RBT Height", "B-Tree Height"]
    rows_tel = [
        [d["size"], f"{d['rbt_ms']:.3f}", f"{d['btree_ms']:.3f}", f"{d['linear_ms']:.3f}",
         f"{d['rbt_speedup']:.1f}x", d["rbt_height"], d["btree_height"]]
        for d in telemetry_data
    ]
    print(ASCIITableFormatter.render(headers_tel, rows_tel, title="Telemetry Range Query Latency (50 Random Intervals)"))

    # Plot generation if requested
    if generate_plots:
        print("\n[Plotting] Generating scientific benchmark figures via matplotlib...")
        plots = PlotGenerator.generate_all_plots(results_dict, output_dir="benchmark_plots")
        for p in plots:
            print(f" - Saved Plot: {os.path.abspath(p)}")


def run_unit_tests():
    """Runs full test suite."""
    print("\n[COMMAND] Discovering and running unit & integration test suites (--test)...")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        description="Thermal-Aware CPU Workload Optimization Framework (DAA Assignment 2)"
    )
    parser.add_argument("--demo", action="store_true", help="Run the thermal workload optimization simulation walkthrough.")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive scalability and complexity benchmarks.")
    parser.add_argument("--plot", action="store_true", help="Generate scientific PNG benchmark graphs into benchmark_plots/.")
    parser.add_argument("--test", action="store_true", help="Execute the complete test suite.")
    parser.add_argument("--all", action="store_true", help="Run simulation, benchmarks, plots, and tests sequentially.")

    args = parser.parse_args()

    # If no flags provided, run demo and concise benchmark
    if not (args.demo or args.benchmark or args.plot or args.test or args.all):
        print("[Notice] No specific flag provided. Defaulting to: Simulation Walkthrough + Benchmarks.")
        run_demonstration()
        run_benchmarks(generate_plots=False)
        print("\n[TIP] Run with '--plot' to generate figures, or '--test' to run the test suite.\n")
        return

    if args.all:
        run_demonstration()
        run_benchmarks(generate_plots=True)
        run_unit_tests()
        return

    if args.demo:
        run_demonstration()

    if args.benchmark or args.plot:
        run_benchmarks(generate_plots=args.plot)

    if args.test:
        run_unit_tests()


if __name__ == "__main__":
    main()
