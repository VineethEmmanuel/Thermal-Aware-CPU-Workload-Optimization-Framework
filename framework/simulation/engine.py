"""
Thermal-Aware CPU Workload Optimization Framework: Integrated Simulation Engine.

Orchestrates all 5 Prescribed DAA Components:
1. Indivisible Job Admission (0/1 Knapsack): Discrete task selection under thermal budget.
2. Divisible Resource Allocation (Fractional Knapsack): Fractional batch task scheduling.
3. Computation Reduction (Matrix Chain Multiplication): Minimizing scalar multiplications.
4. Dynamic Job Priority (Binomial vs Fibonacci Heap): Dynamic urgency scheduling & decrease-key.
5. Historical Telemetry Indexing (Red-Black Tree & B-Tree): High-throughput telemetry range indexing.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from framework.algorithms.knapsack_01 import Knapsack01Result, knapsack_01_dp
from framework.algorithms.knapsack_fractional import (
    FractionalKnapsackResult,
    compare_01_vs_fractional,
    fractional_knapsack_greedy,
)
from framework.algorithms.matrix_chain import MatrixChainResult, matrix_chain_order
from framework.config import SimulationConfig, ThermalConfig
from framework.data_structures.b_tree import BTree
from framework.data_structures.binomial_heap import BinomialHeap
from framework.data_structures.fibonacci_heap import FibonacciHeap
from framework.data_structures.red_black_tree import RedBlackTree
from framework.models import Task, TelemetryRecord, WorkloadType
from framework.simulation.thermal_model import CPUThermalModel


@dataclass
class SimulationRunReport:
    """Summary metrics of the complete thermal-aware CPU optimization execution."""
    thermal_budget: float
    # Phase 1: 0/1 Knapsack
    knapsack_01_result: Knapsack01Result
    # Phase 2: Fractional Knapsack
    fractional_result: FractionalKnapsackResult
    # Phase 3: Matrix Chain Optimization
    mcm_results: List[Tuple[str, MatrixChainResult]]
    total_mcm_ops_saved: int
    total_mcm_joules_saved: float
    # Phase 4: Heaps comparison
    binomial_metrics: Dict[str, int]
    fibonacci_metrics: Dict[str, int]
    # Phase 5: Telemetry Trees
    telemetry_samples_count: int
    rbt_metrics: Dict[str, int]
    btree_metrics: Dict[str, int]
    throttling_intervals: List[Tuple[float, float, float]]  # (t_start, t_end, peak_temp)
    # Physical thermal outcome
    final_core_temps: List[float]
    total_simulation_wall_time_sec: float


class WorkloadOptimizationSimulator:
    """
    Modular Simulator implementing the full Thermal-Aware CPU Workload Optimization Framework.
    """
    def __init__(self, thermal_cfg: ThermalConfig = ThermalConfig(), sim_cfg: SimulationConfig = SimulationConfig()):
        self.thermal_cfg = thermal_cfg
        self.sim_cfg = sim_cfg
        random.seed(sim_cfg.random_seed)
        
        # Instantiate CPU cores
        self.cores = [CPUThermalModel(core_id=i, config=thermal_cfg) for i in range(sim_cfg.num_cores)]
        
        # Instantiate telemetry indexing structures
        self.rbt_index: RedBlackTree[TelemetryRecord] = RedBlackTree()
        self.btree_index: BTree[TelemetryRecord] = BTree(t=sim_cfg.b_tree_min_degree)

    def generate_synthetic_workload(self) -> Tuple[List[Task], List[Task]]:
        """
        Generates realistic heterogeneous CPU workload:
        - Indivisible tasks: AI inference kernels, cryptography blocks, real-time audio threads.
        - Divisible tasks: Background rendering tiles, batch data compression, log aggregators.
        """
        indivisible_tasks = [
            Task(job_id="J_AI_01", name="ResNet50_Inference", workload_type=WorkloadType.MATRIX_PIPELINE,
                 value=95.0, cost=180.0, matrix_dims=[16, 256, 64, 512, 128], urgency=10.0, is_divisible=False),
            Task(job_id="J_CRYPTO_02", name="TLS_Session_Handshakes", workload_type=WorkloadType.COMPUTE_BOUND,
                 value=80.0, cost=120.0, urgency=15.0, is_divisible=False),
            Task(job_id="J_AUDIO_03", name="Spatial_DSP_Kernel", workload_type=WorkloadType.MATRIX_PIPELINE,
                 value=65.0, cost=90.0, matrix_dims=[32, 128, 32, 64], urgency=8.0, is_divisible=False),
            Task(job_id="J_PHYS_04", name="Physics_Engine_Collision", workload_type=WorkloadType.COMPUTE_BOUND,
                 value=70.0, cost=150.0, urgency=25.0, is_divisible=False),
            Task(job_id="J_DB_05", name="ACID_Commit_Flusher", workload_type=WorkloadType.COMPUTE_BOUND,
                 value=90.0, cost=110.0, urgency=5.0, is_divisible=False),
            Task(job_id="J_GRAPH_06", name="Pathfinding_AStar", workload_type=WorkloadType.COMPUTE_BOUND,
                 value=50.0, cost=85.0, urgency=30.0, is_divisible=False),
            Task(job_id="J_TRANS_07", name="VisionTransformer_Attention", workload_type=WorkloadType.MATRIX_PIPELINE,
                 value=110.0, cost=220.0, matrix_dims=[64, 512, 128, 512, 64], urgency=12.0, is_divisible=False),
        ]
        
        divisible_tasks = [
            Task(job_id="D_RENDER_01", name="RayTracing_TileWorker", workload_type=WorkloadType.BACKGROUND_BATCH,
                 value=120.0, cost=200.0, urgency=40.0, is_divisible=True),
            Task(job_id="D_COMPRESS_02", name="Zstd_Log_Compressor", workload_type=WorkloadType.BACKGROUND_BATCH,
                 value=60.0, cost=150.0, urgency=50.0, is_divisible=True),
            Task(job_id="D_INDEX_03", name="Search_Index_Builder", workload_type=WorkloadType.BACKGROUND_BATCH,
                 value=75.0, cost=130.0, urgency=45.0, is_divisible=True),
            Task(job_id="D_ML_BATCH_04", name="Model_Weight_Quantizer", workload_type=WorkloadType.BACKGROUND_BATCH,
                 value=90.0, cost=160.0, urgency=35.0, is_divisible=True),
        ]
        
        return indivisible_tasks, divisible_tasks

    def run_simulation(self, verbose: bool = True) -> SimulationRunReport:
        """
        Executes end-to-end thermal-aware CPU optimization pipeline.
        """
        start_wall_time = time.perf_counter()
        if verbose:
            print("=" * 80)
            print(">>> INITIALIZING THERMAL-AWARE CPU WORKLOAD OPTIMIZATION FRAMEWORK")
            print("=" * 80)

        # Step 1: Compute Safe Thermal Energy Headroom Budget
        avg_headroom = sum(c.calculate_thermal_headroom() for c in self.cores) / len(self.cores)
        total_thermal_budget = sum(c.compute_safe_thermal_budget(self.thermal_cfg.scheduling_window_sec) for c in self.cores)
        budget_int = int(round(total_thermal_budget))
        
        if verbose:
            print(f"\n[Thermal State Initialized]")
            print(f" - Cores Active: {len(self.cores)} | Ambient Temp: {self.thermal_cfg.ambient_temp}°C | Throttle: {self.thermal_cfg.throttle_temp}°C")
            print(f" - Current Core Headroom: {avg_headroom:.1f}°C")
            print(f" - Global Thermal Energy Budget (W): {budget_int} Joules (Capacity Units)")

        # Step 2: Workload Generation
        indivisible_tasks, divisible_tasks = self.generate_synthetic_workload()

        # Step 3: Phase 1 - Indivisible Job Admission (0/1 Knapsack)
        # Allocate 65% of the thermal budget for atomic indivisible tasks
        budget_01 = int(round(budget_int * 0.65))
        if verbose:
            print(f"\n" + "-" * 70)
            print(f"[Phase 1: Indivisible Job Admission (0/1 Knapsack DP)]")
            print(f" - Thermal Sub-Budget Allocated: {budget_01} units")
            print(f" - Candidate Discrete Tasks: {len(indivisible_tasks)}")

        res_01 = knapsack_01_dp(indivisible_tasks, budget_01)
        if verbose:
            print(f" - Admitted Tasks: {[t.job_id for t in res_01.selected_tasks]}")
            print(f" - Total Value Achieved: {res_01.max_value:.1f} | Thermal Cost: {res_01.total_cost:.1f}/{budget_01}")
            print(f" - DP Cell Updates: {res_01.cell_updates:,} | Runtime: {res_01.execution_time_sec * 1000:.3f} ms")

        # Step 4: Phase 2 - Divisible Resource Allocation (Fractional Knapsack)
        # Allocate remaining budget + unused 0/1 budget to divisible background tasks
        rem_headroom = (budget_int - res_01.total_cost)
        if verbose:
            print(f"\n" + "-" * 70)
            print(f"[Phase 2: Divisible Resource Allocation (Fractional Knapsack Greedy)]")
            print(f" - Available Headroom for Divisible Tasks: {rem_headroom:.1f} units")
            
        res_frac = fractional_knapsack_greedy(divisible_tasks, rem_headroom)
        if verbose:
            allocated = [(t.job_id, f"{frac:.2f}") for t, frac in res_frac.allocations if frac > 0]
            print(f" - Fractional Allocations (JobID, Fraction): {allocated}")
            print(f" - Total Value Achieved: {res_frac.max_value:.1f} | Thermal Cost Used: {res_frac.total_cost_used:.1f}")

        # Step 5: Phase 3 - Computation Reduction via Matrix Chain Multiplication (MCM)
        if verbose:
            print(f"\n" + "-" * 70)
            print(f"[Phase 3: Computation Reduction (Matrix Chain Multiplication DP)]")

        mcm_reports: List[Tuple[str, MatrixChainResult]] = []
        total_ops_saved = 0
        total_joules_saved = 0.0

        for task in res_01.selected_tasks:
            if task.workload_type == WorkloadType.MATRIX_PIPELINE and task.matrix_dims:
                mcm_res = matrix_chain_order(task.matrix_dims)
                mcm_reports.append((task.job_id, mcm_res))
                total_ops_saved += mcm_res.operations_saved
                total_joules_saved += mcm_res.estimated_joules_saved
                if verbose:
                    print(f" - Task [{task.job_id}] ({task.name}):")
                    print(f"   Matrix Chain Dims: {task.matrix_dims}")
                    print(f"   Optimal Order: {mcm_res.optimal_parenthesization}")
                    print(f"   Scalar Operations: Naive={mcm_res.naive_left_to_right_cost:,} -> Optimal={mcm_res.optimal_cost:,}")
                    print(f"   Cost Reduction: {mcm_res.cost_reduction_ratio:.1f}x ({mcm_res.operations_saved:,} ALU ops saved)")
                    print(f"   Estimated Energy Saved: {mcm_res.estimated_joules_saved:.4f} J (Die Temp Delta: -{mcm_res.estimated_temp_reduction_c:.4f}°C)")

        # Step 6: Phase 4 - Dynamic Priority Management (Binomial vs Fibonacci Heap)
        if verbose:
            print(f"\n" + "-" * 70)
            print(f"[Phase 4: Dynamic Priority Queue Comparison (Binomial vs Fibonacci Heap)]")

        b_heap: BinomialHeap[Task] = BinomialHeap()
        f_heap: FibonacciHeap[Task] = FibonacciHeap()

        # Insert admitted tasks into both heaps
        b_node_refs = []
        f_node_refs = []
        all_active_tasks = res_01.selected_tasks + [t for t, frac in res_frac.allocations if frac > 0]
        
        for task in all_active_tasks:
            bn = b_heap.insert(task.urgency, task)
            fn = f_heap.insert(task.urgency, task)
            b_node_refs.append((bn, task))
            f_node_refs.append((fn, task))

        # Dynamic priority adjustment simulation:
        # A task experiencing core thermal heating needs immediate dispatch -> decrease_key
        if b_node_refs and f_node_refs:
            # Promote the last inserted task to urgent priority 1.0
            target_b_node, target_task = b_node_refs[-1]
            target_f_node, _ = f_node_refs[-1]
            new_urgency = 1.0
            
            b_heap.decrease_key(target_b_node, new_urgency)
            f_heap.decrease_key(target_f_node, new_urgency)
            
            if verbose:
                print(f" - Dynamic Surge: Task [{target_task.job_id}] urgency boosted from {target_task.urgency} to {new_urgency}")
                print(f" - Binomial Heap Metrics: {b_heap.op_metrics}")
                print(f" - Fibonacci Heap Metrics: {f_heap.op_metrics}")

        # Step 7: Phase 5 - Telemetry Logging & Historical Indexing (Red-Black Tree & B-Tree)
        if verbose:
            print(f"\n" + "-" * 70)
            print(f"[Phase 5: Telemetry Stream Logging & Range Indexing (Red-Black Tree & B-Tree)]")

        dt = self.thermal_cfg.time_step_sec
        steps = int(self.thermal_cfg.scheduling_window_sec / dt)
        throttling_intervals: List[Tuple[float, float, float]] = []

        # Run CPU thermal simulation loop
        for s in range(steps):
            # Alternate active tasks
            active_job = all_active_tasks[s % len(all_active_tasks)].job_id if all_active_tasks else "IDLE"
            for core in self.cores:
                # Add intermittent heavy compute power
                power_load = 75.0 if s % 4 == 0 else 35.0
                telemetry = core.step(active_power_watts=power_load, dt=dt, active_job_id=active_job)
                
                # Index into both Red-Black Tree and B-Tree
                self.rbt_index.insert(telemetry.timestamp, telemetry)
                self.btree_index.insert(telemetry.timestamp, telemetry)

        # Range Query: Query historical samples during middle half of window
        query_start = self.thermal_cfg.scheduling_window_sec * 0.25
        query_end = self.thermal_cfg.scheduling_window_sec * 0.75
        rbt_records = self.rbt_index.range_search(query_start, query_end)
        btree_records = self.btree_index.range_search(query_start, query_end)

        # Check RB-Tree invariants
        is_rb_valid, rb_msg = self.rbt_index.validate_properties()
        if verbose:
            print(f" - Total Telemetry Records Indexed: {len(self.rbt_index)}")
            print(f" - Red-Black Tree Height: {self.rbt_index.height()} | Invariants Valid: {is_rb_valid}")
            print(f" - B-Tree Height: {self.btree_index.height()} | Min Degree t: {self.btree_index.t}")
            print(f" - Range Query [{query_start:.1f}s to {query_end:.1f}s]:")
            print(f"   * RBT Found: {len(rbt_records)} records | Rotations: {self.rbt_index.op_metrics['rotations']}")
            print(f"   * B-Tree Found: {len(btree_records)} records | Splits: {self.btree_index.op_metrics['splits']}")
            
            # Show sample telemetry record
            if rbt_records:
                sample_t, sample_rec = rbt_records[len(rbt_records) // 2]
                print(f"   * Sample Indexed Reading: {sample_rec}")

        elapsed_wall = time.perf_counter() - start_wall_time
        final_temps = [c.current_temp for c in self.cores]

        if verbose:
            print("\n" + "=" * 80)
            print(f">>> SIMULATION COMPLETE (Wall Time: {elapsed_wall * 1000:.2f} ms)")
            print(f">>> Final Core Temps: {[f'{t:.1f}°C' for t in final_temps]}")
            print("=" * 80 + "\n")

        return SimulationRunReport(
            thermal_budget=float(budget_int),
            knapsack_01_result=res_01,
            fractional_result=res_frac,
            mcm_results=mcm_reports,
            total_mcm_ops_saved=total_ops_saved,
            total_mcm_joules_saved=total_joules_saved,
            binomial_metrics=b_heap.op_metrics,
            fibonacci_metrics=f_heap.op_metrics,
            telemetry_samples_count=len(self.rbt_index),
            rbt_metrics=self.rbt_index.op_metrics,
            btree_metrics=self.btree_index.op_metrics,
            throttling_intervals=throttling_intervals,
            final_core_temps=final_temps,
            total_simulation_wall_time_sec=elapsed_wall
        )
