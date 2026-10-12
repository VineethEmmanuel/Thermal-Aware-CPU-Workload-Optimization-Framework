"""
Divisible Resource Allocation: Fractional Knapsack Problem Formulation and Solvers.

Problem Formulation:
--------------------
Given n divisible tasks T = {t_1, t_2, ..., t_n}, where each task has an SLA value
v_i >= 0, thermal/compute cost w_i > 0, and can be allocated fractionally:
    Maximize:   sum_{i=1}^n v_i * x_i
    Subject to: sum_{i=1}^n w_i * x_i <= W
                0 <= x_i <= 1 for all i in {1, ..., n}

Theoretical Complexity & LeetCode Pattern Mapping:
--------------------------------------------------
- LeetCode Pattern: Greedy Allocation / Ratio Sorting (e.g., LeetCode 134, 1710, Continuous Relaxation)
- Standard Greedy (Sorting by v_i / w_i): Time O(n log n), Space O(n)
- Linear-Time Greedy (Median-of-Medians / QuickSelect): Time O(n), Space O(n)
- DAA Contrast with 0/1 Knapsack:
  1. Integrality Relaxation: Relaxing x_i in {0, 1} to x_i in [0, 1] transforms
     the problem from an NP-complete combinatorial packing problem into a continuous
     optimization problem with the Greedy-Choice Property.
  2. Optimal Substructure: Both 0/1 and Fractional exhibit optimal substructure, but
     only Fractional exhibits the greedy-choice property. In 0/1 knapsack, an item with
     the highest density might leave unused capacity that cannot be filled by remaining
     discrete items, causing greedy to yield suboptimal total utility.
  3. Bounding Property: For any problem instance, V_opt(0/1) <= V_opt(Fractional).
"""

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from framework.algorithms.knapsack_01 import Knapsack01Result, knapsack_01_dp
from framework.models import Task


@dataclass
class FractionalKnapsackResult:
    """Encapsulates the solution to the continuous/fractional knapsack problem."""
    max_value: float
    total_cost_used: float
    capacity: float
    allocations: List[Tuple[Task, float]]  # (Task, allocated_fraction in [0.0, 1.0])
    execution_time_sec: float
    algorithm: str

    def __repr__(self) -> str:
        allocated_summary = [f"{t.job_id}:{frac:.2f}" for t, frac in self.allocations if frac > 0]
        return (f"FractionalKnapsackResult(algo='{self.algorithm}', max_val={self.max_value:.2f}, "
                f"cost={self.total_cost_used:.1f}/{self.capacity}, "
                f"allocations={allocated_summary}, time={self.execution_time_sec * 1000:.3f}ms)")


def fractional_knapsack_greedy(tasks: List[Task], capacity: float) -> FractionalKnapsackResult:
    """
    Standard Greedy algorithm using O(n log n) sorting by value density (v/w).
    
    Demonstrates the greedy-choice property: always picking as much as possible
    of the task with the highest marginal utility per unit thermal resource.
    """
    start_time = time.perf_counter()
    if capacity <= 0 or not tasks:
        return FractionalKnapsackResult(0.0, 0.0, capacity, [], time.perf_counter() - start_time, "Greedy_Sort")
    
    # Clone tasks to avoid mutating caller references
    task_copies = [copy.deepcopy(t) for t in tasks]
    # Sort descending by value density
    task_copies.sort(key=lambda t: t.value_density, reverse=True)
    
    total_val = 0.0
    rem_cap = capacity
    allocations: List[Tuple[Task, float]] = []
    
    for t in task_copies:
        if rem_cap <= 0:
            allocations.append((t, 0.0))
            continue
            
        if t.cost <= rem_cap:
            # Take the entire job (fraction = 1.0)
            t.allocated_fraction = 1.0
            total_val += t.value
            rem_cap -= t.cost
            allocations.append((t, 1.0))
        else:
            # Take fractional slice of the job
            fraction = rem_cap / t.cost
            t.allocated_fraction = fraction
            total_val += t.value * fraction
            rem_cap = 0.0
            allocations.append((t, fraction))
            
    elapsed = time.perf_counter() - start_time
    used_cap = capacity - rem_cap
    return FractionalKnapsackResult(
        max_value=total_val,
        total_cost_used=used_cap,
        capacity=capacity,
        allocations=allocations,
        execution_time_sec=elapsed,
        algorithm="Greedy_Sort"
    )


def fractional_knapsack_linear(tasks: List[Task], capacity: float) -> FractionalKnapsackResult:
    """
    Advanced DAA implementation: O(n) worst-case time using QuickSelect / Partitioning.
    
    Finds the threshold ratio without completely sorting all elements, achieving
    linear time complexity O(n).
    """
    start_time = time.perf_counter()
    if capacity <= 0 or not tasks:
        return FractionalKnapsackResult(0.0, 0.0, capacity, [], time.perf_counter() - start_time, "Greedy_QuickSelect")
        
    task_copies = [copy.deepcopy(t) for t in tasks]
    n = len(task_copies)
    
    def quickselect_knapsack(items: List[Task], rem_cap: float) -> Tuple[float, List[Tuple[Task, float]]]:
        if not items or rem_cap <= 0:
            return 0.0, [(t, 0.0) for t in items]
            
        if len(items) == 1:
            item = items[0]
            if item.cost <= rem_cap:
                item.allocated_fraction = 1.0
                return item.value, [(item, 1.0)]
            else:
                frac = rem_cap / item.cost
                item.allocated_fraction = frac
                return item.value * frac, [(item, frac)]
                
        # Pick pivot (median of first, middle, last)
        pivot_task = items[len(items) // 2]
        pivot_ratio = pivot_task.value_density
        
        # Partition into Greater, Equal, Lesser ratios
        greater = [t for t in items if t.value_density > pivot_ratio]
        equal = [t for t in items if t.value_density == pivot_ratio]
        lesser = [t for t in items if t.value_density < pivot_ratio]
        
        sum_w_greater = sum(t.cost for t in greater)
        
        if sum_w_greater >= rem_cap:
            # Optimal subset is entirely within greater partition
            val_g, alloc_g = quickselect_knapsack(greater, rem_cap)
            alloc_others = [(t, 0.0) for t in equal + lesser]
            return val_g, alloc_g + alloc_others
        else:
            # All items in greater can be taken completely
            val_g = sum(t.value for t in greater)
            alloc_g = [(t, 1.0) for t in greater]
            new_cap = rem_cap - sum_w_greater
            
            sum_w_equal = sum(t.cost for t in equal)
            if sum_w_equal >= new_cap:
                # Fill remaining budget with items from equal partition
                val_eq = 0.0
                alloc_eq = []
                cur = new_cap
                for t in equal:
                    if t.cost <= cur:
                        t.allocated_fraction = 1.0
                        val_eq += t.value
                        cur -= t.cost
                        alloc_eq.append((t, 1.0))
                    elif cur > 0:
                        frac = cur / t.cost
                        t.allocated_fraction = frac
                        val_eq += t.value * frac
                        cur = 0.0
                        alloc_eq.append((t, frac))
                    else:
                        alloc_eq.append((t, 0.0))
                alloc_less = [(t, 0.0) for t in lesser]
                return val_g + val_eq, alloc_g + alloc_eq + alloc_less
            else:
                # Take all equal items completely, search into lesser
                val_eq = sum(t.value for t in equal)
                alloc_eq = [(t, 1.0) for t in equal]
                new_cap -= sum_w_equal
                val_less, alloc_less = quickselect_knapsack(lesser, new_cap)
                return val_g + val_eq + val_less, alloc_g + alloc_eq + alloc_less

    total_val, allocations = quickselect_knapsack(task_copies, capacity)
    elapsed = time.perf_counter() - start_time
    used_cap = sum(t.cost * frac for t, frac in allocations)
    
    return FractionalKnapsackResult(
        max_value=total_val,
        total_cost_used=used_cap,
        capacity=capacity,
        allocations=allocations,
        execution_time_sec=elapsed,
        algorithm="Greedy_QuickSelect"
    )


@dataclass
class KnapsackComparisonReport:
    """Comparative analysis report contrasting 0/1 vs Fractional Knapsack formulations."""
    capacity: float
    val_01: float
    cost_01: float
    time_01_sec: float
    tasks_01: List[str]
    
    val_fractional: float
    cost_fractional: float
    time_fractional_sec: float
    allocations_fractional: Dict[str, float]
    
    duality_gap: float            # V_frac - V_01
    integrality_gap_ratio: float  # V_frac / V_01 if V_01 > 0 else inf
    analysis_text: str


def compare_01_vs_fractional(tasks: List[Task], capacity: int) -> KnapsackComparisonReport:
    """
    Directly contrasts the 0/1 (integer programming) and Fractional (linear programming relaxation)
    formulations on the identical task set and capacity budget.
    """
    res_01 = knapsack_01_dp(tasks, capacity)
    res_frac = fractional_knapsack_greedy(tasks, float(capacity))
    
    duality_gap = res_frac.max_value - res_01.max_value
    gap_ratio = (res_frac.max_value / res_01.max_value) if res_01.max_value > 0 else float('inf')
    
    analysis = (
        f"DAA COMPARATIVE ANALYSIS:\n"
        f"1. Optimal Utility: 0/1={res_01.max_value:.2f} vs Fractional={res_frac.max_value:.2f}\n"
        f"2. Integrality Gap: Delta={duality_gap:.2f} (Ratio={gap_ratio:.3f})\n"
        f"   - Theory validates V_01 <= V_fractional because fractional knapsack relaxes the integrality\n"
        f"     constraint x_i in {{0, 1}} to continuous interval x_i in [0, 1], expanding the feasible region.\n"
        f"3. Algorithmic Paradigm:\n"
        f"   - 0/1 Knapsack requires Dynamic Programming O(nW) or Branch & Bound due to absence of greedy-choice.\n"
        f"   - Fractional Knapsack possesses the Greedy-Choice Property, solvable optimally in O(n log n) or O(n).\n"
        f"4. CPU Workload Context:\n"
        f"   - Indivisible tasks represent atomic transactions / non-preemptible thread execution.\n"
        f"   - Divisible tasks model time-sliced batch workers / malleable compute threads."
    )
    
    frac_map = {t.job_id: frac for t, frac in res_frac.allocations if frac > 0}
    
    return KnapsackComparisonReport(
        capacity=float(capacity),
        val_01=res_01.max_value,
        cost_01=res_01.total_cost,
        time_01_sec=res_01.execution_time_sec,
        tasks_01=[t.job_id for t in res_01.selected_tasks],
        val_fractional=res_frac.max_value,
        cost_fractional=res_frac.total_cost_used,
        time_fractional_sec=res_frac.execution_time_sec,
        allocations_fractional=frac_map,
        duality_gap=duality_gap,
        integrality_gap_ratio=gap_ratio,
        analysis_text=analysis
    )
