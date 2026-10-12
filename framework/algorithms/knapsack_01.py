"""
Indivisible Job Admission: 0/1 Knapsack Problem Formulation and Solvers.

Problem Formulation:
--------------------
Given a set of n indivisible tasks T = {t_1, t_2, ..., t_n}, each with an SLA value
v_i >= 0 and a thermal energy/compute cost w_i > 0, and a strict CPU thermal headroom
budget W >= 0 within a scheduling window:
    Maximize:   sum_{i=1}^n v_i * x_i
    Subject to: sum_{i=1}^n w_i * x_i <= W
                x_i in {0, 1} for all i in {1, ..., n}

Theoretical Complexity:
-----------------------
- Dynamic Programming Tabulation: Time O(n * W), Space O(n * W)
- Space-Optimized DP: Time O(n * W), Space O(W)
- Memoized Top-Down DP: Time O(n * W), Space O(n * W) recursion/memo overhead
- Complexity Class: NP-complete in general. The DP algorithm runs in
  PSEUDO-POLYNOMIAL time, because the complexity depends linearly on the numeric
  magnitude W, which is exponential in the input size (bits required: log2 W).
- Greedy Heuristic: Time O(n log n), Space O(1). Suboptimal for 0/1 Knapsack.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from framework.models import Task


@dataclass
class Knapsack01Result:
    """Encapsulates the solution to the 0/1 Knapsack optimization problem."""
    max_value: float
    total_cost: float
    capacity: int
    selected_tasks: List[Task]
    dp_table: Optional[List[List[float]]]
    cell_updates: int
    execution_time_sec: float
    method: str

    def __repr__(self) -> str:
        selected_ids = [t.job_id for t in self.selected_tasks]
        return (f"Knapsack01Result(method='{self.method}', max_val={self.max_value:.2f}, "
                f"cost={self.total_cost:.1f}/{self.capacity}, "
                f"tasks={selected_ids}, cell_updates={self.cell_updates}, "
                f"time={self.execution_time_sec * 1000:.3f}ms)")


def knapsack_01_dp(tasks: List[Task], capacity: int) -> Knapsack01Result:
    """
    Solves 0/1 Knapsack using classic Dynamic Programming bottom-up tabulation.
    
    Reconstructs the optimal set of admitted indivisible tasks by backtracking
    through the DP table.
    
    Args:
        tasks: List of candidate tasks.
        capacity: Integer capacity budget (thermal energy headroom).
        
    Returns:
        Knapsack01Result with optimal value, admitted tasks, and DP table.
    """
    start_time = time.perf_counter()
    
    # Boundary checks
    if capacity < 0:
        raise ValueError(f"Capacity cannot be negative: {capacity}")
    
    n = len(tasks)
    if n == 0 or capacity == 0:
        return Knapsack01Result(
            max_value=0.0,
            total_cost=0.0,
            capacity=capacity,
            selected_tasks=[],
            dp_table=[[0.0] * (capacity + 1)] if capacity >= 0 else None,
            cell_updates=0,
            execution_time_sec=time.perf_counter() - start_time,
            method="DP_Tabulation"
        )
    
    # Initialize DP table: dimensions (n + 1) x (capacity + 1)
    # dp[i][w] = maximum utility achievable using a subset of the first i tasks with budget w
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    cell_updates = 0
    
    # Bottom-up Tabulation
    for i in range(1, n + 1):
        task = tasks[i - 1]
        weight = int(round(task.cost))
        val = task.value
        
        for w in range(1, capacity + 1):
            cell_updates += 1
            if weight <= w:
                # Choice: either exclude task i, or include task i
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + val)
            else:
                # Cannot include task i if its cost exceeds current capacity subproblem
                dp[i][w] = dp[i - 1][w]
                
    max_value = dp[n][capacity]
    
    # Backtrack to reconstruct the optimal task set
    selected: List[Task] = []
    curr_w = capacity
    for i in range(n, 0, -1):
        # If value came from dp[i-1][curr_w], item was not included
        if dp[i][curr_w] != dp[i - 1][curr_w]:
            task = tasks[i - 1]
            selected.append(task)
            curr_w -= int(round(task.cost))
            
    selected.reverse()
    total_cost = sum(t.cost for t in selected)
    
    elapsed = time.perf_counter() - start_time
    return Knapsack01Result(
        max_value=max_value,
        total_cost=total_cost,
        capacity=capacity,
        selected_tasks=selected,
        dp_table=dp,
        cell_updates=cell_updates,
        execution_time_sec=elapsed,
        method="DP_Tabulation"
    )


def knapsack_01_dp_space_optimized(tasks: List[Task], capacity: int) -> Tuple[float, int, float]:
    """
    Space-optimized DP table using 1D rolling array.
    Reduces auxiliary space from O(nW) to O(W).
    
    Returns:
        (max_value, cell_updates, execution_time_sec)
    """
    start_time = time.perf_counter()
    if capacity <= 0 or not tasks:
        return 0.0, 0, time.perf_counter() - start_time
        
    dp = [0.0] * (capacity + 1)
    cell_updates = 0
    
    for task in tasks:
        weight = int(round(task.cost))
        val = task.value
        # Iterate backwards to prevent using the same item multiple times in one step
        for w in range(capacity, weight - 1, -1):
            cell_updates += 1
            if dp[w - weight] + val > dp[w]:
                dp[w] = dp[w - weight] + val
                
    elapsed = time.perf_counter() - start_time
    return dp[capacity], cell_updates, elapsed


def knapsack_01_memoized(tasks: List[Task], capacity: int) -> Knapsack01Result:
    """
    Top-Down Recursion with Memoization for 0/1 Knapsack.
    Avoids computing subproblems that are not reachable from the root state (n, W).
    """
    start_time = time.perf_counter()
    if capacity < 0:
        raise ValueError(f"Capacity cannot be negative: {capacity}")
    
    n = len(tasks)
    if n == 0 or capacity == 0:
        return Knapsack01Result(
            max_value=0.0,
            total_cost=0.0,
            capacity=capacity,
            selected_tasks=[],
            dp_table=None,
            cell_updates=0,
            execution_time_sec=time.perf_counter() - start_time,
            method="DP_Memoized"
        )
        
    memo: Dict[Tuple[int, int], float] = {}
    chosen_decisions: Dict[Tuple[int, int], bool] = {}
    updates = 0
    
    def solve(idx: int, rem_w: int) -> float:
        nonlocal updates
        if idx < 0 or rem_w <= 0:
            return 0.0
        state = (idx, rem_w)
        if state in memo:
            return memo[state]
            
        updates += 1
        task = tasks[idx]
        w = int(round(task.cost))
        
        # Option 1: Do not pick
        val_without = solve(idx - 1, rem_w)
        
        # Option 2: Pick (if affordable)
        if w <= rem_w:
            val_with = solve(idx - 1, rem_w - w) + task.value
            if val_with > val_without:
                memo[state] = val_with
                chosen_decisions[state] = True
                return val_with
                
        memo[state] = val_without
        chosen_decisions[state] = False
        return val_without
        
    max_val = solve(n - 1, capacity)
    
    # Backtrack decisions
    selected: List[Task] = []
    curr_w = capacity
    for idx in range(n - 1, -1, -1):
        if chosen_decisions.get((idx, curr_w), False):
            selected.append(tasks[idx])
            curr_w -= int(round(tasks[idx].cost))
            
    selected.reverse()
    total_cost = sum(t.cost for t in selected)
    elapsed = time.perf_counter() - start_time
    
    return Knapsack01Result(
        max_value=max_val,
        total_cost=total_cost,
        capacity=capacity,
        selected_tasks=selected,
        dp_table=None,
        cell_updates=updates,
        execution_time_sec=elapsed,
        method="DP_Memoized"
    )


def knapsack_01_greedy_heuristic(tasks: List[Task], capacity: int) -> Knapsack01Result:
    """
    Greedy heuristic for 0/1 knapsack: Sort by value density (v/w) descending
    and greedily admit tasks that fit indivisibly.
    
    NOTE: In DAA theory, greedy is NOT guaranteed to find the optimal solution
    for 0/1 knapsack because the discrete constraint prevents filling leftover capacity.
    """
    start_time = time.perf_counter()
    if capacity <= 0 or not tasks:
        return Knapsack01Result(0.0, 0.0, capacity, [], None, 0, time.perf_counter() - start_time, "Greedy_Heuristic")
        
    # Sort tasks by value density in descending order
    sorted_tasks = sorted(tasks, key=lambda t: t.value_density, reverse=True)
    
    selected: List[Task] = []
    curr_cost = 0.0
    curr_val = 0.0
    comparisons = 0
    
    for task in sorted_tasks:
        comparisons += 1
        if curr_cost + task.cost <= capacity:
            selected.append(task)
            curr_cost += task.cost
            curr_val += task.value
            
    elapsed = time.perf_counter() - start_time
    return Knapsack01Result(
        max_value=curr_val,
        total_cost=curr_cost,
        capacity=capacity,
        selected_tasks=selected,
        dp_table=None,
        cell_updates=comparisons,
        execution_time_sec=elapsed,
        method="Greedy_Heuristic"
    )
