"""
Computation Reduction: Matrix Chain Multiplication (MCM) Optimization.

Problem Formulation:
--------------------
In thermal-constrained CPU processing, matrix-heavy pipelines (e.g., deep learning
inference layers, transformation shaders, DSP filters) execute chained matrix
multiplications: A_1 x A_2 x ... x A_n, where matrix A_i has dimensions p_{i-1} x p_i.

Because matrix multiplication is associative, the product matrix is identical regardless
of parenthesization, but the number of scalar multiplications (ALU compute cycles and
consequent thermal dissipation) varies drastically by orders of magnitude.

Dynamic Programming Recurrence:
-------------------------------
Let m[i, j] be the minimum scalar multiplications to compute A_i ... A_j:
    m[i, j] = 0                                                           if i == j
    m[i, j] = min_{i <= k < j} { m[i, k] + m[k+1, j] + p_{i-1} * p_k * p_j } if i < j

Split Table s[i, j]:
--------------------
Stores index k where optimal parenthesization splits A_i ... A_j.

Theoretical Complexity & LeetCode Pattern Mapping:
--------------------------------------------------
- LeetCode Pattern: Interval DP / Range DP (e.g., LeetCode 312: Burst Balloons, LeetCode 1000: Minimum Cost to Merge Stones)
- DP Tabulation: Time O(n^3), Space O(n^2)
- Brute Force / Exhaustive Search: Catalan number C_{n-1} = Omega(4^n / n^{3/2})
- Memoized Top-Down DP: Time O(n^3), Space O(n^2)
"""

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MatrixChainResult:
    """Encapsulates the solution to the Matrix Chain Multiplication optimization."""
    optimal_cost: int
    optimal_parenthesization: str
    naive_left_to_right_cost: int
    naive_right_to_left_cost: int
    operations_saved: int
    cost_reduction_ratio: float
    m_table: List[List[int]]
    s_table: List[List[int]]
    execution_time_sec: float
    estimated_joules_saved: float
    estimated_temp_reduction_c: float

    def __repr__(self) -> str:
        return (f"MatrixChainResult(opt_cost={self.optimal_cost:,}, "
                f"naive_cost={self.naive_left_to_right_cost:,}, "
                f"reduction={self.cost_reduction_ratio:.1f}x, "
                f"order='{self.optimal_parenthesization}', "
                f"time={self.execution_time_sec * 1000:.3f}ms)")


def matrix_chain_order(dims: List[int], joules_per_flop: float = 1e-7, thermal_capacitance: float = 120.0) -> MatrixChainResult:
    """
    Computes optimal matrix chain parenthesization using bottom-up DP tabulation.
    
    Args:
        dims: Sequence of dimensions <p_0, p_1, ..., p_n> for n matrices.
              Matrix A_i has dimensions dims[i-1] x dims[i], 1-indexed.
        joules_per_flop: Thermal energy dissipation factor per scalar operation.
        thermal_capacitance: Heat capacity C_th of the CPU die in Joules / °C.
        
    Returns:
        MatrixChainResult with optimal cost, parenthesized string, and thermal metrics.
    """
    start_time = time.perf_counter()
    if len(dims) < 2:
        raise ValueError("Matrix chain requires at least 2 dimensions for 1 matrix.")
        
    n = len(dims) - 1  # Number of matrices
    
    if n == 1:
        elapsed = time.perf_counter() - start_time
        return MatrixChainResult(
            optimal_cost=0,
            optimal_parenthesization="A1",
            naive_left_to_right_cost=0,
            naive_right_to_left_cost=0,
            operations_saved=0,
            cost_reduction_ratio=1.0,
            m_table=[[0]],
            s_table=[[0]],
            execution_time_sec=elapsed,
            estimated_joules_saved=0.0,
            estimated_temp_reduction_c=0.0
        )
        
    # 1-indexed tables for standard CLRS DAA convention: dimensions (n+1) x (n+1)
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Chain lengths l from 2 to n
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):
                # Recurrence evaluation: cost of left chain + right chain + cross multiplication
                cost = m[i][k] + m[k + 1][j] + (dims[i - 1] * dims[k] * dims[j])
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
                    
    optimal_cost = m[1][n]
    parenthesization = reconstruct_optimal_parenthesization(s, 1, n)
    
    # Calculate baseline costs without DAA optimization
    naive_ltr = compute_naive_left_to_right(dims)
    naive_rtl = compute_naive_right_to_left(dims)
    max_naive = max(naive_ltr, naive_rtl)
    
    ops_saved = max(0, max_naive - optimal_cost)
    reduction_ratio = (max_naive / optimal_cost) if optimal_cost > 0 else 1.0
    
    # Thermal impact estimation:
    # Energy saved = Ops saved * Joules / Op
    # Delta T = Energy saved / Thermal capacitance
    joules_saved = ops_saved * joules_per_flop
    temp_reduction = joules_saved / thermal_capacitance
    
    elapsed = time.perf_counter() - start_time
    return MatrixChainResult(
        optimal_cost=optimal_cost,
        optimal_parenthesization=parenthesization,
        naive_left_to_right_cost=naive_ltr,
        naive_right_to_left_cost=naive_rtl,
        operations_saved=ops_saved,
        cost_reduction_ratio=reduction_ratio,
        m_table=m,
        s_table=s,
        execution_time_sec=elapsed,
        estimated_joules_saved=joules_saved,
        estimated_temp_reduction_c=temp_reduction
    )


def reconstruct_optimal_parenthesization(s: List[List[int]], i: int, j: int) -> str:
    """
    Recursively builds human-readable parenthesization string from split table s.
    Example: '((A1 x A2) x (A3 x A4))'
    """
    if i == j:
        return f"A{i}"
    k = s[i][j]
    left_str = reconstruct_optimal_parenthesization(s, i, k)
    right_str = reconstruct_optimal_parenthesization(s, k + 1, j)
    return f"({left_str} x {right_str})"


def compute_naive_left_to_right(dims: List[int]) -> int:
    """
    Computes scalar multiplications for non-optimized left-to-right evaluation:
    (...((A1 x A2) x A3)... x An)
    """
    n = len(dims) - 1
    if n <= 1:
        return 0
    total_cost = 0
    cur_rows = dims[0]
    cur_cols = dims[1]
    total_cost += cur_rows * cur_cols * dims[2] if n >= 2 else 0
    cur_cols = dims[2] if n >= 2 else cur_cols
    
    for i in range(3, n + 1):
        total_cost += cur_rows * cur_cols * dims[i]
        cur_cols = dims[i]
        
    return total_cost


def compute_naive_right_to_left(dims: List[int]) -> int:
    """
    Computes scalar multiplications for non-optimized right-to-left evaluation:
    (A1 x (A2 x ...(An-1 x An)...))
    """
    n = len(dims) - 1
    if n <= 1:
        return 0
    total_cost = 0
    cur_rows = dims[n - 1]
    cur_cols = dims[n]
    
    for i in range(n - 2, -1, -1):
        total_cost += dims[i] * cur_rows * cur_cols
        cur_rows = dims[i]
        
    return total_cost


def compute_scalar_ops_cost(dims: List[int], parenthesized_tree) -> int:
    """Helper to verify any arbitrary tree evaluation cost."""
    # Used for verifying manual orders
    return 0


def matrix_chain_brute_force(dims: List[int], i: int = 1, j: Optional[int] = None) -> int:
    """
    Naive recursive exponential algorithm: Omega(4^n / n^{3/2}) time complexity.
    Used exclusively in benchmarking and complexity analysis to contrast with DP.
    """
    if j is None:
        j = len(dims) - 1
    if i == j:
        return 0
        
    min_cost = sys.maxsize
    for k in range(i, j):
        cost = (matrix_chain_brute_force(dims, i, k) + 
                matrix_chain_brute_force(dims, k + 1, j) + 
                (dims[i - 1] * dims[k] * dims[j]))
        if cost < min_cost:
            min_cost = cost
            
    return min_cost
