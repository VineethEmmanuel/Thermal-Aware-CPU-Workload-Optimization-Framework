"""
LeetCode Algorithmic Pattern: Matrix Chain Multiplication & Interval DP.

LeetCode Problem Mapping:
-------------------------
- Pattern: Interval / Range Dynamic Programming (DP on Subarrays [i, j])
  * LeetCode 312: Burst Balloons (Direct dual maximization interval DP)
  * LeetCode 1000: Minimum Cost to Merge Stones (Generalized k-way interval merge)
  * LeetCode 375: Guess Number Higher or Lower II (Minimax range DP)
  * LeetCode 1547: Minimum Cost to Cut a Stick (Optimal partition cut order)

Algorithmic Invariant:
----------------------
In Interval DP, subproblems are organized by interval length `l` from 2 up to `n`.
The state dp[i][j] represents the optimal cost to process the subchain from index i to j.
For every intermediate split point k in [i, j-1]:
    cost = dp[i][k] + dp[k+1][j] + cost_to_combine(i, k, j)
    dp[i][j] = min_{k} (cost)

Complexity:
-----------
- Time Complexity : O(n^3) - O(n^2) interval states, each evaluating O(n) split positions k.
- Space Complexity: O(n^2) - stores 2D memo/tabulation table.
"""

import sys
from typing import List, Tuple


def matrix_chain_multiplication(dims: List[int]) -> Tuple[int, str, int]:
    """
    Finds the optimal parenthesization of a chain of matrices to minimize
    total scalar multiplications. Includes an operation counter to measure computational load.
    
    Args:
        dims: Sequence of dimensions <p_0, p_1, ..., p_n> for n matrices.
              Matrix A_i has dimension dims[i-1] x dims[i].
              
    Returns:
        (optimal_scalar_cost, optimal_parenthesized_expression, total_subproblem_operations)
        
    Complexity:
        Time : O(n^3)
        Space: O(n^2)
    """
    if len(dims) < 2:
        return 0, "", 0
        
    n = len(dims) - 1
    if n == 1:
        return 0, "A1", 0
        
    # m[i][j] stores minimum multiplications for A_i ... A_j (1-indexed)
    m = [[0] * (n + 1) for _ in range(n + 1)]
    # s[i][j] stores the split index k
    s = [[0] * (n + 1) for _ in range(n + 1)]
    
    operations_count = 0
    
    # Iterate over chain length l from 2 to n
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):
                operations_count += 1
                # Transition: cost(A_i..A_k) + cost(A_{k+1}..A_j) + combining cost
                cost = m[i][k] + m[k + 1][j] + (dims[i - 1] * dims[k] * dims[j])
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
                    
    def build_string(i: int, j: int) -> str:
        if i == j:
            return f"A{i}"
        k = s[i][j]
        return f"({build_string(i, k)} x {build_string(k + 1, j)})"
        
    optimal_cost = m[1][n]
    parenthesization = build_string(1, n)
    return optimal_cost, parenthesization, operations_count


def burst_balloons(nums: List[int]) -> int:
    """
    Direct LeetCode 312: Burst Balloons.
    
    Demonstrates the canonical Interval DP pattern matching Matrix Chain Multiplication.
    Pads nums with 1 at both boundaries: [1] + nums + [1].
    
    dp[i][j] = max_{i < k < j} { dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j] }
    
    Complexity:
        Time : O(n^3)
        Space: O(n^2)
    """
    if not nums:
        return 0
        
    padded = [1] + nums + [1]
    n = len(padded)
    dp = [[0] * n for _ in range(n)]
    
    # Interval length from 3 (smallest subarray with at least 1 balloon inside) to n
    for length in range(3, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            for k in range(i + 1, j):
                coins = dp[i][k] + dp[k][j] + (padded[i] * padded[k] * padded[j])
                if coins > dp[i][j]:
                    dp[i][j] = coins
                    
    return dp[0][n - 1]
