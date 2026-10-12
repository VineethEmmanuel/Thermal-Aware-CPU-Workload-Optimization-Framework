"""
LeetCode Algorithmic Pattern: 0/1 Knapsack & Fractional Knapsack Solvers.

LeetCode Problem Mapping:
-------------------------
- Pattern: 0/1 Knapsack (Subset Sum, Bounded Capacity Dynamic Programming)
  * LeetCode 416: Partition Equal Subset Sum (Decision version of 0/1 Knapsack)
  * LeetCode 474: Ones and Zeroes (2-Dimensional 0/1 Knapsack)
  * LeetCode 494: Target Sum (Transformable to 0/1 Subset Sum)
  * LeetCode 1049: Last Stone Weight II (0/1 Knapsack Partitioning)
- Pattern: Fractional Knapsack (Greedy Value Density Allocation)
  * Standard Fractional Knapsack (Continuous Linear Relaxation)

Complexity Comparison:
----------------------
- 0/1 Knapsack (DP):
  * Time Complexity : O(n * W) [Pseudo-polynomial; linear in W magnitude, exponential in bit-length log2(W)]
  * Space Complexity: O(n * W) for 2D tabulation table / O(W) for 1D rolling array optimization
- Fractional Knapsack (Greedy):
  * Time Complexity : O(n log n) with sorting by value density (v_i / w_i)
  * Space Complexity: O(n) auxiliary space
"""

from typing import Dict, List, Tuple


def knapsack_01_tabulation(weights: List[int], values: List[float], capacity: int) -> Tuple[float, List[int]]:
    """
    Solves 0/1 Knapsack using Dynamic Programming Tabulation (Bottom-Up).
    
    Equivalent to the core state-transition logic in LeetCode 416 and 474:
    dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])
    
    Args:
        weights: List of positive integer weights.
        values: List of item values.
        capacity: Maximum integer capacity budget.
        
    Returns:
        (max_value, selected_indices)
        
    Complexity:
        Time : O(n * W) - fills (n + 1) * (W + 1) table cells.
        Space: O(n * W) - stores table for full solution reconstruction.
    """
    if capacity <= 0 or not weights or not values:
        return 0.0, []
        
    n = len(weights)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    
    # State transitions
    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for c in range(1, capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
            else:
                dp[i][c] = dp[i - 1][c]
                
    max_value = dp[n][capacity]
    
    # Backtrack to identify selected indices
    selected_indices: List[int] = []
    curr_c = capacity
    for i in range(n, 0, -1):
        if dp[i][curr_c] != dp[i - 1][curr_c]:
            selected_indices.append(i - 1)
            curr_c -= weights[i - 1]
            
    selected_indices.reverse()
    return max_value, selected_indices


def knapsack_01_space_optimized(weights: List[int], values: List[float], capacity: int) -> float:
    """
    1D Space-Optimized 0/1 Knapsack DP.
    
    Equivalent to space-optimized LeetCode 416 pattern:
    dp[w] = max(dp[w], dp[w - weight] + value) evaluated right-to-left.
    
    Complexity:
        Time : O(n * W)
        Space: O(W)
    """
    if capacity <= 0 or not weights or not values:
        return 0.0
        
    dp = [0.0] * (capacity + 1)
    for w, v in zip(weights, values):
        # Traverse right-to-left to ensure items are used at most once
        for c in range(capacity, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                
    return dp[capacity]


def knapsack_01_memoized(weights: List[int], values: List[float], capacity: int) -> float:
    """
    Top-Down Recursion with Memoization for 0/1 Knapsack.
    
    Complexity:
        Time : O(n * W)
        Space: O(n * W) recursion stack and memo table.
    """
    memo: Dict[Tuple[int, int], float] = {}
    
    def solve(idx: int, rem_cap: int) -> float:
        if idx < 0 or rem_cap <= 0:
            return 0.0
        state = (idx, rem_cap)
        if state in memo:
            return memo[state]
            
        w = weights[idx]
        v = values[idx]
        
        # Choice 1: Exclude item
        res = solve(idx - 1, rem_cap)
        # Choice 2: Include item (if affordable)
        if w <= rem_cap:
            res = max(res, solve(idx - 1, rem_cap - w) + v)
            
        memo[state] = res
        return res

    return solve(len(weights) - 1, capacity)


def can_partition_subset_sum(nums: List[int]) -> bool:
    """
    Direct LeetCode 416: Partition Equal Subset Sum.
    
    Determines whether nums can be partitioned into two subsets with equal sum.
    Reduces directly to 0/1 Knapsack with capacity = sum(nums) // 2.
    
    Complexity:
        Time : O(n * S) where S = sum(nums) // 2.
        Space: O(S) using 1D rolling boolean array.
    """
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
        
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for c in range(target, num - 1, -1):
            dp[c] = dp[c] or dp[c - num]
            
    return dp[target]


def fractional_knapsack(weights: List[float], values: List[float], capacity: float) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Solves Fractional Knapsack using Greedy Ratio Sorting.
    
    Greedy Choice Property:
    Always choose as much of the item with highest value-to-cost ratio (v_i / w_i).
    
    Args:
        weights: List of weights (capacities).
        values: List of values.
        capacity: Continuous capacity headroom.
        
    Returns:
        (max_value, list of (item_index, allocated_fraction in [0.0, 1.0]))
        
    Complexity:
        Time : O(n log n) - dominated by sorting.
        Space: O(n) - auxiliary storage for ratios and result list.
    """
    if capacity <= 0 or not weights or not values:
        return 0.0, []
        
    # Build item tuples: (index, weight, value, ratio)
    items = []
    for i, (w, v) in enumerate(zip(weights, values)):
        ratio = (v / w) if w > 0 else float('inf')
        items.append((i, w, v, ratio))
        
    # Sort descending by ratio
    items.sort(key=lambda x: x[3], reverse=True)
    
    total_val = 0.0
    rem_cap = capacity
    allocations: List[Tuple[int, float]] = []
    
    for idx, w, v, _ in items:
        if rem_cap <= 0:
            allocations.append((idx, 0.0))
            continue
            
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
            allocations.append((idx, 1.0))
        else:
            frac = rem_cap / w
            total_val += v * frac
            rem_cap = 0.0
            allocations.append((idx, frac))
            
    return total_val, allocations
