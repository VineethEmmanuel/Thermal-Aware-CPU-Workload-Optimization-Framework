"""
LeetCode Algorithmic Pattern: Advanced Priority Queues & Heap Operations.

LeetCode Problem Mapping:
-------------------------
- Pattern: Priority Queue / Min-Heap & Dynamic Priority Updates
  * LeetCode 23: Merge k Sorted Lists (Optimal O(N log k) priority queue merging)
  * LeetCode 218: The Skyline Problem (Dynamic multiset/heap priority tracking)
  * LeetCode 295: Find Median from Data Stream (Max-Heap + Min-Heap balance)
  * LeetCode 743: Network Delay Time (Dijkstra shortest path with decrease-key)

Data Structures & Operations:
-----------------------------
1. Indexed Min-Heap:
   - Binary min-heap with hash map index tracking allowing O(log n) decrease-key.
   - Operations: insert O(log n), extract_min O(log n), decrease_key O(log n).
2. Binomial Heap & Fibonacci Heap Wrappers:
   - Connects DAA framework heaps to algorithmic competitive programming patterns.
   - Fibonacci Heap achieves O(1) amortized decrease-key for Dijkstra/Prims.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from framework.data_structures.binomial_heap import BinomialHeap
from framework.data_structures.fibonacci_heap import FibonacciHeap

T = TypeVar('T')


class IndexedMinHeap(Generic[T]):
    """
    Min-Heap Priority Queue supporting O(log n) decrease-key via position tracking.
    
    In standard heaps, decrease_key requires O(n) search to find the element.
    IndexedMinHeap maintains an index map {item_id: heap_index} to achieve O(1) lookup
    and O(log n) bubble-up.
    """
    def __init__(self):
        # Heap array storing (priority, item_id, payload)
        self.heap: List[Tuple[float, str, T]] = []
        # Mapping item_id -> current array index in self.heap
        self.pos_map: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.heap)

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    def _swap(self, i: int, j: int) -> None:
        """Swaps elements at indices i and j, updating their position mappings."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos_map[self.heap[i][1]] = i
        self.pos_map[self.heap[j][1]] = j

    def _sift_up(self, idx: int) -> None:
        """Restores min-heap property upwards (bubble up)."""
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] < self.heap[parent][0]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        """Restores min-heap property downwards (bubble down)."""
        n = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
                
            if smallest != idx:
                self._swap(idx, smallest)
                idx = smallest
            else:
                break

    def insert(self, item_id: str, priority: float, value: T) -> None:
        """
        Inserts (item_id, priority, value) in O(log n) time.
        """
        if item_id in self.pos_map:
            raise KeyError(f"Item '{item_id}' already present in heap.")
            
        idx = len(self.heap)
        self.heap.append((priority, item_id, value))
        self.pos_map[item_id] = idx
        self._sift_up(idx)

    def peek_min(self) -> Optional[Tuple[float, str, T]]:
        """Returns the minimum element in O(1) time without removing it."""
        if self.is_empty():
            return None
        return self.heap[0]

    def extract_min(self) -> Tuple[float, str, T]:
        """
        Extracts and returns the minimum element (priority, item_id, value) in O(log n) time.
        """
        if self.is_empty():
            raise IndexError("extract_min from empty IndexedMinHeap")
            
        min_elem = self.heap[0]
        last_elem = self.heap.pop()
        del self.pos_map[min_elem[1]]
        
        if self.heap:
            self.heap[0] = last_elem
            self.pos_map[last_elem[1]] = 0
            self._sift_down(0)
            
        return min_elem

    def decrease_key(self, item_id: str, new_priority: float) -> None:
        """
        Decreases the priority of item_id to new_priority in O(log n) time.
        """
        if item_id not in self.pos_map:
            raise KeyError(f"Item '{item_id}' not found in heap.")
            
        idx = self.pos_map[item_id]
        curr_priority, _, value = self.heap[idx]
        
        if new_priority > curr_priority:
            raise ValueError(f"New priority {new_priority} is higher than current {curr_priority}")
            
        self.heap[idx] = (new_priority, item_id, value)
        self._sift_up(idx)


def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    Direct LeetCode 23: Merge k Sorted Lists.
    
    Merges k sorted lists into one sorted list using a Min-Heap Priority Queue.
    
    Complexity:
        Time : O(N log k) where N is the total number of elements across all k lists.
        Space: O(k) for the heap holding at most one head element per list.
    """
    heap = IndexedMinHeap[Tuple[int, int]]()  # Stores (list_idx, element_idx)
    result: List[int] = []
    
    # Initialize heap with the first element of each non-empty list
    for list_idx, lst in enumerate(lists):
        if lst:
            heap.insert(f"L{list_idx}_0", float(lst[0]), (list_idx, 0))
            
    step = 0
    while not heap.is_empty():
        val, _, (list_idx, elem_idx) = heap.extract_min()
        result.append(int(val))
        
        # Advance in the list from which min was taken
        next_elem_idx = elem_idx + 1
        if next_elem_idx < len(lists[list_idx]):
            step += 1
            heap.insert(f"L{list_idx}_{step}", float(lists[list_idx][next_elem_idx]), (list_idx, next_elem_idx))
            
    return result
