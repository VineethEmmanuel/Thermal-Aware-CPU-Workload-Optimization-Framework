"""
Binomial Heap Data Structure for Dynamic Job Priority Management.

A Binomial Heap is a collection of Binomial Trees satisfying:
1. Each binomial tree obeys the min-heap property (key(parent) <= key(child)).
2. For any non-negative integer k, there is at most one binomial tree of degree k.

Asymptotic Complexity & LeetCode Pattern Mapping:
--------------------------------------------------
- LeetCode Pattern: Priority Queue / Heap Operations (e.g., LeetCode 23: Merge k Sorted Lists, LeetCode 218: Skyline)
- Find-Min:      O(log n) [or O(1) with cached min pointer]
- Insert:        O(log n) worst-case, O(1) amortized
- Extract-Min:   O(log n)
- Decrease-Key:  O(log n)
- Union (Merge): O(log n)
- Delete:        O(log n)
"""

import math
from typing import Any, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')


class BinomialNode(Generic[T]):
    """Node in a Binomial Tree."""
    __slots__ = ('key', 'value', 'degree', 'parent', 'child', 'sibling')

    def __init__(self, key: float, value: T):
        self.key: float = key
        self.value: T = value
        self.degree: int = 0
        self.parent: Optional['BinomialNode[T]'] = None
        self.child: Optional['BinomialNode[T]'] = None
        self.sibling: Optional['BinomialNode[T]'] = None

    def __repr__(self) -> str:
        return f"BinomialNode(key={self.key}, deg={self.degree}, val={self.value})"


class BinomialHeap(Generic[T]):
    """
    Min-heap implementation of Binomial Heap with operational metric tracking.
    """
    def __init__(self):
        self.head: Optional[BinomialNode[T]] = None
        self._size: int = 0
        
        # Operational metric instrumentation for DAA comparative profiling
        self.op_metrics = {
            "inserts": 0,
            "extract_mins": 0,
            "decrease_keys": 0,
            "links": 0,
            "unions": 0,
            "comparisons": 0,
        }

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.head is None

    def _link(self, y: BinomialNode[T], z: BinomialNode[T]) -> None:
        """
        Links binomial tree B_{k-1} rooted at y as the leftmost child of B_{k-1} rooted at z.
        Resulting tree rooted at z has degree k.
        Requires key(z) <= key(y).
        """
        self.op_metrics["links"] += 1
        y.parent = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def _merge_root_lists(self, h1: Optional[BinomialNode[T]], h2: Optional[BinomialNode[T]]) -> Optional[BinomialNode[T]]:
        """
        Merges root lists of two heaps in non-decreasing order of degree.
        Equivalent to the merge step of MergeSort.
        """
        if not h1:
            return h2
        if not h2:
            return h1

        dummy = BinomialNode(0.0, None)  # type: ignore
        tail = dummy
        curr1 = h1
        curr2 = h2

        while curr1 and curr2:
            if curr1.degree <= curr2.degree:
                tail.sibling = curr1
                curr1 = curr1.sibling
            else:
                tail.sibling = curr2
                curr2 = curr2.sibling
            tail = tail.sibling

        tail.sibling = curr1 if curr1 else curr2
        return dummy.sibling

    def union(self, other: 'BinomialHeap[T]') -> None:
        """
        Unions self with another BinomialHeap in O(log n) time.
        Consolidates trees of identical degree.
        """
        self.op_metrics["unions"] += 1
        self._size += other._size

        new_head = self._merge_root_lists(self.head, other.head)
        self.head = None
        other.head = None
        other._size = 0

        if not new_head:
            return

        prev: Optional[BinomialNode[T]] = None
        curr: Optional[BinomialNode[T]] = new_head
        nxt: Optional[BinomialNode[T]] = curr.sibling

        while nxt:
            # Case 1 & 2: Degrees differ, or three consecutive roots have same degree
            if (curr.degree != nxt.degree) or (nxt.sibling and nxt.sibling.degree == curr.degree):
                prev = curr
                curr = nxt
            else:
                # Case 3 & 4: curr and nxt have same degree; link them
                self.op_metrics["comparisons"] += 1
                if curr.key <= nxt.key:
                    # curr becomes root, nxt becomes its child
                    curr.sibling = nxt.sibling
                    self._link(nxt, curr)
                else:
                    # nxt becomes root, curr becomes its child
                    if not prev:
                        new_head = nxt
                    else:
                        prev.sibling = nxt
                    self._link(curr, nxt)
                    curr = nxt
            nxt = curr.sibling

        self.head = new_head

    def insert(self, key: float, value: T) -> BinomialNode[T]:
        """
        Inserts a new key-value pair into the Binomial Heap in O(log n) worst-case time.
        Returns the created BinomialNode for subsequent decrease_key references.
        """
        self.op_metrics["inserts"] += 1
        new_node = BinomialNode(key, value)
        temp_heap = BinomialHeap[T]()
        temp_heap.head = new_node
        temp_heap._size = 1
        self.union(temp_heap)
        return new_node

    def get_min(self) -> Optional[BinomialNode[T]]:
        """
        Finds the minimum key in O(log n) by traversing root list.
        """
        if not self.head:
            return None
        min_node = self.head
        curr = self.head.sibling
        while curr:
            self.op_metrics["comparisons"] += 1
            if curr.key < min_node.key:
                min_node = curr
            curr = curr.sibling
        return min_node

    def extract_min(self) -> Tuple[float, T]:
        """
        Extracts and returns the minimum key-value pair in O(log n) time.
        """
        if not self.head:
            raise IndexError("extract_min called on empty BinomialHeap")

        self.op_metrics["extract_mins"] += 1

        # 1. Locate min root and its predecessor
        prev_min: Optional[BinomialNode[T]] = None
        min_node: BinomialNode[T] = self.head
        prev: Optional[BinomialNode[T]] = self.head
        curr: Optional[BinomialNode[T]] = self.head.sibling

        while curr:
            self.op_metrics["comparisons"] += 1
            if curr.key < min_node.key:
                min_node = curr
                prev_min = prev
            prev = curr
            curr = curr.sibling

        # 2. Remove min_node from root list
        if not prev_min:
            self.head = min_node.sibling
        else:
            prev_min.sibling = min_node.sibling

        # 3. Reverse min_node's children to form new root list with increasing degrees
        child_head: Optional[BinomialNode[T]] = None
        child = min_node.child
        num_children = 0
        while child:
            next_child = child.sibling
            child.sibling = child_head
            child.parent = None
            child_head = child
            child = next_child
            num_children += 1

        self._size -= 1

        # 4. Union the remaining heap with the reversed child heap
        if child_head:
            child_heap = BinomialHeap[T]()
            child_heap.head = child_head
            child_heap._size = (1 << min_node.degree) - 1
            # Adjust sizes before union
            self._size -= child_heap._size
            self.union(child_heap)

        return min_node.key, min_node.value

    def decrease_key(self, node: BinomialNode[T], new_key: float) -> None:
        """
        Decreases the key of a given node in O(log n) time by bubbling up.
        """
        self.op_metrics["decrease_keys"] += 1
        self.op_metrics["comparisons"] += 1
        if new_key > node.key:
            raise ValueError(f"New key {new_key} is greater than current key {node.key}")

        node.key = new_key
        curr = node
        parent = curr.parent

        while parent is not None:
            self.op_metrics["comparisons"] += 1
            if curr.key < parent.key:
                # Exchange key and value
                curr.key, parent.key = parent.key, curr.key
                curr.value, parent.value = parent.value, curr.value
                curr = parent
                parent = curr.parent
            else:
                break

    def delete(self, node: BinomialNode[T]) -> Tuple[float, T]:
        """Deletes an arbitrary node in O(log n) time."""
        self.decrease_key(node, -float('inf'))
        return self.extract_min()

    def get_degrees(self) -> List[int]:
        """Returns degrees of trees currently present in the root list."""
        degrees = []
        curr = self.head
        while curr:
            degrees.append(curr.degree)
            curr = curr.sibling
        return degrees
