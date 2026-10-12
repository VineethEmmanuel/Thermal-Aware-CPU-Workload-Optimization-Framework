"""
Fibonacci Heap Data Structure for Dynamic Job Priority Management.

A Fibonacci Heap is a collection of min-heap-ordered trees utilizing lazy consolidation
and cascading cuts to achieve state-of-the-art amortized time bounds:

Asymptotic Complexity:
-----------------------
- Find-Min:      O(1) actual and amortized
- Insert:        O(1) actual and amortized
- Union (Merge): O(1) actual and amortized
- Decrease-Key:  O(1) AMORTIZED (O(log n) worst-case)
- Extract-Min:   O(log n) AMORTIZED (O(n) worst-case during consolidation)
- Delete:        O(log n) AMORTIZED

DAA Theoretical Significance:
-----------------------------
In scheduling workloads where task urgencies are adjusted repeatedly due to thermal
surges (frequent decrease_key operations), the Fibonacci Heap's O(1) amortized
decrease_key operation drastically outperforms the Binomial Heap's O(log n) bound.
"""

import math
from typing import Any, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')


class FibonacciNode(Generic[T]):
    """Node in a Fibonacci Heap."""
    __slots__ = ('key', 'value', 'degree', 'marked', 'parent', 'child', 'left', 'right')

    def __init__(self, key: float, value: T):
        self.key: float = key
        self.value: T = value
        self.degree: int = 0
        self.marked: bool = False
        self.parent: Optional['FibonacciNode[T]'] = None
        self.child: Optional['FibonacciNode[T]'] = None
        self.left: 'FibonacciNode[T]' = self
        self.right: 'FibonacciNode[T]' = self

    def __repr__(self) -> str:
        return f"FibonacciNode(key={self.key}, deg={self.degree}, marked={self.marked}, val={self.value})"


class FibonacciHeap(Generic[T]):
    """
    Fibonacci Heap implementation with operational metric tracking.
    """
    def __init__(self):
        self.min_node: Optional[FibonacciNode[T]] = None
        self.total_nodes: int = 0
        
        # Operational metrics for DAA empirical evaluation
        self.op_metrics = {
            "inserts": 0,
            "extract_mins": 0,
            "decrease_keys": 0,
            "cuts": 0,
            "cascading_cuts": 0,
            "links": 0,
            "consolidations": 0,
            "comparisons": 0,
        }

    def __len__(self) -> int:
        return self.total_nodes

    def is_empty(self) -> bool:
        return self.min_node is None

    def _add_to_root_list(self, node: FibonacciNode[T]) -> None:
        """Inserts a node into the circular doubly linked root list."""
        if self.min_node is None:
            self.min_node = node
            node.left = node
            node.right = node
        else:
            # Splice node between min_node and min_node.right
            node.right = self.min_node.right
            node.left = self.min_node
            self.min_node.right.left = node
            self.min_node.right = node
            self.op_metrics["comparisons"] += 1
            if node.key < self.min_node.key:
                self.min_node = node

    def _remove_from_list(self, node: FibonacciNode[T]) -> None:
        """Removes a node from its circular doubly linked sibling/root list."""
        node.left.right = node.right
        node.right.left = node.left

    def insert(self, key: float, value: T) -> FibonacciNode[T]:
        """
        Inserts a new element into the Fibonacci Heap in O(1) time.
        Returns the node reference for future decrease_key invocations.
        """
        self.op_metrics["inserts"] += 1
        node = FibonacciNode(key, value)
        self._add_to_root_list(node)
        self.total_nodes += 1
        return node

    def get_min(self) -> Optional[FibonacciNode[T]]:
        """Returns the minimum node in O(1) time."""
        return self.min_node

    def union(self, other: 'FibonacciHeap[T]') -> None:
        """
        Unions self with another Fibonacci Heap in O(1) time.
        Concatenates root lists and updates the min_node pointer.
        """
        if other.min_node is None:
            return
        if self.min_node is None:
            self.min_node = other.min_node
            self.total_nodes = other.total_nodes
            other.min_node = None
            other.total_nodes = 0
            return

        # Splice root lists together
        self_min_right = self.min_node.right
        other_min_left = other.min_node.left

        self.min_node.right = other.min_node
        other.min_node.left = self.min_node

        self_min_right.left = other_min_left
        other_min_left.right = self_min_right

        self.op_metrics["comparisons"] += 1
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node

        self.total_nodes += other.total_nodes
        other.min_node = None
        other.total_nodes = 0

    def extract_min(self) -> Tuple[float, T]:
        """
        Extracts and returns the minimum key-value pair.
        Amortized complexity O(log n), worst-case O(n).
        """
        z = self.min_node
        if z is None:
            raise IndexError("extract_min called on empty FibonacciHeap")

        self.op_metrics["extract_mins"] += 1

        # 1. Promote all children of z to the root list
        if z.child is not None:
            children: List[FibonacciNode[T]] = []
            curr = z.child
            while True:
                children.append(curr)
                curr = curr.right
                if curr == z.child:
                    break
            
            for child in children:
                child.parent = None
                self._add_to_root_list(child)

        # 2. Remove z from root list
        if z == z.right:
            self.min_node = None
        else:
            self.min_node = z.right
            self._remove_from_list(z)
            self._consolidate()

        self.total_nodes -= 1
        return z.key, z.value

    def _consolidate(self) -> None:
        """
        Consolidates trees of the root list so that no two roots have the same degree.
        Amortized cost O(log n).
        """
        self.op_metrics["consolidations"] += 1
        # Max degree bound: D(n) <= floor(log_phi(n))
        # phi = (1 + sqrt(5)) / 2 approx 1.618, log_phi(n) approx 1.4404 * log2(n)
        if self.total_nodes == 0:
            return
            
        max_deg = int(math.ceil(1.4404 * math.log2(max(self.total_nodes, 2)))) + 2
        a_table: List[Optional[FibonacciNode[T]]] = [None] * (max_deg + 1)

        # Collect current roots
        root_nodes: List[FibonacciNode[T]] = []
        curr = self.min_node
        if curr is not None:
            while True:
                root_nodes.append(curr)
                curr = curr.right
                if curr == self.min_node:
                    break

        for w in root_nodes:
            x = w
            d = x.degree
            while d < len(a_table) and a_table[d] is not None:
                y = a_table[d]
                if y == x:
                    break
                self.op_metrics["comparisons"] += 1
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                a_table[d] = None
                d += 1
                if d >= len(a_table):
                    a_table.extend([None] * (d - len(a_table) + 2))
            if d < len(a_table):
                a_table[d] = x

        # Reconstruct root list
        self.min_node = None
        for node in a_table:
            if node is not None:
                node.left = node
                node.right = node
                node.parent = None
                self._add_to_root_list(node)

    def _link(self, y: FibonacciNode[T], x: FibonacciNode[T]) -> None:
        """
        Links y as a child of x.
        y loses its root status and becomes child of x.
        """
        self.op_metrics["links"] += 1
        self._remove_from_list(y)
        y.parent = x
        y.marked = False

        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            y.right = x.child.right
            y.left = x.child
            x.child.right.left = y
            x.child.right = y

        x.degree += 1

    def decrease_key(self, x: FibonacciNode[T], new_key: float) -> None:
        """
        Decreases the key of node x to new_key.
        Achieves O(1) AMORTIZED time complexity via Cascading Cuts.
        """
        self.op_metrics["decrease_keys"] += 1
        self.op_metrics["comparisons"] += 1
        if new_key > x.key:
            raise ValueError(f"New key {new_key} is greater than current key {x.key}")

        x.key = new_key
        y = x.parent

        if y is not None and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)

        self.op_metrics["comparisons"] += 1
        if self.min_node is not None and x.key < self.min_node.key:
            self.min_node = x

    def _cut(self, x: FibonacciNode[T], y: FibonacciNode[T]) -> None:
        """Cuts x from child list of y and moves x to root list."""
        self.op_metrics["cuts"] += 1
        # Remove x from y.child list
        if x.right == x:
            y.child = None
        else:
            if y.child == x:
                y.child = x.right
            self._remove_from_list(x)

        y.degree -= 1
        self._add_to_root_list(x)
        x.parent = None
        x.marked = False

    def _cascading_cut(self, y: FibonacciNode[T]) -> None:
        """
        Recursively cuts marked ancestor nodes up to root list.
        Preserves degree invariants of Fibonacci trees.
        """
        self.op_metrics["cascading_cuts"] += 1
        z = y.parent
        if z is not None:
            if not y.marked:
                # Mark y to indicate it has now lost one child
                y.marked = True
            else:
                # y was already marked and lost a second child; cut it
                self._cut(y, z)
                self._cascading_cut(z)

    def delete(self, x: FibonacciNode[T]) -> Tuple[float, T]:
        """Deletes node x from the heap in amortized O(log n) time."""
        self.decrease_key(x, -float('inf'))
        return self.extract_min()
