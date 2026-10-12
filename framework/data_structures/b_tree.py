"""
Historical Telemetry Indexing: B-Tree Data Structure.

A B-Tree of minimum degree t >= 2 is a balanced multi-way search tree where:
1. Every node has at most 2t - 1 keys and at least t - 1 keys (except root).
2. Every internal node has one more child than its number of keys.
3. All leaf nodes appear at the exact same depth.
4. Keys within each node are sorted in ascending order.

Asymptotic Complexity:
-----------------------
- Search:       O(t * log_t n) or O(log n)
- Insert:       O(t * log_t n) with proactive child splitting
- Delete:       O(t * log_t n) with sibling borrowing / merging
- Range Query:  O(log_t n + k)
"""

from typing import Any, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')


class BTreeNode(Generic[T]):
    """Node in a B-Tree."""
    __slots__ = ('keys', 'values', 'children', 'is_leaf')

    def __init__(self, is_leaf: bool = True):
        self.is_leaf: bool = is_leaf
        self.keys: List[float] = []
        self.values: List[T] = []
        self.children: List['BTreeNode[T]'] = []

    def __repr__(self) -> str:
        return f"BTreeNode(keys={self.keys}, leaf={self.is_leaf})"


class BTree(Generic[T]):
    """
    B-Tree implementation designed for block-structured telemetry record storage.
    """
    def __init__(self, t: int = 3):
        if t < 2:
            raise ValueError(f"Minimum degree t must be >= 2, got {t}")
        self.t: int = t
        self.root: BTreeNode[T] = BTreeNode(is_leaf=True)
        self._size: int = 0
        
        # Operational metrics for DAA empirical evaluation
        self.op_metrics = {
            "splits": 0,
            "inserts": 0,
            "searches": 0,
            "range_queries": 0,
            "nodes_visited_range": 0,
        }

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return len(self.root.keys) == 0

    def search(self, key: float) -> Optional[T]:
        """Searches for key in O(t * log_t n) time."""
        self.op_metrics["searches"] += 1
        
        def _search_node(node: BTreeNode[T], k: float) -> Optional[T]:
            i = 0
            while i < len(node.keys) and k > node.keys[i]:
                i += 1
            if i < len(node.keys) and abs(node.keys[i] - k) < 1e-9:
                return node.values[i]
            if node.is_leaf:
                return None
            return _search_node(node.children[i], k)

        return _search_node(self.root, key)

    def insert(self, key: float, value: T) -> None:
        """
        Inserts (key, value) into the B-Tree.
        Splits root if full (2t - 1 keys) before descending.
        """
        self.op_metrics["inserts"] += 1
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            # Root is full: create new root and split old root
            new_root = BTreeNode[T](is_leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0, root)
            self.root = new_root
            self._insert_non_full(new_root, key, value)
        else:
            self._insert_non_full(root, key, value)
        self._size += 1

    def _split_child(self, parent: BTreeNode[T], index: int, full_child: BTreeNode[T]) -> None:
        """Splits full_child of parent at median index t - 1."""
        self.op_metrics["splits"] += 1
        t = self.t
        z = BTreeNode[T](is_leaf=full_child.is_leaf)

        # z receives rightmost t - 1 keys and values
        z.keys = full_child.keys[t:]
        z.values = full_child.values[t:]
        
        if not full_child.is_leaf:
            z.children = full_child.children[t:]

        median_key = full_child.keys[t - 1]
        median_val = full_child.values[t - 1]

        # Truncate full_child to leftmost t - 1 keys
        full_child.keys = full_child.keys[:t - 1]
        full_child.values = full_child.values[:t - 1]
        if not full_child.is_leaf:
            full_child.children = full_child.children[:t]

        # Insert median key and value into parent
        parent.children.insert(index + 1, z)
        parent.keys.insert(index, median_key)
        parent.values.insert(index, median_val)

    def _insert_non_full(self, node: BTreeNode[T], key: float, value: T) -> None:
        """Inserts key and value into a non-full node."""
        i = len(node.keys) - 1
        if node.is_leaf:
            # Shift keys to maintain sorted order
            node.keys.append(key)
            node.values.append(value)
            while i >= 0 and node.keys[i] > key:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            # Check if child is full
            if len(node.children[i].keys) == (2 * self.t - 1):
                self._split_child(node, i, node.children[i])
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def range_search(self, low_key: float, high_key: float) -> List[Tuple[float, T]]:
        """
        Retrieves all key-value pairs within [low_key, high_key] in sorted order.
        """
        self.op_metrics["range_queries"] += 1
        results: List[Tuple[float, T]] = []

        def _traverse(node: BTreeNode[T]) -> None:
            self.op_metrics["nodes_visited_range"] += 1
            n_keys = len(node.keys)

            for i in range(n_keys):
                # Child i contains elements < node.keys[i]
                if not node.is_leaf:
                    if (i == 0 or node.keys[i - 1] <= high_key) and (low_key <= node.keys[i]):
                        _traverse(node.children[i])

                # Process key i
                if low_key <= node.keys[i] <= high_key:
                    results.append((node.keys[i], node.values[i]))
                elif node.keys[i] > high_key:
                    return

            # Last child contains elements > node.keys[-1]
            if not node.is_leaf:
                if n_keys == 0 or node.keys[-1] <= high_key:
                    _traverse(node.children[n_keys])

        _traverse(self.root)
        return results

    def height(self) -> int:
        """Computes height of B-Tree."""
        h = 0
        curr = self.root
        while not curr.is_leaf:
            h += 1
            curr = curr.children[0]
        return h + 1
