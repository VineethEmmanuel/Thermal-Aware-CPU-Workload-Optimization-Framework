"""
Historical Telemetry Indexing: Red-Black Tree Data Structure.

A Red-Black Tree is a self-balancing binary search tree satisfying:
1. Every node is either RED or BLACK.
2. The root is BLACK.
3. Every leaf (NIL sentinel) is BLACK.
4. If a node is RED, both its children are BLACK (no two RED nodes in a row).
5. For each node, all simple paths to descendant leaves contain the same number of BLACK nodes.

Asymptotic Complexity:
-----------------------
- Search:       O(log n)
- Insert:       O(log n) (at most 2 rotations)
- Delete:       O(log n) (at most 3 rotations)
- Range Query:  O(log n + k) where k is the number of returned records
- Min / Max:    O(log n)
"""

from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')


class Color(Enum):
    RED = 0
    BLACK = 1


class RBNode(Generic[T]):
    """Node in a Red-Black Tree."""
    __slots__ = ('key', 'value', 'color', 'left', 'right', 'parent')

    def __init__(self, key: float, value: T, color: Color = Color.RED):
        self.key: float = key
        self.value: T = value
        self.color: Color = color
        self.left: 'RBNode[T]' = self
        self.right: 'RBNode[T]' = self
        self.parent: 'RBNode[T]' = self

    def __repr__(self) -> str:
        c = "R" if self.color == Color.RED else "B"
        return f"RBNode({self.key}: {c})"


class RedBlackTree(Generic[T]):
    """
    Self-Balancing Red-Black Tree for high-frequency telemetry log indexing.
    Supports exact lookups, insertions, deletions, and efficient range queries.
    """
    def __init__(self):
        # NIL Sentinel Node to simplify boundary conditions
        self.NIL: RBNode[T] = RBNode(key=-float('inf'), value=None, color=Color.BLACK)  # type: ignore
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.NIL.parent = self.NIL
        self.root: RBNode[T] = self.NIL
        self._size: int = 0
        
        # Operational metrics for DAA empirical evaluation
        self.op_metrics = {
            "rotations": 0,
            "inserts": 0,
            "deletes": 0,
            "searches": 0,
            "range_queries": 0,
            "nodes_visited_range": 0,
        }

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.root == self.NIL

    def _left_rotate(self, x: RBNode[T]) -> None:
        """Left rotates node x."""
        self.op_metrics["rotations"] += 1
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode[T]) -> None:
        """Right rotates node y."""
        self.op_metrics["rotations"] += 1
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent == self.NIL:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    def insert(self, key: float, value: T) -> RBNode[T]:
        """
        Inserts (key, value) into the Red-Black Tree in O(log n) time.
        Restores Red-Black invariants using recoloring and at most 2 rotations.
        """
        self.op_metrics["inserts"] += 1
        z = RBNode(key=key, value=value, color=Color.RED)
        z.left = self.NIL
        z.right = self.NIL
        
        y: RBNode[T] = self.NIL
        x: RBNode[T] = self.root

        while x != self.NIL:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right

        z.parent = y
        if y == self.NIL:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

        self._insert_fixup(z)
        self._size += 1
        return z

    def _insert_fixup(self, z: RBNode[T]) -> None:
        """Maintains Red-Black properties after insertion."""
        while z.parent.color == Color.RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # Uncle
                if y.color == Color.RED:
                    # Case 1: Uncle is RED -> Recolor
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        # Case 2: Uncle is BLACK & z is right child -> Left rotate
                        z = z.parent
                        self._left_rotate(z)
                    # Case 3: Uncle is BLACK & z is left child -> Right rotate & recolor
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._right_rotate(z.parent.parent)
            else:
                # Symmetric cases when parent is right child of grandparent
                y = z.parent.parent.left  # Uncle
                if y.color == Color.RED:
                    # Case 1
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        # Case 2
                        z = z.parent
                        self._right_rotate(z)
                    # Case 3
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._left_rotate(z.parent.parent)

        self.root.color = Color.BLACK

    def search(self, key: float) -> Optional[T]:
        """Searches for a record by exact key in O(log n) time."""
        self.op_metrics["searches"] += 1
        curr = self.root
        while curr != self.NIL:
            if abs(curr.key - key) < 1e-9:
                return curr.value
            elif key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def search_node(self, key: float) -> Optional[RBNode[T]]:
        """Finds node with exact key, or None if not found."""
        curr = self.root
        while curr != self.NIL:
            if abs(curr.key - key) < 1e-9:
                return curr
            elif key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def range_search(self, low_key: float, high_key: float) -> List[Tuple[float, T]]:
        """
        Retrieves all records with keys in [low_key, high_key] in sorted order.
        Prunes subtrees outside the interval to achieve O(log n + k) complexity.
        """
        self.op_metrics["range_queries"] += 1
        results: List[Tuple[float, T]] = []

        def _traverse(node: RBNode[T]) -> None:
            if node == self.NIL:
                return
            self.op_metrics["nodes_visited_range"] += 1
            # If low_key <= node.key, left subtree might have candidates
            if node.key >= low_key:
                _traverse(node.left)
            # If current node is within range, record it
            if low_key <= node.key <= high_key:
                results.append((node.key, node.value))
            # If high_key >= node.key, right subtree might have candidates
            if node.key <= high_key:
                _traverse(node.right)

        _traverse(self.root)
        return results

    def _transplant(self, u: RBNode[T], v: RBNode[T]) -> None:
        """Replaces subtree rooted at node u with subtree rooted at node v."""
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node: RBNode[T]) -> RBNode[T]:
        """Finds minimum key node in subtree rooted at node."""
        curr = node
        while curr.left != self.NIL:
            curr = curr.left
        return curr

    def delete(self, key: float) -> bool:
        """
        Deletes a record with given key.
        Maintains Red-Black invariants with at most 3 rotations.
        """
        z = self.search_node(key)
        if z is None or z == self.NIL:
            return False

        self.op_metrics["deletes"] += 1
        y = z
        y_orig_color = y.color

        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_orig_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_orig_color == Color.BLACK:
            self._delete_fixup(x)

        self._size -= 1
        return True

    def _delete_fixup(self, x: RBNode[T]) -> None:
        """Restores Red-Black invariants after node deletion."""
        while x != self.root and x.color == Color.BLACK:
            if x == x.parent.left:
                w = x.parent.right  # Sibling
                if w.color == Color.RED:
                    # Case 1: Sibling is RED
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._left_rotate(x.parent)
                    w = x.parent.right

                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    # Case 2: Sibling is BLACK and both children are BLACK
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right.color == Color.BLACK:
                        # Case 3: Sibling is BLACK, near child is RED, far child is BLACK
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._right_rotate(w)
                        w = x.parent.right
                    # Case 4: Sibling is BLACK, far child is RED
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                # Symmetric cases when x is right child
                w = x.parent.left
                if w.color == Color.RED:
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._right_rotate(x.parent)
                    w = x.parent.left

                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left.color == Color.BLACK:
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self._right_rotate(x.parent)
                    x = self.root

        x.color = Color.BLACK

    def validate_properties(self) -> Tuple[bool, str]:
        """
        Formally verifies all 5 Red-Black Tree invariants:
        1. Root is BLACK.
        2. Leaves (NIL) are BLACK.
        3. No consecutive RED nodes.
        4. Equal Black-Height along all paths to NIL leaves.
        5. BST ordering property.
        """
        if self.root == self.NIL:
            return True, "Empty tree is valid."

        # Invariant 2: Root must be BLACK
        if self.root.color != Color.BLACK:
            return False, "Violation: Root is not BLACK."

        # Invariant 3: Leaves are BLACK (self.NIL is BLACK by definition)
        if self.NIL.color != Color.BLACK:
            return False, "Violation: NIL leaf sentinel is not BLACK."

        black_height: Optional[int] = None

        def _check(node: RBNode[T], current_bh: int, min_val: float, max_val: float) -> Tuple[bool, str]:
            nonlocal black_height
            if node == self.NIL:
                if black_height is None:
                    black_height = current_bh
                elif current_bh != black_height:
                    return False, f"Violation: Black-height mismatch ({current_bh} != {black_height})."
                return True, ""

            # Invariant 5: BST ordering
            if not (min_val <= node.key <= max_val):
                return False, f"Violation: BST order violated at key {node.key} not in [{min_val}, {max_val}]."

            # Invariant 4: No consecutive RED nodes
            if node.color == Color.RED:
                if node.left.color == Color.RED or node.right.color == Color.RED:
                    return False, f"Violation: Consecutive RED nodes detected at node {node.key}."

            next_bh = current_bh + (1 if node.color == Color.BLACK else 0)

            ok_l, msg_l = _check(node.left, next_bh, min_val, node.key)
            if not ok_l:
                return False, msg_l

            ok_r, msg_r = _check(node.right, next_bh, node.key, max_val)
            if not ok_r:
                return False, msg_r

            return True, ""

        return _check(self.root, 0, -float('inf'), float('inf'))

    def height(self) -> int:
        """Returns height of tree."""
        def _h(node: RBNode[T]) -> int:
            if node == self.NIL:
                return 0
            return 1 + max(_h(node.left), _h(node.right))
        return _h(self.root)
