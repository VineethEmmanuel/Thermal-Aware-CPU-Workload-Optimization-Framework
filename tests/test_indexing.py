"""
Unit tests for Historical Telemetry Indexing data structures:
- Red-Black Tree (Self-Balancing Binary Search Tree)
- B-Tree (Balanced Multi-Way Search Tree)
"""

import random
import unittest

from framework.data_structures.b_tree import BTree
from framework.data_structures.red_black_tree import Color, RedBlackTree
from framework.models import TelemetryRecord


class TestTelemetryIndexing(unittest.TestCase):
    """Test suite for Red-Black Tree and B-Tree indexing engines."""

    def test_red_black_tree_invariants_random_insertions(self):
        """
        Inserts 150 random keys into Red-Black Tree and validates all 5 RB invariants:
        1. Root is BLACK.
        2. NIL leaves are BLACK.
        3. No consecutive RED nodes.
        4. Equal Black-Height along all paths.
        5. BST ordering.
        """
        random.seed(42)
        rbt = RedBlackTree[str]()
        keys = [random.uniform(0.0, 1000.0) for _ in range(150)]

        for k in keys:
            rbt.insert(k, f"Reading_{k:.2f}")

        self.assertEqual(len(rbt), 150)
        is_valid, msg = rbt.validate_properties()
        self.assertTrue(is_valid, f"Red-Black Tree invariant violation: {msg}")

    def test_red_black_tree_pathological_sorted_insertions(self):
        """
        Pathological test: inserts strictly sorted keys (which would degenerate
        an unbalanced BST into a linked list of height O(n)).
        Verifies Red-Black Tree balances to height O(log n).
        """
        rbt = RedBlackTree[int]()
        n = 100
        for i in range(n):
            rbt.insert(float(i), i)

        is_valid, msg = rbt.validate_properties()
        self.assertTrue(is_valid, msg)
        
        # Height must be at most 2 * log2(n + 1) approx 14
        h = rbt.height()
        self.assertLessEqual(h, 15)

    def test_red_black_tree_search_and_range_queries(self):
        """Verifies exact lookups and range queries [t_start, t_end]."""
        rbt = RedBlackTree[str]()
        keys = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        for k in keys:
            rbt.insert(k, f"Val_{k}")

        # Exact searches
        self.assertEqual(rbt.search(30.0), "Val_30.0")
        self.assertIsNone(rbt.search(99.0))

        # Range query [25.0, 55.0] -> should return 30.0, 40.0, 50.0
        range_res = rbt.range_search(25.0, 55.0)
        res_keys = [k for k, v in range_res]
        self.assertEqual(res_keys, [30.0, 40.0, 50.0])

        # Range query matching empty interval
        empty_res = rbt.range_search(100.0, 200.0)
        self.assertEqual(empty_res, [])

    def test_red_black_tree_deletion_with_invariant_preservation(self):
        """
        Tests deletion of leaves, single-child nodes, two-child nodes, and root,
        verifying that RB invariants hold after each deletion fixup.
        """
        rbt = RedBlackTree[int]()
        keys = [50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43, 68, 93]
        for k in keys:
            rbt.insert(float(k), k)

        # 1. Delete leaf node (6)
        self.assertTrue(rbt.delete(6.0))
        is_valid, msg = rbt.validate_properties()
        self.assertTrue(is_valid, f"After leaf deletion: {msg}")

        # 2. Delete node with 2 children (25)
        self.assertTrue(rbt.delete(25.0))
        is_valid, msg = rbt.validate_properties()
        self.assertTrue(is_valid, f"After 2-child deletion: {msg}")

        # 3. Delete root node (50)
        self.assertTrue(rbt.delete(50.0))
        is_valid, msg = rbt.validate_properties()
        self.assertTrue(is_valid, f"After root deletion: {msg}")

        # 4. Search deleted keys
        self.assertIsNone(rbt.search(6.0))
        self.assertIsNone(rbt.search(25.0))
        self.assertIsNone(rbt.search(50.0))

        # 5. Non-existent deletion returns False
        self.assertFalse(rbt.delete(999.0))

    def test_btree_insert_splits_and_search(self):
        """Tests B-Tree node splitting and multi-way search."""
        btree = BTree[str](t=3)
        keys = [10, 20, 5, 6, 12, 30, 7, 17, 25, 35, 40, 2, 4]
        for k in keys:
            btree.insert(float(k), f"Log_{k}")

        self.assertEqual(len(btree), len(keys))

        # Search existing
        for k in keys:
            self.assertEqual(btree.search(float(k)), f"Log_{k}")

        # Search non-existing
        self.assertIsNone(btree.search(999.0))

    def test_btree_range_search(self):
        """Verifies B-Tree range search returns sorted slice."""
        btree = BTree[int](t=3)
        for i in range(1, 101):
            btree.insert(float(i), i)

        range_res = btree.range_search(25.0, 35.0)
        res_vals = [v for k, v in range_res]
        expected = list(range(25, 36))
        self.assertEqual(res_vals, expected)

    def test_boundary_empty_structures(self):
        """Boundary test: searching empty RBT and B-Tree."""
        rbt = RedBlackTree[int]()
        btree = BTree[int](t=3)

        self.assertIsNone(rbt.search(10.0))
        self.assertEqual(rbt.range_search(0.0, 100.0), [])
        self.assertFalse(rbt.delete(10.0))

        self.assertIsNone(btree.search(10.0))
        self.assertEqual(btree.range_search(0.0, 100.0), [])


if __name__ == "__main__":
    unittest.main()
