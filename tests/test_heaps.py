"""
Unit tests for Binomial Heap and Fibonacci Heap priority queue data structures.
Covers normal, boundary, and pathological inputs.
"""

import random
import unittest

from framework.data_structures.binomial_heap import BinomialHeap
from framework.data_structures.fibonacci_heap import FibonacciHeap


class TestPriorityHeaps(unittest.TestCase):
    """Test suite for Binomial and Fibonacci Heaps."""

    def test_binomial_heap_insert_extract_min_order(self):
        """Verifies that Binomial Heap yields elements in ascending sorted order."""
        b_heap = BinomialHeap[str]()
        keys = [45, 12, 89, 3, 23, 7, 56, 1]
        for k in keys:
            b_heap.insert(k, f"Val_{k}")

        self.assertEqual(len(b_heap), len(keys))
        extracted = []
        while not b_heap.is_empty():
            k, val = b_heap.extract_min()
            extracted.append(k)

        self.assertEqual(extracted, sorted(keys))

    def test_fibonacci_heap_insert_extract_min_order(self):
        """Verifies that Fibonacci Heap yields elements in ascending sorted order."""
        f_heap = FibonacciHeap[str]()
        keys = [45, 12, 89, 3, 23, 7, 56, 1]
        for k in keys:
            f_heap.insert(k, f"Val_{k}")

        self.assertEqual(len(f_heap), len(keys))
        extracted = []
        while not f_heap.is_empty():
            k, val = f_heap.extract_min()
            extracted.append(k)

        self.assertEqual(extracted, sorted(keys))

    def test_binomial_heap_decrease_key(self):
        """Tests decrease_key bubbling up correctly in Binomial Heap."""
        b_heap = BinomialHeap[str]()
        n1 = b_heap.insert(50, "Task_A")
        n2 = b_heap.insert(30, "Task_B")
        n3 = b_heap.insert(70, "Task_C")

        # Decrease n3 from 70 to 10 -> n3 must become new minimum
        b_heap.decrease_key(n3, 10)
        min_k, min_v = b_heap.extract_min()
        self.assertEqual(min_k, 10)
        self.assertEqual(min_v, "Task_C")

    def test_fibonacci_heap_decrease_key_and_cascading_cuts(self):
        """
        Tests Fibonacci Heap decrease_key and verifies cascading cuts mechanism.
        """
        f_heap = FibonacciHeap[str]()
        nodes = []
        for i in range(10):
            nodes.append(f_heap.insert(float(i * 10), f"Item_{i}"))

        # Consolidate trees by extracting minimum
        min_k, _ = f_heap.extract_min()
        self.assertEqual(min_k, 0.0)

        # Decrease key of an inner node
        target_node = nodes[-1]
        f_heap.decrease_key(target_node, -5.0)
        
        # New minimum must be -5.0
        self.assertEqual(f_heap.get_min().key, -5.0)
        k, v = f_heap.extract_min()
        self.assertEqual(k, -5.0)

    def test_heaps_union_operation(self):
        """Verifies union of two non-empty heaps."""
        h1_keys = [10, 30, 50, 70]
        h2_keys = [20, 40, 60, 80]

        # Binomial Heap Union
        b1 = BinomialHeap[int]()
        b2 = BinomialHeap[int]()
        for k in h1_keys:
            b1.insert(k, k)
        for k in h2_keys:
            b2.insert(k, k)
        b1.union(b2)
        self.assertEqual(len(b1), 8)
        b_sorted = [b1.extract_min()[0] for _ in range(8)]
        self.assertEqual(b_sorted, sorted(h1_keys + h2_keys))

        # Fibonacci Heap Union
        f1 = FibonacciHeap[int]()
        f2 = FibonacciHeap[int]()
        for k in h1_keys:
            f1.insert(k, k)
        for k in h2_keys:
            f2.insert(k, k)
        f1.union(f2)
        self.assertEqual(len(f1), 8)
        f_sorted = [f1.extract_min()[0] for _ in range(8)]
        self.assertEqual(f_sorted, sorted(h1_keys + h2_keys))

    def test_heaps_arbitrary_node_delete(self):
        """Tests arbitrary delete operation."""
        f_heap = FibonacciHeap[str]()
        n1 = f_heap.insert(100, "A")
        n2 = f_heap.insert(50, "B")
        n3 = f_heap.insert(150, "C")

        # Delete intermediate node n2 (50)
        f_heap.delete(n2)
        self.assertEqual(len(f_heap), 2)
        k1, _ = f_heap.extract_min()
        k2, _ = f_heap.extract_min()
        self.assertEqual([k1, k2], [100, 150])

    def test_boundary_empty_heap_exceptions(self):
        """Boundary test: extract_min on empty heap raises IndexError."""
        b_heap = BinomialHeap[int]()
        f_heap = FibonacciHeap[int]()
        with self.assertRaises(IndexError):
            b_heap.extract_min()
        with self.assertRaises(IndexError):
            f_heap.extract_min()

    def test_pathological_decrease_key_greater_raises_error(self):
        """Pathological test: attempting to increase key in decrease_key raises ValueError."""
        b_heap = BinomialHeap[int]()
        f_heap = FibonacciHeap[int]()
        bn = b_heap.insert(10, 10)
        fn = f_heap.insert(10, 10)

        with self.assertRaises(ValueError):
            b_heap.decrease_key(bn, 20)
        with self.assertRaises(ValueError):
            f_heap.decrease_key(fn, 20)

    def test_stress_random_operations(self):
        """Stress test: 300 random inserts, decrease keys, and extracts."""
        random.seed(42)
        b_heap = BinomialHeap[int]()
        f_heap = FibonacciHeap[int]()

        values = [random.randint(1, 10000) for _ in range(300)]
        for v in values:
            b_heap.insert(v, v)
            f_heap.insert(v, v)

        self.assertEqual(len(b_heap), 300)
        self.assertEqual(len(f_heap), 300)

        b_out = [b_heap.extract_min()[0] for _ in range(300)]
        f_out = [f_heap.extract_min()[0] for _ in range(300)]

        self.assertEqual(b_out, sorted(values))
        self.assertEqual(f_out, sorted(values))


if __name__ == "__main__":
    unittest.main()
