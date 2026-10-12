"""
Benchmarking Harness: Accurate Timing and Memory Profiling for DAA Algorithms.
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple


@dataclass
class BenchmarkRecord:
    """Stores performance profiling metrics for an algorithmic run."""
    algorithm_name: str
    parameter_label: str
    parameter_val: Any
    elapsed_ms: float
    peak_memory_kb: float
    operations_count: int = 0
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Benchmark({self.algorithm_name}: {self.parameter_label}={self.parameter_val} "
                f"-> {self.elapsed_ms:.3f}ms, Mem: {self.peak_memory_kb:.2f}KB, "
                f"Ops: {self.operations_count})")


class BenchmarkHarness:
    """
    Measures CPU execution time and peak memory footprint using tracemalloc.
    """
    @staticmethod
    def measure(func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
        """
        Executes func(*args, **kwargs) and records:
        - Return value
        - Elapsed wall-clock time in milliseconds
        - Peak memory allocated during invocation in Kilobytes (KB)
        """
        tracemalloc.start()
        start_ns = time.perf_counter_ns()
        
        result = func(*args, **kwargs)
        
        end_ns = time.perf_counter_ns()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        peak_kb = peak_mem / 1024.0
        
        return result, elapsed_ms, peak_kb
