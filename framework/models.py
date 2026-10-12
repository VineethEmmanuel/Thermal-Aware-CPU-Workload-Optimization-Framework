"""
Data models representing Tasks, Workload types, and Telemetry logs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class WorkloadType(Enum):
    COMPUTE_BOUND = "COMPUTE_BOUND"
    MATRIX_PIPELINE = "MATRIX_PIPELINE"
    IO_BOUND = "IO_BOUND"
    BACKGROUND_BATCH = "BACKGROUND_BATCH"


@dataclass
class Task:
    """
    Represents a discrete or divisible job in the CPU workload.
    
    Attributes:
        job_id: Unique identifier for the job.
        name: Descriptive job name (e.g., 'ResNet_ConvChain', 'RenderBatch').
        workload_type: Category of workload.
        value: Utility / priority score / SLA reward for completing this job.
        cost: Resource cost (thermal energy in Joules, execution cycles, or weight budget).
        matrix_dims: If MATRIX_PIPELINE, sequence of matrix dimensions [p0, p1, ..., pn].
        urgency: Priority metric for heap scheduling (lower value = higher urgency/priority).
        is_divisible: Whether this job allows fractional execution.
        allocated_fraction: Fraction [0.0, 1.0] allocated during fractional scheduling.
    """
    job_id: str
    name: str
    workload_type: WorkloadType
    value: float
    cost: float
    matrix_dims: Optional[List[int]] = None
    urgency: float = 0.0
    is_divisible: bool = False
    allocated_fraction: float = 0.0

    @property
    def value_density(self) -> float:
        """Utility per unit thermal cost (ratio v/w used in greedy allocation)."""
        if self.cost <= 0:
            return float('inf')
        return self.value / self.cost

    def __repr__(self) -> str:
        div_str = f", frac={self.allocated_fraction:.2f}" if self.is_divisible else ""
        m_str = f", dims={self.matrix_dims}" if self.matrix_dims else ""
        return (f"Task(id='{self.job_id}', name='{self.name}', "
                f"v={self.value:.1f}, w={self.cost:.1f}, ratio={self.value_density:.3f}"
                f"{div_str}{m_str})")


@dataclass(order=True)
class TelemetryRecord:
    """
    Historical telemetry sensor reading indexed in Red-Black Tree and B-Tree.
    Sorted primarily by timestamp.
    """
    timestamp: float
    core_id: int = field(compare=False)
    temperature_c: float = field(compare=False)
    power_watts: float = field(compare=False)
    frequency_ghz: float = field(compare=False)
    throttled: bool = field(compare=False)
    active_job_id: Optional[str] = field(default=None, compare=False)
    notes: str = field(default="", compare=False)

    def __repr__(self) -> str:
        th = " [THROTTLED]" if self.throttled else ""
        job = f" (Job: {self.active_job_id})" if self.active_job_id else ""
        return (f"Telemetry(t={self.timestamp:.2f}s, Core={self.core_id}, "
                f"T={self.temperature_c:.1f}°C, P={self.power_watts:.1f}W, "
                f"Freq={self.frequency_ghz:.2f}GHz{th}{job})")
