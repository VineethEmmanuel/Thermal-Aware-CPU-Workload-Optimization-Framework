"""
Configuration constants and parameters for Thermal-Aware CPU Workload Optimization.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalConfig:
    """CPU Thermal and Power parameters modeling physical heat dissipation."""
    # Temperatures in degrees Celsius (°C)
    ambient_temp: float = 25.0
    throttle_temp: float = 85.0
    critical_temp: float = 100.0
    
    # Lumped-element thermal model constants
    # R_th: Thermal resistance to ambient (degrees C per Watt)
    thermal_resistance: float = 0.50
    # C_th: Thermal capacitance (Joules per degree C)
    thermal_capacitance: float = 120.0
    
    # Power specifications in Watts
    base_idle_power: float = 15.0
    tdp_power: float = 125.0
    max_power: float = 180.0
    
    # Frequency scaling (DVFS states in GHz)
    nominal_frequency_ghz: float = 3.8
    boost_frequency_ghz: float = 5.0
    throttled_frequency_ghz: float = 2.0
    
    # Time step for discrete telemetry simulation (seconds)
    time_step_sec: float = 0.5
    
    # Scheduling window duration (seconds)
    scheduling_window_sec: float = 10.0


@dataclass(frozen=True)
class SimulationConfig:
    """General workload simulator parameters."""
    num_cores: int = 4
    default_thermal_energy_budget: float = 650.0  # Joules or capacity units
    b_tree_min_degree: int = 3
    random_seed: int = 42
