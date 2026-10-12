"""
Thermal Modeling: Lumped-Element RC Heat Transfer & DVFS Dynamics.

Differential Equation:
----------------------
The instantaneous temperature T(t) of the CPU die is governed by Newton's Law of Cooling
coupled with Joule heating dissipation:
    C_th * (dT/dt) = P_total(t) - (T(t) - T_ambient) / R_th

Where:
- C_th: Thermal capacitance (Joules / °C) of the CPU silicon die and heat spreader.
- R_th: Thermal resistance (°C / Watt) of the thermal interface and heatsink.
- P_total(t): Total power dissipation = P_static(T) + P_dynamic(f, V).
- T_ambient: Surrounding chassis/ambient air temperature.

When T(t) exceeds T_throttle, the thermal governor triggers DVFS throttling,
scaling down CPU frequency and reducing computing capacity budget.
"""

from typing import Tuple

from framework.config import ThermalConfig
from framework.models import TelemetryRecord


class CPUThermalModel:
    """
    Physical thermal simulator modeling temperature evolution across CPU cores.
    """
    def __init__(self, core_id: int = 0, config: ThermalConfig = ThermalConfig()):
        self.core_id: int = core_id
        self.config: ThermalConfig = config
        self.current_temp: float = config.ambient_temp
        self.current_time: float = 0.0
        self.is_throttled: bool = False
        self.active_frequency_ghz: float = config.nominal_frequency_ghz

    def step(self, active_power_watts: float, dt: float, active_job_id: str = None) -> TelemetryRecord:
        """
        Advances the thermal simulation by dt seconds under the given active power load.
        Updates internal temperature using the discrete RC model.
        """
        # Determine frequency and throttling state
        if self.current_temp >= self.config.throttle_temp:
            self.is_throttled = True
            self.active_frequency_ghz = self.config.throttled_frequency_ghz
            # Throttling reduces power dissipation
            power_dissipated = min(active_power_watts * 0.55 + self.config.base_idle_power, self.config.tdp_power)
        else:
            # Hysteresis recovery
            if self.is_throttled and self.current_temp < (self.config.throttle_temp - 5.0):
                self.is_throttled = False
                self.active_frequency_ghz = self.config.nominal_frequency_ghz
            power_dissipated = active_power_watts + self.config.base_idle_power

        # Newton's Law of Cooling differential update:
        # dT = [ P - (T - T_amb) / R_th ] * (dt / C_th)
        heat_loss_watts = (self.current_temp - self.config.ambient_temp) / self.config.thermal_resistance
        net_heat_flow_watts = power_dissipated - heat_loss_watts
        delta_temp = (net_heat_flow_watts * dt) / self.config.thermal_capacitance
        
        self.current_temp += delta_temp
        # Die cannot drop below ambient
        if self.current_temp < self.config.ambient_temp:
            self.current_temp = self.config.ambient_temp
            
        self.current_time += dt

        return TelemetryRecord(
            timestamp=self.current_time,
            core_id=self.core_id,
            temperature_c=self.current_temp,
            power_watts=power_dissipated,
            frequency_ghz=self.active_frequency_ghz,
            throttled=self.is_throttled,
            active_job_id=active_job_id,
            notes="Thermal step complete"
        )

    def calculate_thermal_headroom(self) -> float:
        """
        Calculates remaining degrees before hitting the throttling threshold.
        """
        return max(0.0, self.config.throttle_temp - self.current_temp)

    def compute_safe_thermal_budget(self, window_sec: float) -> float:
        """
        Computes maximum allowable thermal energy (Joules) without violating T_throttle:
        Budget_E = C_th * Delta_T_headroom + (Delta_T_headroom / R_th) * window_sec
        """
        headroom = self.calculate_thermal_headroom()
        # Conservative budget to avoid throttling
        budget = (self.config.thermal_capacitance * headroom * 0.8) + (headroom / self.config.thermal_resistance) * window_sec * 0.5
        return max(50.0, budget)
