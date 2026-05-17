"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Register the optional UI extension only when omni.ui is available.
try:
    from .ui_extension_example import UsdrtExamplePythonExtension
except ModuleNotFoundError as exc:
    if exc.name != "omni.ui":
        raise
    UsdrtExamplePythonExtension = None

__all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData", "UsdrtExamplePythonExtension"]
