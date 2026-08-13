"""Direct-action visual Student artifacts for the PandaHand RMA-XY Teacher.

The package is deliberately separate from :mod:`rma_xy_models`: the Teacher
still consumes simulator-only cube XY and bilateral contact force, while this
Student has a three-input deployment interface only.
"""

from .artifacts import RMA_DIRECT_ACTION_STUDENT_DR_TASK
from .models import RMADirectActionVisualStudent

__all__ = ("RMA_DIRECT_ACTION_STUDENT_DR_TASK", "RMADirectActionVisualStudent")
