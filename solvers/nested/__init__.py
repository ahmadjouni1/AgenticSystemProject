"""Our teammate's nested dimensional synthesis solver, copied unmodified (see ATTRIBUTION.md).

``synthesis.py`` is the teammate's file; import it from here:

    from solvers.nested import NestedSynthesis, SerialURDF, Settings, Task
"""

from solvers.nested.synthesis import (InnerResult, KinematicModel, NestedSynthesis, SerialURDF,
                                      Settings, Task, load_problem)

__all__ = ["InnerResult", "KinematicModel", "NestedSynthesis", "SerialURDF", "Settings", "Task",
           "load_problem"]
