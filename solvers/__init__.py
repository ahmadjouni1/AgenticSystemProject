"""Dimensional synthesis: find a mechanism's dimensions so its output point follows a target path.

- ``solvers/nested/``: our teammate's nested solver, copied unmodified (see its ATTRIBUTION.md).
- ``solvers/search.py``: how the replica runs it (a time limit, then a polish step).
- ``solvers/base.py``: the plug-in interface (``solve(target, constraints)``) and the
  subprocess runner with a timeout.
- ``solvers/serial_arm.py``: plug-in A, serial arms through the teammate's URDF adapter.
- ``solvers/planar.py``: plug-in B (experimental), a four-bar and a slider-crank, posed by
  the replica's own loop solver.
"""
