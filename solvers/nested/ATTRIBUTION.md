# Attribution

`synthesis.py` and `README.md` in this folder are **unmodified copies** of our teammate's
standalone solver, "General nested kinematic dimensional synthesis":

- **Author:** Suhaib Abu-Raidah (teammate), as recorded in the project's git history.
- **Source:** `Previous work/Agentic_Systems_Project/Project_extra_tools/`, added in commit
  `03f4c6c` ("Open-loop mechanisms dimensions solver tool added", 2026-11-23).
- **Copied:** 2026-11-25. The SHA-256 of each copy equals the original's
  (`tests/test_solvers.py` checks that the copies stay unmodified).
- **Method:** the README's own references (Russo et al., RCIM 2021, and SciPy).

The source folder holds only these two files. The README mentions `examples/spatial_3r.json`
and a `tests/` folder with eight tests, but neither was ever committed. So
`examples/spatial_3r.json` and `examples/spatial_3r.urdf` here, and the solver tests in
`tests/test_solvers.py`, were **written for the replica**, following the README's
description. They are not the teammate's files.

Everything that adapts the solver to the replica lives outside this folder:
`solvers/base.py` (the plug-in interface and the subprocess runner), `solvers/serial_arm.py`
(plug-in A) and `solvers/planar.py` (plug-in B).
