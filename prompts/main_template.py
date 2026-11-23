"""The object: replace this docstring with one line saying what it is and how it moves."""

import math  # noqa: F401

from sdk import Assembly, Box, Cylinder  # noqa: F401

model = Assembly("object")

# Bodies: model.body(name), then body.add(shape_name, geometry, xyz=..., rpy=...).

# Joints: model.joint(name, "revolute" | "prismatic" | "fixed", parent.at(...), child.at(...),
#                     axis=(x, y, z), limits=(lower, upper))  # limits must contain 0

object_model = model


def run_tests(ctx):
    """Optional: assert what the design intends (ctx.expect_contact, ctx.expect_gap, ...)."""
