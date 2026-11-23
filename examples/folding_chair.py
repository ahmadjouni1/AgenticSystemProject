"""Folding chair: two bent-tube frames crossing at a pivot, a seat, and a back slat.

Helpers shown: ``wire_from_points`` (U-shaped tube frames with filleted bends),
``PipeGeometry`` with ``rounded_rect_profile`` (a flat back slat), ``RoundedBoxGeometry``
(seat). Seen from the side, the frames cross like an X; folding turns the rear frame
about the pivot where the two tubes cross.
"""

from __future__ import annotations

from sdk import Assembly, PipeGeometry, RoundedBoxGeometry, rounded_rect_profile, wire_from_points

TUBE = 0.012
REAR_HALF_WIDTH = 0.22 - 2 * TUBE  # rear tubes run just inside the front ones
# Side view (y, z) of each frame's leg; the frames are mirrored to x = +-half width.
FRONT_LEG = ((0.20, 0.0), (-0.10, 0.85))  # front frame: foot to backrest top
REAR_LEG = ((-0.20, 0.0), (0.05, 0.55))  # rear frame: foot to seat height


SLAT_Y = FRONT_LEG[0][0] + (FRONT_LEG[1][0] - FRONT_LEG[0][0]) * 0.70 / FRONT_LEG[1][1]
"""y of the front leg at the slat height (z = 0.70), so the slat sits on the legs."""


def _pivot() -> tuple[float, float]:
    """Where the two legs cross in the side view."""
    (ay, az), (by, bz) = FRONT_LEG
    (cy, cz), (dy, dz) = REAR_LEG
    # Solve a + t (b - a) = c + s (d - c) for t.
    det = (by - ay) * (dz - cz) - (bz - az) * (dy - cy)
    t = ((cy - ay) * (dz - cz) - (cz - az) * (dy - cy)) / det
    return ay + t * (by - ay), az + t * (bz - az)


def _u_frame(half_width: float, leg: tuple[tuple[float, float], tuple[float, float]]):
    (y0, z0), (y1, z1) = leg
    return wire_from_points(
        [(-half_width, y0, z0), (-half_width, y1, z1), (half_width, y1, z1), (half_width, y0, z0)],
        radius=TUBE, cap_ends=True, corner_radius=0.03, corner_segments=8,
    )


def build() -> Assembly:
    chair = Assembly("folding_chair")
    py, pz = _pivot()

    front = chair.body("front_frame")
    front.add("tube", _u_frame(0.22, FRONT_LEG))
    front.add("back_slat", PipeGeometry(
        rounded_rect_profile(0.012, 0.08, 0.004), [(-0.22, SLAT_Y, 0.70), (0.22, SLAT_Y, 0.70)],
        cap=True,
    ))

    rear = chair.body("rear_frame")
    rear.add("tube", _u_frame(REAR_HALF_WIDTH, REAR_LEG))

    seat = chair.body("seat")
    seat.add("cushion", RoundedBoxGeometry((0.38, 0.34, 0.03), 0.008))

    chair.joint("fold", "revolute", front.at((0, py, pz)), rear.at((0, py, pz)),
                axis=(1, 0, 0), limits=(-0.55, 0.0))  # at -33 deg the seat meets the back slat
    chair.joint("seat_mount", "fixed", rear.at((0, 0.05, 0.55 + TUBE)), seat.at((0, 0.05, -0.015)))
    return chair


def main() -> None:
    chair = build()
    print(chair, "\npivot (y, z):", tuple(round(v, 4) for v in _pivot()))


if __name__ == "__main__":
    main()
