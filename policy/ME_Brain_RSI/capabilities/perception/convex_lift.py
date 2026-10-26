"""Bounded silhouette completion after an already-selected wrist mask fails fitting."""
from __future__ import annotations

import cv2
import numpy as np


def fit_lift(geometry, mask: np.ndarray, ee, rank: int, xy, rise: float):
    """Keep every original acceptance; try one convex contour with identical gates.

    The caller retains the original mask for tracking. This helper never selects
    an ambiguous component, joins components, changes rank, or relaxes a gate.
    A newly accepted fit remains a hypothesis needing native qualification.
    """
    fitted, original = geometry.fit_lift(mask, ee, rank, xy, rise)
    if fitted is not None:
        return fitted, original
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fitted, original
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    if cv2.contourArea(hull) <= 0:
        return fitted, original
    completed = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(completed, hull, 1)
    if np.array_equal(completed > 0, mask > 0):
        return fitted, original
    proposed, diagnostic = geometry.fit_lift(completed, ee, rank, xy, rise)
    evidence = dict(used=proposed is not None, original_fit=original,
        contour_area=float(cv2.contourArea(contour)), hull_area=float(cv2.contourArea(hull)),
        original_pixels=int(np.count_nonzero(mask)), completed_pixels=int(completed.sum()),
        original_tracking_mask_retained=True, components_merged=False)
    if proposed is not None:
        return proposed, dict(diagnostic, convex_fallback=evidence)
    return fitted, dict(original, convex_fallback=dict(evidence, rejected_fit=diagnostic))
