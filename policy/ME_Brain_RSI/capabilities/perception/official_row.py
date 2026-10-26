"""Policy-side geometry checks on RGB estimates, without scoring feedback.

The numeric task conditions are public. Passing this check is an RGB estimate,
not the official simulator verdict. Uprightness remains a current foreground
fit to the public upright CAD profile; it does not measure a hidden object axis.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

OFFICIAL_ACTIONS = 1050
ROW_Y_RANGE_M = .035
ADJACENT_X_MIN_M = .03
HOME_POSITION_EACH_AXIS_M = .15
HOME_ROTATION_DEG = 20.
HOME_RESERVE_ACTIONS = 30
MIN_REPAIR_ACTIONS = 90


def row_geometry(xy):
    xy = np.asarray(xy, float)
    if xy.shape != (5, 2) or not np.isfinite(xy).all():
        return {'valid': False, 'reason': 'five_finite_ranked_xy_required'}
    gaps = np.diff(xy[:, 0])
    spread = float(np.ptp(xy[:, 1]))
    return {'valid': bool(spread <= ROW_Y_RANGE_M and np.all(gaps > ADJACENT_X_MIN_M)),
            'rank_order': 'smallest_to_largest', 'y_range_m': spread,
            'adjacent_x_gaps_m': gaps.tolist(),
            'y_range_le_m': ROW_Y_RANGE_M, 'adjacent_x_strictly_gt_m': ADJACENT_X_MIN_M}


def unique_current_contours(objects):
    """Reject assigning one current foreground contour to multiple ranks."""
    if len(objects) != 5 or len({int(d.get('rank', -1)) for d in objects}) != 5:
        return False
    masks = []
    for d in objects:
        if not d.get('object_observation', {}).get('valid', False):
            return False
        contour = np.asarray(d.get('contour', []), np.int32).reshape(-1, 2)
        if len(contour) < 3:
            return False
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(contour), 1)
        if not mask.any():
            return False
        for other in masks:
            iou = np.count_nonzero(mask & other) / max(1, np.count_nonzero(mask | other))
            if iou > .7:
                return False
        masks.append(mask)
    return True


def home_geometry(state, initial):
    details = {}
    for arm in ('left', 'right'):
        pose = np.asarray(state[arm + '_ee_pose'], float)
        home = np.asarray(initial[arm + '_ee_pose'], float)
        translation = np.abs(pose[:3] - home[:3])
        a = Rotation.from_quat(np.roll(pose[3:], -1))
        b = Rotation.from_quat(np.roll(home[3:], -1))
        angle = float(np.degrees((a.inv() * b).magnitude()))
        details[arm] = {'position_error_each_axis_m': translation.tolist(),
                        'rotation_error_deg': angle,
                        'valid': bool(np.all(translation <= HOME_POSITION_EACH_AXIS_M)
                                      and angle <= HOME_ROTATION_DEG)}
    return {'valid': all(d['valid'] for d in details.values()), 'arms': details,
            'position_each_axis_le_m': HOME_POSITION_EACH_AXIS_M,
            'rotation_le_deg': HOME_ROTATION_DEG}


def repair_budget(step, relay=False):
    required = MIN_REPAIR_ACTIONS * (2 if relay else 1) + HOME_RESERVE_ACTIONS
    return {'allowed': OFFICIAL_ACTIONS - int(step) >= required,
            'remaining_actions': max(0, OFFICIAL_ACTIONS - int(step)),
            'estimated_minimum_repair_and_home_actions': required,
            'source': 'remaining_public_action_budget_not_per_object_retry_count'}


def select_repair(xy, goal_row, radii):
    """Choose one visibly misplaced object and a clear relative-row target.

    The planner retains the original safe spacing as a target, but no success
    decision depends on distance to that target. Current RGB positions are used
    for occupancy and rank-neighbor constraints.
    """
    xy = np.asarray(xy, float)
    goal = np.asarray(goal_row, float)
    radii = np.asarray(radii, float)
    diag = row_geometry(xy)
    if diag['valid']:
        return None
    median_y = float(np.median(xy[:, 1]))
    gaps = np.diff(xy[:, 0])
    bad = set()
    if np.ptp(xy[:, 1]) > ROW_Y_RANGE_M:
        bad.update(np.flatnonzero(np.abs(xy[:, 1] - median_y) > ROW_Y_RANGE_M / 2).tolist())
    for i in np.flatnonzero(gaps <= ADJACENT_X_MIN_M):
        bad.update((int(i), int(i + 1)))
    choices = []
    for rank in sorted(bad):
        # First prefer a y-only correction. It cannot disturb already valid x
        # order. Then consider the planned x and positions between neighbors.
        candidates = [float(xy[rank, 0]), float(goal[rank, 0])]
        lower = -.48 if rank == 0 else float(xy[rank - 1, 0] + .06)
        upper = .48 if rank == 4 else float(xy[rank + 1, 0] - .06)
        if lower <= upper:
            candidates.append(float(np.clip(goal[rank, 0], lower, upper)))
        for x in candidates:
            target = np.array([x, median_y])
            if abs(x) + radii[rank] + .015 >= .55 or not -.33 <= median_y <= -.015:
                continue
            other = [i for i in range(5) if i != rank]
            clearance = min(float(np.linalg.norm(target - xy[j]) - radii[rank] - radii[j]) for j in other)
            if clearance < .015:
                continue
            proposed = xy.copy()
            proposed[rank] = target
            old_bad = int(np.count_nonzero(gaps <= ADJACENT_X_MIN_M))
            new_bad = int(np.count_nonzero(np.diff(proposed[:, 0]) <= ADJACENT_X_MIN_M))
            old_spread = float(np.ptp(xy[:, 1]))
            new_spread = float(np.ptp(proposed[:, 1]))
            improvement = (old_bad - new_bad) * .1 + old_spread - new_spread
            if improvement <= 1e-6:
                continue
            choices.append((not row_geometry(proposed)['valid'], -improvement,
                            float(np.linalg.norm(target - xy[rank])), rank, target, clearance))
    if not choices:
        return None
    _, _, distance, rank, target, clearance = min(choices, key=lambda c: c[:4])
    return {'rank': rank, 'target_xy': target.tolist(), 'travel_m': distance,
            'clearance_m': clearance, 'reason': 'observed_official_relative_row_violation'}
