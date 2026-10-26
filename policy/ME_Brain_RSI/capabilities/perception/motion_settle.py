"""Conservative observation-based completion of a fully commanded motion tail.

No layout, reward, trace or episode-outcome input is accepted. This gate only
removes up to four repeated endpoint commands; it never changes a path target.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

MOTION_PHASES = frozenset(('approach', 'descend', 'wrist_center', 'lift',
                          'transport', 'place', 'retreat'))


def _pose(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.isfinite(result).all() or np.linalg.norm(result[3:]) < 1e-8:
        raise ValueError('Invalid observed pose')
    return result


def _angle(first, second):
    a = Rotation.from_quat(first[[4, 5, 6, 3]])
    b = Rotation.from_quat(second[[4, 5, 6, 3]])
    return float((a.inv()*b).magnitude())


def same_target(first, second):
    return np.linalg.norm(first[:3]-second[:3]) <= 1e-10 and _angle(first, second) <= 1e-10


def motion_complete(policy) -> bool:
    """Require two consecutive stable observations after endpoint commands.

    The caller resets ``_motion_settle_previous`` at every stage entry and
    evaluates this only after the existing attachment/identity checks. Original
    duration, endpoint timeout and all phase-transition verification remain.
    Park is excluded because it can concurrently preposition the other arm.
    """
    previous = getattr(policy, '_motion_settle_previous', None)
    policy._motion_settle_previous = None
    if policy.phase not in MOTION_PHASES or policy.arm not in ('left', 'right'):
        return False
    try:
        target = _pose(policy.target)
        current = _pose(policy.obs['state'][policy.arm+'_ee_pose'])
        command = _pose(policy.command[policy.arm+'_ee_pose'])
        record = dict(step=policy.step, phase=policy.phase, arm=policy.arm,
                      target=target.copy(), pose=current.copy(), command=command.copy())
        policy._motion_settle_previous = record
        if (previous is None or previous['step'] != policy.step-1
                or previous['phase'] != policy.phase or previous['arm'] != policy.arm
                or not max(2, policy.duration-4) <= policy.tick < policy.duration
                or not same_target(previous['target'], target)
                or not same_target(command, target)
                or not same_target(previous['command'], target)):
            return False
        old = previous['pose']
        return bool(max(np.linalg.norm(old[:3]-target[:3]), np.linalg.norm(current[:3]-target[:3])) <= .018
            and np.linalg.norm(current[:3]-old[:3]) <= .001
            and max(_angle(old, target), _angle(current, target)) <= .09
            and _angle(old, current) <= .01)
    except (ValueError, TypeError, KeyError, IndexError):
        policy._motion_settle_previous = None
        return False
