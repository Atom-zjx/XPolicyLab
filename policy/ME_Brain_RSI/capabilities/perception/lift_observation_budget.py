"""Keep the original lift observation budget until visual attachment is confirmed."""
from __future__ import annotations

from .motion_settle import motion_complete as _motion_complete


def motion_complete(policy) -> bool:
    """Suppress only early lift completion, never extend the original duration.

    Confirmation is the existing RGB tracker's decision. Open/close commands
    and measured arm arrival alone are not attachment evidence.
    """
    if policy.phase == 'lift':
        tracker = getattr(policy, 'tracker', None)
        if (tracker is None or not getattr(tracker, 'valid', False)
                or not getattr(tracker, 'confirmed', False)
                or getattr(policy, 'missing', None) != 0):
            policy._motion_settle_previous = None
            return False
    return _motion_complete(policy)
