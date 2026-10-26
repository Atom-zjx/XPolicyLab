"""Numeric pose helpers shared by RGB motion adapters; no contact library reads."""
from __future__ import annotations
from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Any
import numpy as np
from rdj_master.tasks.general_pickup.envelope import GeneralPickupEnvelopeError


def _numbers(value: object, size: int, name: str) -> list[float]:
    raw = value
    for method_name in ("detach", "cpu"):
        method = getattr(raw, method_name, None)
        if callable(method):
            raw = method()
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    try:
        result = [float(item) for item in raw]  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError) as error:
        raise GeneralPickupEnvelopeError(f"{name} must be numeric") from error
    if len(result) != size or not all(isfinite(item) for item in result):
        raise GeneralPickupEnvelopeError(f"{name} must contain {size} finite values")
    return result

def _rotation(quaternion: Sequence[float]) -> np.ndarray:
    return _shared_rotation_from_quaternion(quaternion, error_type=GeneralPickupEnvelopeError)

def _quaternion(matrix: np.ndarray) -> list[float]:
    return _shared_quaternion_from_matrix(matrix)

def semantic_pose_to_ee(robot: Any, pose: Sequence[float]) -> list[float]:
    return _shared_grasp_pose_to_end_effector(robot, pose, rotation_fn=_rotation, quaternion_fn=_quaternion)

def _offset_local_x(pose: Sequence[float], distance: float) -> list[float]:
    result = np.asarray(pose[:3], dtype=float) + _rotation(pose[3:])[:, 0] * float(distance)
    return result.tolist() + [float(item) for item in pose[3:]]

def _compose_pose(parent: Sequence[float], local: Sequence[float]) -> list[float]:
    return _shared_compose_pose(parent, local, rotation_fn=_rotation, quaternion_fn=_quaternion)

from lxm6wuec_skill_library.pose_skills import quaternion_from_matrix as _shared_quaternion_from_matrix
from lxm6wuec_skill_library.pose_skills import rotation_from_quaternion as _shared_rotation_from_quaternion
from lxm6wuec_skill_library.pose_skills import grasp_pose_to_end_effector as _shared_grasp_pose_to_end_effector
from lxm6wuec_skill_library.pose_skills import compose_pose as _shared_compose_pose
