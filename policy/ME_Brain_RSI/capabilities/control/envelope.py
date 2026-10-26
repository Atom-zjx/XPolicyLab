"""Exact wire contract for ``general_pickup``."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral
from typing import Any


class GeneralPickupEnvelopeError(ValueError):
    """The trusted bridge or remote model received an invalid value."""


ENVELOPE_KEYS = frozenset({"env_idx", "target_z", "plans", "robot"})
PLAN_KEYS = frozenset(
    {
        "arm",
        "pregrasp_joint_state",
        "contact_joint_state",
        "lift_warmup_joint_state",
        "lift_joint_state",
    }
)
ROBOT_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
)
ROBOT_KEY_SET = frozenset(ROBOT_KEYS)
ACTION_KEYS = ROBOT_KEY_SET


def _mapping(value: object, keys: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GeneralPickupEnvelopeError(f"{name} keys must be exactly {sorted(keys)!r}")
    return value


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise GeneralPickupEnvelopeError(f"{name} must be a non-negative integer")
    return int(value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise GeneralPickupEnvelopeError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeneralPickupEnvelopeError(f"{name} must be finite") from error
    if not isfinite(result):
        raise GeneralPickupEnvelopeError(f"{name} must be finite")
    return result


def _vector(value: object, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise GeneralPickupEnvelopeError(f"{name} must be a numeric sequence")
    try:
        items = list(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise GeneralPickupEnvelopeError(f"{name} must be a numeric sequence") from error
    if len(items) != size:
        raise GeneralPickupEnvelopeError(f"{name} must contain {size} values")
    return [_number(item, name) for item in items]


def _plan(value: object, name: str) -> dict[str, object]:
    plan = _mapping(value, PLAN_KEYS, name)
    arm = plan["arm"]
    if arm not in {"left", "right"}:
        raise GeneralPickupEnvelopeError(f"{name}.arm must be 'left' or 'right'")
    return {
        "arm": arm,
        "pregrasp_joint_state": _vector(
            plan["pregrasp_joint_state"], 6, f"{name}.pregrasp_joint_state"
        ),
        "contact_joint_state": _vector(
            plan["contact_joint_state"], 6, f"{name}.contact_joint_state"
        ),
        "lift_warmup_joint_state": _vector(
            plan["lift_warmup_joint_state"], 6, f"{name}.lift_warmup_joint_state"
        ),
        "lift_joint_state": _vector(plan["lift_joint_state"], 6, f"{name}.lift_joint_state"),
    }


def normalize_general_pickup_envelope(value: Mapping[str, object]) -> dict[str, Any]:
    root = _mapping(value, ENVELOPE_KEYS, "general_pickup envelope")
    robot = _mapping(root["robot"], ROBOT_KEY_SET, "robot")
    raw_plans = root["plans"]
    if isinstance(raw_plans, (str, bytes, bytearray, Mapping)):
        raise GeneralPickupEnvelopeError("plans must contain exactly two plans")
    try:
        plan_items = list(raw_plans)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise GeneralPickupEnvelopeError("plans must contain exactly two plans") from error
    if len(plan_items) != 2:
        raise GeneralPickupEnvelopeError("plans must contain exactly two plans")
    return {
        "env_idx": _index(root["env_idx"], "env_idx"),
        "target_z": _number(root["target_z"], "target_z"),
        "plans": [_plan(item, f"plans[{index}]") for index, item in enumerate(plan_items)],
        "robot": {
            "left_arm_joint_state": _vector(
                robot["left_arm_joint_state"], 6, "robot.left_arm_joint_state"
            ),
            "right_arm_joint_state": _vector(
                robot["right_arm_joint_state"], 6, "robot.right_arm_joint_state"
            ),
            "left_ee_joint_state": _vector(
                robot["left_ee_joint_state"], 1, "robot.left_ee_joint_state"
            ),
            "right_ee_joint_state": _vector(
                robot["right_ee_joint_state"], 1, "robot.right_ee_joint_state"
            ),
        },
    }


def validate_general_pickup_action(value: Mapping[str, object]) -> dict[str, list[float]]:
    action = _mapping(value, ACTION_KEYS, "general_pickup action")
    return {
        "left_arm_joint_state": _vector(
            action["left_arm_joint_state"], 6, "action.left_arm_joint_state"
        ),
        "right_arm_joint_state": _vector(
            action["right_arm_joint_state"], 6, "action.right_arm_joint_state"
        ),
        "left_ee_joint_state": _vector(
            action["left_ee_joint_state"], 1, "action.left_ee_joint_state"
        ),
        "right_ee_joint_state": _vector(
            action["right_ee_joint_state"], 1, "action.right_ee_joint_state"
        ),
    }


__all__ = [
    "ACTION_KEYS",
    "ENVELOPE_KEYS",
    "PLAN_KEYS",
    "ROBOT_KEYS",
    "GeneralPickupEnvelopeError",
    "normalize_general_pickup_envelope",
    "validate_general_pickup_action",
]
