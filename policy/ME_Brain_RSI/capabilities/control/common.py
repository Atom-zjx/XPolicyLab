"""Normalization and CAP-boundary helpers shared by state providers."""

from __future__ import annotations

import re
from collections.abc import Mapping

from rdj_master.core import (
    CANONICAL_BLOCK_IDENTITIES,
    ObjectEstimate,
    ObjectState,
    Pose,
    RobotState,
)


class CapObservationError(ValueError):
    """Raised when a CAP observation contains privileged or GT information."""


_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "scene_manager",
        "reward_manager",
        "_label_pose",
        "label_pose",
        "_label_info",
        "label_info",
        "layout_id",
        "seed",
        "env_seed",
        "depth",
        "approximate_depth",
        "distance_to_image_plane",
        "segmentation",
        "semantic_segmentation",
        "instance_segmentation",
        "ground_truth",
        "privileged",
        "task_env",
        "environment",
        "simulator",
        "reward",
        "task",
        "gt",
        "gt_handle",
        "ground_truth_handle",
        "object_pose",
        "target_pose",
        "cube0",
        "cube1",
        "cube2",
    }
)
_FORBIDDEN_TOKENS = frozenset(
    {"depth", "segmentation", "seed", "privileged", "simulator", "reward", "task", "gt"}
)
_ALLOWED_VISION_FIELDS = frozenset({"color", "intrinsic_matrix", "extrinsic_matrix", "shape"})
_ALLOWED_ROBOT_STATE_FIELDS = frozenset(
    {
        "arm_joint_state",
        "ee_joint_state",
        "ee_pose",
        "tcp_pose",
        "delta_ee_pose",
        "left_arm_joint_state",
        "left_ee_joint_state",
        "left_ee_pose",
        "left_tcp_pose",
        "left_delta_ee_pose",
        "right_arm_joint_state",
        "right_ee_joint_state",
        "right_ee_pose",
        "right_tcp_pose",
        "right_delta_ee_pose",
    }
)


def _normalized_key(key: object) -> str:
    value = str(key).strip().replace("-", "_").replace(" ", "_")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    return value.lower()


def _is_forbidden_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized in _FORBIDDEN_EXACT_KEYS:
        return True
    tokens = frozenset(part for part in normalized.split("_") if part)
    return bool(tokens & _FORBIDDEN_TOKENS)


def _guard_privileged_keys(value: object, *, path: str = "observation") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise CapObservationError(f"CAP observation contains forbidden field {path}.{key}")
            _guard_privileged_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            # Numeric tensors/lists end recursion cheaply; nested mappings are
            # still checked, including metadata attached to camera collections.
            if isinstance(child, (Mapping, tuple, list)):
                _guard_privileged_keys(child, path=f"{path}[{index}]")


def sanitize_cap_observation(official_observation: Mapping[str, object]) -> dict[str, object]:
    """Return the strict RGB/calibration/proprio subset exposed to CAP perception.

    The guard scans the complete input before constructing a new mapping, so a
    forbidden field cannot be hidden under an otherwise ignored metadata key.
    """

    if not isinstance(official_observation, Mapping):
        raise TypeError("official_observation must be a mapping")
    _guard_privileged_keys(official_observation)

    vision = official_observation.get("vision")
    if not isinstance(vision, Mapping) or not vision:
        raise CapObservationError("CAP observation requires a non-empty 'vision' mapping")

    sanitized_vision: dict[str, dict[str, object]] = {}
    has_color = False
    for raw_camera_name, raw_camera in vision.items():
        camera_name = str(raw_camera_name)
        if not camera_name or not isinstance(raw_camera, Mapping):
            raise CapObservationError("each CAP camera observation must be a mapping")
        unknown = {_normalized_key(key) for key in raw_camera} - _ALLOWED_VISION_FIELDS
        if unknown:
            raise CapObservationError(
                f"CAP camera {camera_name!r} contains unsupported fields: {sorted(unknown)!r}"
            )
        camera: dict[str, object] = {}
        for raw_key, value in raw_camera.items():
            key = _normalized_key(raw_key)
            camera[key] = value
            has_color = has_color or key == "color"
        sanitized_vision[camera_name] = camera
    if not has_color:
        raise CapObservationError("CAP observation requires at least one color image")

    state = official_observation.get("state")
    if not isinstance(state, Mapping):
        raise CapObservationError("CAP observation requires official robot 'state'")

    sanitized_state: dict[str, object] = {}
    for raw_key, value in state.items():
        key = _normalized_key(raw_key)
        if key not in _ALLOWED_ROBOT_STATE_FIELDS:
            raise CapObservationError(f"CAP robot state contains unsupported field: {raw_key!r}")
        sanitized_state[key] = value

    # A shallow copy is enough at this boundary: robot_state_from_observation
    # converts all proprioception to floats, while perception image/calibration
    # arrays remain opaque input data and are never stored in SceneState.
    return {"vision": sanitized_vision, "state": sanitized_state}


def coerce_pose(value: object, *, name: str = "pose") -> Pose:
    """Convert safe numeric pose representations to :class:`Pose`."""

    if isinstance(value, Pose):
        return value
    if isinstance(value, Mapping):
        if "pose" in value and not any(
            key in value
            for key in ("position", "translation", "quaternion_wxyz", "quaternion", "orientation")
        ):
            return coerce_pose(value["pose"], name=name)
        position = value.get("position", value.get("translation"))
        quaternion = value.get(
            "quaternion_wxyz",
            value.get("quaternion", value.get("orientation", value.get("rotation"))),
        )
        if position is None or quaternion is None:
            raise ValueError(f"{name} mapping requires position and quaternion")
        return Pose(position, quaternion)  # type: ignore[arg-type]

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a Pose, numeric length-7 iterable, or (position, quaternion)"
        )
    try:
        components = tuple(value)  # type: ignore[union-attr]
    except TypeError as error:
        raise TypeError(
            f"{name} must be a Pose, numeric length-7 iterable, or (position, quaternion)"
        ) from error
    if len(components) == 7:
        return Pose(components[:3], components[3:])  # type: ignore[arg-type]
    if len(components) == 2:
        return Pose(components[0], components[1])  # type: ignore[arg-type]
    raise ValueError(f"{name} must contain 7 values or a (position, quaternion) pair")


def coerce_estimate(value: object, *, name: str) -> ObjectEstimate:
    if isinstance(value, ObjectEstimate):
        return value
    if isinstance(value, ObjectState):
        return ObjectEstimate(value.pose, value.confidence)
    if isinstance(value, Mapping) and "confidence" in value:
        pose_value = value.get("pose", value)
        return ObjectEstimate(coerce_pose(pose_value, name=name), float(value["confidence"]))
    return ObjectEstimate(coerce_pose(value, name=name))


def canonicalize_blocks(blocks: object) -> tuple[ObjectState, ObjectState, ObjectState]:
    """Assign source-neutral identities by deterministic world XYZ spatial order."""

    if isinstance(blocks, Mapping):
        raw_blocks = tuple(blocks.values())
    else:
        try:
            raw_blocks = tuple(blocks)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("blocks must be a mapping or iterable of three estimates") from error
    if len(raw_blocks) != 3:
        raise ValueError(f"expected exactly three blocks, got {len(raw_blocks)}")
    estimates = [
        coerce_estimate(value, name=f"blocks[{index}]") for index, value in enumerate(raw_blocks)
    ]
    estimates.sort(key=lambda estimate: estimate.pose.position)
    return tuple(
        ObjectState(identity, estimate.pose, estimate.confidence)
        for identity, estimate in zip(CANONICAL_BLOCK_IDENTITIES, estimates, strict=True)
    )  # type: ignore[return-value]


def robot_state_from_observation(official_observation: Mapping[str, object]) -> RobotState:
    """Copy robot state only from the official observation ``state`` mapping."""

    if not isinstance(official_observation, Mapping):
        raise TypeError("official_observation must be a mapping")
    state = official_observation.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("official_observation requires a robot 'state' mapping")
    _guard_privileged_keys(state, path="observation.state")

    joint_states: dict[str, object] = {}
    end_effector_poses: dict[str, Pose] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        if name == "ee_pose" or name.endswith(("_ee_pose", "end_effector_pose")):
            end_effector_poses[name] = coerce_pose(value, name=f"state.{name}")
        else:
            joint_states[name] = value
    return RobotState(joint_states=joint_states, end_effector_poses=end_effector_poses)
