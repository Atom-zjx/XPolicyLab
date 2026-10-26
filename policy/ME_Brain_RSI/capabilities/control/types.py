"""Immutable, simulator-neutral values shared by both CAP compositions.

Quaternions use the RoboDojo/CuRobo ``(w, x, y, z)`` convention.  All container
fields are converted to tuples so a normalized state cannot retain a mutable
simulator object or observation mapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot, isfinite
from typing import TypeAlias

Vector3: TypeAlias = tuple[float, float, float]
QuaternionWXYZ: TypeAlias = tuple[float, float, float, float]
NamedVector: TypeAlias = tuple[str, tuple[float, ...]]
NamedPose: TypeAlias = tuple[str, "Pose"]

CANONICAL_BLOCK_IDENTITIES = ("block_0", "block_1", "block_2")


def _finite_tuple(value: object, *, name: str, length: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a numeric iterable")
    try:
        result = tuple(float(component) for component in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric iterable") from error
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    if not all(isfinite(component) for component in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _named_vectors(value: object, *, name: str) -> tuple[NamedVector, ...]:
    items: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        items = value.items()
    else:
        try:
            items = iter(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError(f"{name} must be a mapping or an iterable of pairs") from error

    normalized: list[NamedVector] = []
    seen: set[str] = set()
    try:
        for raw_key, raw_vector in items:
            key = str(raw_key)
            if not key:
                raise ValueError(f"{name} keys must not be empty")
            if key in seen:
                raise ValueError(f"duplicate {name} key: {key}")
            seen.add(key)
            normalized.append((key, _finite_tuple(raw_vector, name=f"{name}[{key!r}]")))
    except (TypeError, ValueError):
        raise
    except Exception as error:
        raise TypeError(f"{name} must contain (name, numeric iterable) pairs") from error
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _named_poses(value: object) -> tuple[NamedPose, ...]:
    items: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        items = value.items()
    else:
        try:
            items = iter(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError(
                "end_effector_poses must be a mapping or an iterable of pairs"
            ) from error

    normalized: list[NamedPose] = []
    seen: set[str] = set()
    for raw_key, pose in items:
        key = str(raw_key)
        if not key:
            raise ValueError("end_effector_poses keys must not be empty")
        if key in seen:
            raise ValueError(f"duplicate end_effector_poses key: {key}")
        if not isinstance(pose, Pose):
            raise TypeError(f"end_effector_poses[{key!r}] must be a Pose")
        seen.add(key)
        normalized.append((key, pose))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class Pose:
    """A world-frame rigid pose with a canonical unit quaternion."""

    position: Vector3
    quaternion_wxyz: QuaternionWXYZ

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position, name="position", length=3)
        quaternion = _finite_tuple(self.quaternion_wxyz, name="quaternion_wxyz", length=4)
        scale = max(abs(component) for component in quaternion)
        if scale == 0.0:
            raise ValueError("quaternion_wxyz must have non-zero magnitude")
        scaled = tuple(component / scale for component in quaternion)
        norm = hypot(*scaled)
        quaternion = tuple(component / norm for component in scaled)

        # q and -q encode the same rotation.  Canonicalizing the sign makes
        # privileged and perception estimates directly comparable as values.
        first_nonzero = next((component for component in quaternion if component != 0.0), 1.0)
        if first_nonzero < 0.0:
            quaternion = tuple(-component for component in quaternion)

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)

    @property
    def quaternion(self) -> QuaternionWXYZ:
        """Alias whose ordering remains explicitly documented as WXYZ."""

        return self.quaternion_wxyz

    def as_wxyz(self) -> tuple[float, ...]:
        return self.position + self.quaternion_wxyz


@dataclass(frozen=True, slots=True)
class ObjectState:
    """Normalized semantic object state; never a simulator object or handle."""

    identity: str
    pose: Pose
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("object identity must be a non-empty string")
        if not isinstance(self.pose, Pose):
            raise TypeError("object pose must be a Pose")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("object confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class ObjectEstimate:
    """A perception estimate before a canonical semantic identity is assigned."""

    pose: Pose
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose):
            raise TypeError("estimate pose must be a Pose")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("estimate confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Object estimates returned by a CAP perception backend."""

    blocks: tuple[ObjectEstimate, ObjectEstimate, ObjectEstimate]
    target: ObjectEstimate

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if len(blocks) != 3 or not all(isinstance(block, ObjectEstimate) for block in blocks):
            raise ValueError("perception result must contain exactly three ObjectEstimate blocks")
        if not isinstance(self.target, ObjectEstimate):
            raise TypeError("perception target must be an ObjectEstimate")
        object.__setattr__(self, "blocks", blocks)


@dataclass(frozen=True, slots=True)
class RobotState:
    """Official robot proprioception copied into immutable numeric values."""

    joint_states: tuple[NamedVector, ...] = ()
    end_effector_poses: tuple[NamedPose, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "joint_states", _named_vectors(self.joint_states, name="joint_states")
        )
        object.__setattr__(self, "end_effector_poses", _named_poses(self.end_effector_poses))

    def joint_state(self, name: str) -> tuple[float, ...]:
        for candidate, values in self.joint_states:
            if candidate == name:
                return values
        raise KeyError(name)

    def end_effector_pose(self, name: str) -> Pose:
        for candidate, pose in self.end_effector_poses:
            if candidate == name:
                return pose
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class CollisionGeometry:
    """Optional normalized primitive collision geometry."""

    identity: str
    kind: str
    dimensions: tuple[float, ...]
    pose: Pose

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("collision geometry identity must be a non-empty string")
        if self.kind not in {"box", "sphere", "capsule", "cylinder"}:
            raise ValueError(f"unsupported collision geometry kind: {self.kind!r}")
        dimensions = _finite_tuple(self.dimensions, name="collision geometry dimensions")
        if not dimensions or any(dimension <= 0.0 for dimension in dimensions):
            raise ValueError("collision geometry dimensions must be positive")
        if not isinstance(self.pose, Pose):
            raise TypeError("collision geometry pose must be a Pose")
        object.__setattr__(self, "dimensions", dimensions)


@dataclass(frozen=True, slots=True)
class SceneState:
    """Provider-independent state consumed by tasks, skills, and planning."""

    blocks: tuple[ObjectState, ObjectState, ObjectState]
    target: ObjectState
    robot: RobotState
    collision_geometry: tuple[CollisionGeometry, ...] = ()

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not all(isinstance(block, ObjectState) for block in blocks):
            raise TypeError("blocks must contain only ObjectState values")
        identities = tuple(block.identity for block in blocks)
        if len(blocks) != 3 or identities != CANONICAL_BLOCK_IDENTITIES:
            raise ValueError(
                "blocks must contain the canonical identities "
                f"{CANONICAL_BLOCK_IDENTITIES!r} in spatial order"
            )
        if not isinstance(self.target, ObjectState) or self.target.identity != "target":
            raise ValueError("target must be an ObjectState with identity 'target'")
        if not isinstance(self.robot, RobotState):
            raise TypeError("robot must be a RobotState")
        collision_geometry = tuple(self.collision_geometry)
        if not all(isinstance(geometry, CollisionGeometry) for geometry in collision_geometry):
            raise TypeError("collision_geometry must contain only CollisionGeometry values")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "collision_geometry", collision_geometry)

    @property
    def objects(self) -> tuple[ObjectState, ...]:
        return self.blocks + (self.target,)

    def object(self, identity: str) -> ObjectState:
        for candidate in self.objects:
            if candidate.identity == identity:
                return candidate
        raise KeyError(identity)
