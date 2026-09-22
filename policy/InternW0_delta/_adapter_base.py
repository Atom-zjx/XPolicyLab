"""XPolicyLab-facing stateful policy adapter shared by the local model bridge."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def instruction(obs: dict[str, Any], fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    return str(value).strip() if value is not None and str(value).strip() else fallback


def rgb(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class StatefulWAMAdapter(ModelTemplate):
    """Single-environment WAM adapter; subclasses provide model loading."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "arx_x5")
        if self.action_type != "joint":
            raise ValueError("This checkpoint was trained for joint actions")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if list(self.robot_action_dim_info.get("arm_dim", [])) != [6, 6]:
            raise ValueError(f"Checkpoint requires dual ARX-X5 arms: {self.robot_action_dim_info}")
        if list(self.robot_action_dim_info.get("ee_dim", [])) != [1, 1]:
            raise ValueError(f"Checkpoint requires one gripper scalar per arm: {self.robot_action_dim_info}")

        self.default_instruction = str(
            self.model_cfg.get("default_instruction") or "follow the instruction"
        )
        self.last_obs: dict[str, Any] | None = None
        self.last_instruction = self.default_instruction
        self.allow_dummy_policy = as_bool(self.model_cfg.get("allow_dummy_policy", False))
        self.session = None
        self.runtime = None
        self.action_horizon = int(self.model_cfg.get("action_horizon") or 32)
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 10)
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                f"replan_steps must be in [1, {self.action_horizon}], got {self.replan_steps}"
            )
        if not self.allow_dummy_policy:
            self._load_real_policy()

    def _load_real_policy(self) -> None:
        raise NotImplementedError

    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        vision = obs["vision"]
        state = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Expected a finite 14D RoboDojo state, got {state.shape}")
        return {
            "observation": {
                "head_camera": {"rgb": rgb(vision["cam_head"]["color"], "cam_head")},
                "left_camera": {
                    "rgb": rgb(vision["cam_left_wrist"]["color"], "cam_left_wrist")
                },
                "right_camera": {
                    "rgb": rgb(vision["cam_right_wrist"]["color"], "cam_right_wrist")
                },
            },
            "joint_action": {"vector": state},
        }

    def update_obs(self, obs):
        adapted = self._adapt_obs(obs)
        self.last_obs = adapted
        self.last_instruction = instruction(obs, self.default_instruction)
        if self.session is not None and self.session.pending_model_actions:
            self.session.update_obs(adapted)

    def update_obs_batch(self, obs_list):
        del obs_list
        raise NotImplementedError("Use eval_batch=false for stateful WAM inference")

    def _dummy_actions(self) -> list[dict[str, np.ndarray]]:
        zeros = np.zeros((self.replan_steps, 14), dtype=np.float32)
        return unpack_robot_state(
            zeros, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("Call update_obs before get_action")
        if self.allow_dummy_policy:
            return self._dummy_actions()
        if self.session.pending_model_actions:
            raise RuntimeError("Previous actions have not all been acknowledged")
        packed = np.asarray(
            self.session.get_action(
                {"observation": self.last_obs, "instruction": self.last_instruction}
            ),
            dtype=np.float32,
        )
        if packed.ndim != 2 or packed.shape[1] != 14 or not np.isfinite(packed).all():
            raise ValueError(f"WAM returned an invalid action chunk: {packed.shape}")
        packed[:, 6] = np.clip(packed[:, 6], 0.0, 1.0)
        packed[:, 13] = np.clip(packed[:, 13], 0.0, 1.0)
        return unpack_robot_state(
            packed, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def get_action_batch(self, env_idx_list=None):
        del env_idx_list
        raise NotImplementedError("Batched stateful WAM inference is disabled")

    def get_timing_rollout(self):
        return {} if self.session is None else self.session.get_timing_rollout()

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self.session is not None:
            self.session.reset_model()
