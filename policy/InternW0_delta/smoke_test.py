"""One-forward real-weight smoke test for the packaged evaluation runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan22", type=Path, required=True)
    parser.add_argument("--rynnbrain", type=Path, required=True)
    parser.add_argument(
        "--robodojo-root",
        type=Path,
        help="Optional parent workspace used when this checkout is standalone",
    )
    args = parser.parse_args()

    # Reused environments can inject an unrelated editable XPolicyLab via
    # sitecustomize. Select the requested parent workspace for shared RoboDojo
    # metadata, then prepend this checkout's policy directory only.
    source_xpl_root = Path(__file__).resolve().parents[2]
    if args.robodojo_root is not None:
        source_xpl_root = args.robodojo_root.resolve() / "XPolicyLab"
    sys.path[:] = [str(source_xpl_root)] + [
        entry for entry in sys.path if entry != str(source_xpl_root)
    ]
    for module_name in list(sys.modules):
        if module_name == "XPolicyLab" or module_name.startswith("XPolicyLab."):
            del sys.modules[module_name]
    import XPolicyLab.policy

    local_policy_root = Path(__file__).resolve().parents[1]
    XPolicyLab.policy.__path__[:] = [str(local_policy_root)] + [
        entry for entry in XPolicyLab.policy.__path__ if entry != str(local_policy_root)
    ]
    from XPolicyLab.policy.InternW0_delta.model import Model

    model = Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "checkpoint_path": str(args.checkpoint.resolve()),
            "base_model_dir": str(args.wan22.resolve()),
            "vlm_model_path": str(args.rynnbrain.resolve()),
            "action_horizon": 32,
            "replan_steps": 10,
            "num_inference_steps": 10,
            "device": "cuda",
            "mixed_precision": "bf16",
        }
    )
    state = {
        "left_arm_joint_state": np.zeros(6, dtype=np.float32),
        "left_ee_joint_state": np.zeros(1, dtype=np.float32),
        "right_arm_joint_state": np.zeros(6, dtype=np.float32),
        "right_ee_joint_state": np.zeros(1, dtype=np.float32),
    }
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    model.update_obs(
        {
            "vision": {
                "cam_head": {"color": image},
                "cam_left_wrist": {"color": image},
                "cam_right_wrist": {"color": image},
            },
            "state": state,
            "instruction": "stack the bowls",
        }
    )
    actions = model.get_action()
    if len(actions) != 10:
        raise RuntimeError(f"Expected 10 executed actions, got {len(actions)}")
    print("REAL_WEIGHT_SMOKE_OK actions=10", flush=True)


if __name__ == "__main__":
    main()
