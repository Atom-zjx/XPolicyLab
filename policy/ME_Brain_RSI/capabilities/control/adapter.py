"""Three-RGB and own-proprioception boundary for the nesting-doll controller."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import sys
import numpy as np
from .cameras import HeadCamera, use_camera

VERSION = 'lxm6wuec_nesting_v3_rgb_official_aligned'
CAMERAS = {'head':'cam_head', 'left_wrist':'cam_left_wrist', 'right_wrist':'cam_right_wrist'}
ROBOT_FIELDS = ('left_arm_joint_state','right_arm_joint_state',
                'left_ee_joint_state','right_ee_joint_state','left_ee_pose','right_ee_pose')


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json(v) for v in value]
    return value


def white_observation(observation):
    """Construct a fresh observation; do not forward an env or source mapping."""
    images = {}
    for name, key in CAMERAS.items():
        rgb = np.asarray(observation['images'][name])
        if rgb.dtype != np.uint8 or rgb.shape != (480,640,3):
            raise ValueError(f'Expected 640x480 uint8 RGB: {name}')
        images[key] = {'color':np.ascontiguousarray(rgb).copy()}
    state = {}
    for name in ROBOT_FIELDS:
        value = np.asarray(observation['robot'][name], dtype=float)
        size = 6 if '_arm_' in name else 7 if name.endswith('_pose') else 1
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid own robot state: {name}')
        state[name] = value.copy()
    instruction = observation.get('instruction', '')
    if not isinstance(instruction, str):
        raise ValueError('Instruction must be text')
    return {'env_idx':0, 'vision':images, 'state':state, 'instruction':instruction}


class Policy:
    def __init__(self, config, *, reader=None):
        # reader injection exists solely for deterministic CPU boundary checks;
        # the runtime create_policy factory never accepts it from configuration.
        self.config = copy.deepcopy(config)
        self.last_step = -1
        self.initialization_error = None
        self.model = None
        try:
            options = self.config.get('options', {})
            self.camera = HeadCamera(options.get('rgb_calibration', {}))
            if options.get('wrist_camera_model', {}).get('mode') != 'upstream_static_prior':
                raise ValueError('Explicit upstream_static_prior wrist model is required')
            if any(os.environ.get(k) for k in ('LXM6WUEC_OPERATION_PLAN','LXM6WUEC_OPERATION_PLAN_SHA256')):
                raise ValueError('External operation plans are disabled')
            assets = Path(options['assets_root']).expanduser().resolve(strict=True)
            trace = Path(options['trace_dir']).expanduser().resolve()
            trace.mkdir(parents=True, exist_ok=True)
            dino = assets/'assets/grounding-dino-tiny'
            sam3 = assets/'assets/sam3.pt'
            if reader is None:
                if not (dino/'config.json').is_file() or not sam3.is_file():
                    raise FileNotFoundError('Verified local DINO/SAM3 assets are missing')
                if not any((dino/n).is_file() for n in ('model.safetensors','pytorch_model.bin')):
                    raise FileNotFoundError('Verified local DINO weights are missing')
            vendor = Path(__file__).resolve().parent/'vendor/sam3_source'
            sys.path.insert(0, str(assets))
            sys.path.insert(0, str(vendor))
            # These exact local paths prohibit automatic Hub checkpoint fetches.
            os.environ['NESTING_DINO_MODEL'] = str(dino)
            os.environ['NESTING_SAM3_CHECKPOINT'] = str(sam3)
            os.environ['NESTING_TRACE_DIR'] = str(trace)
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.asset_root = str(assets)
            from .nesting_rgb.model import Model
            self.model = Model({'trace_dir':str(trace), 'robot_static':self.config['robot_static']}, reader=reader)
        except (ValueError, KeyError, FileNotFoundError) as error:
            self.initialization_error = f'Nesting RGB configuration: {error}'

    def step(self, observation):
        if self.initialization_error:
            return {'stop_for_feedback':self.initialization_error,
                    'diagnostics':{'version':VERSION,'stage':'configuration','oracle':False}}
        step = int(observation['step_index'])
        if step <= self.last_step:
            raise ValueError('Each policy call requires a new RGB/proprioception frame')
        clean = white_observation(observation)
        self.last_step = step
        if 0 in self.model.envs:
            self.model.envs[0].events = []
        with use_camera(self.camera):
            actions = self.model.infer([clean])
        episode = self.model.envs[0]
        diagnostics = {
            'version':VERSION, 'stage':episode.phase, 'rank':episode.rank,
            'pick_index':episode.pick_index, 'completed_placements':len(episode.placed),
            'upstream_done':bool(episode.done), 'upstream_failure':episode.failure_reason,
            'visual_unconfirmed':episode.unconfirmed_reason,
            'official_alignment':{'x_adjacent_strictly_gt_m':0.03,'y_range_le_m':0.035,
                                  'upright_target_deg':15.0,
                                  'upright_evidence':'current_RGB_public_upright_profile_proxy_not_measured_angle',
                                  'home_position_each_axis_le_m':0.15,'home_rotation_le_deg':20.0},
            'source':'three_rgb_and_robot_proprioception', 'oracle':False,
            'camera_sources':{'head':'rgb_fingertip_fk_full_projection',
                              'wrists':'upstream_public_static_prior',
                              'wrist_rgb_estimated':False,'wrist_rgb_validated':False},
            'gripper_state_semantics':'previous_normalized_command_not_measured_opening_or_contact',
            'perception_models':['local_grounding_dino_tiny','local_sam3'],
            'motion_backend':'upstream_numpy_scipy_differential_ik_public_urdf',
            'events':episode.events,
        }
        if episode.failure_reason is not None:
            return {'stop_for_feedback':'Lxm6wuec nesting gate: '+episode.failure_reason,
                    'diagnostics':_json(diagnostics)}
        if len(actions) != 1 or len(actions[0]) != 1:
            raise ValueError('Expected one joint action from the original controller')
        action = actions[0][0]
        expected = {'left_arm_joint_state','right_arm_joint_state','left_ee_joint_state','right_ee_joint_state'}
        if set(action) != expected:
            raise ValueError('Controller action is not the permitted joint/gripper action')
        for key, value in action.items():
            values = np.asarray(value, float)
            if values.shape != ((6,) if '_arm_' in key else (1,)) or not np.isfinite(values).all():
                raise ValueError('Nonfinite or malformed controller action')
        return {'action':_json(action), 'diagnostics':_json(diagnostics)}


def create_policy(config):
    return Policy(config)
