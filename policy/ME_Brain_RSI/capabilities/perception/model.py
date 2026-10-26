"""registered RGB-bound shallow pick, table relay, observed attachment and row placement."""
from ..stage_programs import execute as _execute_memory
import os, time, json
from collections import deque
from .joint_control import JointControl
from .convex_lift import fit_lift as convex_fit_lift
from .lift_observation_budget import motion_complete
from .row_assignment import choose_row, temporal_consensus
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .perception import Reader
from .contours import TABLE
from .attachment import Attachment
from .wrist_attachment import WristAttachment
from .silhouette_attachment import SilhouetteAttachment
from .wrist_geometry import WristGeometry
from .temporal_mask import propagate
from .object_observer import ObjectObserver
from .temporal_attachment_persistence import TemporalAttachmentPersistence
from .upright_identity_lock import UprightIdentityLock
from .official_row import row_geometry, unique_current_contours, home_geometry, repair_budget, select_repair
KEYS = ('left_ee_pose', 'right_ee_pose', 'left_ee_joint_state', 'right_ee_joint_state')
TIP = 0.13257000232458113
QUAT = {'left': np.array([-0.61239, 0.353523, -0.61239, -0.353524]), 'right': np.array([-0.353523, 0.61239, -0.353524, -0.61239])}

def rotation(q):
    return Rotation.from_quat(np.roll(q, -1))

class EpisodePolicy:

    def __init__(self, model_cfg, reader=None):
        self.joint_control = JointControl(model_cfg['robot_static'])
        self.joint_observation = None
        self.reader = reader if reader is not None else Reader()
        self.object_observer = ObjectObserver(self.reader.head_size)
        self.geometry = WristGeometry()
        self.output = Path(model_cfg.get('trace_dir', os.environ['NESTING_TRACE_DIR']))
        self.output.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        if hasattr(self, 'trace_file'):
            self.trace_file.close()
        self.trace_io_seconds = 0.0
        self.step = 0
        self.obs = None
        self.home = None
        self.identity_lock = UprightIdentityLock()
        self.phase = 'initial'
        self.trace = self.output / f'episode_{time.time_ns()}.jsonl'
        self.done = False
        self.failure_reason = None
        self.unconfirmed_reason = None
        self.wait_counts = {}
        self.wait_context = None
        self.wait_tick = 0
        self.repair_mode = False
        self.row_observation_attempts = 0
        self.last_row_evidence = None
        self.events = []
        self.tracker = None
        self.temporal_attachment = None
        self.rank = 0
        self.relay = False
        self.missing = 0
        self.trace_file = self.trace.open('a', buffering=262144)
        self.home_joints = None
        self.repair_counts = {}
        self.placed = {}
        self.relay_xy = np.array([0.0, -0.3])
        self.preposition = None
        self.prepared_pick = None
        self.empty_release_qualified = False
        self.pick_identity = None
        self.last_qualified = {}
        self.pick_order = list(range(5))
        self.pick_index = 0
        self.goal_row = np.c_[np.arange(5) * 0.12 - 0.24, np.full(5, -0.12)]

    def prepare_case(self, case_meta=None):
        self.reset()

    def on_trial_end(self, result=None):
        pass

    def mark_natural_end(self):
        self.log({'phase': 'official_natural_end', 'trace_io_seconds': self.trace_io_seconds})
        self.trace_file.flush()

    def update_obs(self, obs):
        self.joint_observation = {arm: np.asarray(obs['state'][arm + '_arm_joint_state'], float) for arm in ('left', 'right')}
        self.obs = {'vision': {k: {'color': np.asarray(v['color'], np.uint8)} for k, v in obs['vision'].items()}, 'state': {k: np.asarray(obs['state'][k], float).tolist() for k in KEYS}}

    def log(self, data):
        self.events.append({'step': self.step, 'phase': self.phase, 'rank': self.rank, **data})
        t = time.perf_counter()
        self.trace_file.write(json.dumps({'wall_time': time.time(), 'step': self.step, 'phase': self.phase, 'rank': self.rank, 'pick_index': self.pick_index, 'completed_placements': len(self.placed), **data}, default=lambda x: np.asarray(x).tolist()) + '\n')
        if self.step % 25 == 0 or data.get('event') in ('stage_begin', 'stage_end', 'failure') or data.get('phase') == 'official_natural_end':
            self.trace_file.flush()
        self.trace_io_seconds += time.perf_counter() - t

    def rgb(self):
        return self.obs['vision']['cam_head']['color']

    def snapshot(self):
        for cam, v in self.obs['vision'].items():
            cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_{self.phase}_{cam}.png'), cv2.cvtColor(v['color'], cv2.COLOR_RGB2BGR))

    def read(self):
        t = time.perf_counter()
        p = self.reader.read(self.rgb())
        self.log({'event': 'perception', 'perception': p, 'state': self.obs['state'], 'perception_wall_seconds': time.perf_counter() - t})
        return p

    def fail(self, reason):
        # A failed visual gate is uncertainty, not an official task verdict.
        # Freeze measured joints rather than carrying an unverified object.
        prior_phase = self.phase
        self.unconfirmed_reason = str(reason)
        self.done = False
        self.failure_reason = None
        self.command = {k: list(v) for k, v in self.obs['state'].items()}
        self.wait_context = {'phase': prior_phase, 'reason': str(reason),
                             'rank': self.rank, 'identity_resume': getattr(self, 'identity_resume', 'pick')}
        self.wait_tick = 0
        self.phase = 'evidence_wait'
        self.log({'event': 'visual_unconfirmed_hold', 'reason': reason,
                  'resume_phase': prior_phase, 'policy_action': 'measured_joint_hold',
                  'official_verdict': 'not_available_to_policy'})
        self.snapshot()

    def evidence_wait(self):
        self.wait_tick += 1
        self.command = {k: list(v) for k, v in self.obs['state'].items()}
        context = self.wait_context
        key = (context['reason'], context['rank'], context['phase'])
        count = self.wait_counts.get(key, 0)
        # New current RGB frames only; no motion under missing held evidence.
        # Three expensive re-observations are bounded, then hold until the
        # unmodified official task reaches its natural 1050-action verdict.
        if self.wait_tick % 12 or count >= 3:
            if self.wait_tick % 30 == 0:
                self.log({'event': 'unconfirmed_wait', 'reason': context['reason'],
                          'reobservations': count, 'remaining_actions': max(0, 1050 - self.step)})
            return
        self.wait_counts[key] = count + 1
        self.log({'event': 'unconfirmed_fresh_rgb_retry', 'reason': context['reason'],
                  'attempt': count + 1, 'held_motion_allowed': False})
        reason = context['reason']
        if reason == 'rgb_five_contours_or_height_separation':
            self.phase = 'observe'
            self.observe_tick = 35
            self.unconfirmed_reason = None
        elif reason in ('rgb_stationary_attachment_unconfirmed', 'stationary_semantic_texture_missing',
                        'rgb_attachment_not_confirmed', 'lift_semantic_texture_missing'):
            self.phase = context['phase']
            self.advance()
            if self.phase != 'evidence_wait':
                self.unconfirmed_reason = None
        elif reason in ('rgb_object_identity_or_upright_unconfirmed', 'rgb_final_row_object_unconfirmed',
                        'rgb_final_row_geometry_unconfirmed', 'rgb_final_row_unique_objects_unconfirmed'):
            if context['phase'] == 'row_observe':
                self.phase = 'row_observe'
                self.row_observation_attempts = 0
                self.final_row()
            else:
                resume = context['identity_resume']
                xy = self.xy if resume == 'pick' else self.drop
                gates = None if resume == 'pick' else {'iou': .5, 'size': .35, 'displacement': .08, 'mode': 'released_or_park'}
                d = self.observe_target(xy, self.rank, gates, fresh_only=True)
                if d is not None:
                    self.unconfirmed_reason = None
                    if resume == 'pick':
                        self.prepared_pick = {'rank': self.rank, 'arm': self.arm, 'object': d,
                                              'observation_step': self.step,
                                              'prototype': self.reader.appearance_prototype(self.rgb(), d['contour'])}
                        self.begin_pick()
                    else:
                        self.complete_placement(d)
        # Other physical/control gates keep the measured-joint hold. Repeated
        # grasp motion without object identity would exceed the RGB evidence.

    def enter(self, phase, xyz, grip, duration, q=None):
        self._motion_settle_previous = None
        self.phase = phase
        self.start = np.asarray(self.obs['state'][self.arm + '_ee_pose'])
        self.target = np.r_[xyz, self.q if q is None else q]
        self.grip = grip
        self.duration = max(6, int(duration * getattr(self, 'duration_scale', 1.0)))
        self.tick = 0
        if phase not in ('close', 'release', 'reconfirm'):
            distance = float(np.linalg.norm(self.target[:3] - self.start[:3]))
            angle = float((rotation(self.start[3:]).inv() * rotation(self.target[3:])).magnitude())
            self.duration = max(self.duration, int(np.ceil(1.875 * distance / 0.03)) + 5, int(np.ceil(1.875 * angle / 0.18)) + 5)
        if phase == 'close':
            self.duration = 12
            self.close_samples = deque(maxlen=4)
        self.log({'event': 'stage_begin', 'target': self.target, 'grip': grip, 'observed': self.obs['state']})
        self.snapshot()

    def observe_target(self, xy, rank, gates=None, fresh_only=False):
        self.identity_pending = False
        gates = {'iou': 0.65, 'size': 0.24, 'displacement': 0.055, 'mode': 'precontact'} if gates is None else gates
        started = time.perf_counter()
        d, mask, diag = self.object_observer.observe(self.rgb(), xy, rank, gates)
        self.log({'event': 'object_local_observation', 'target_rank': rank, 'expected_xy': xy, 'diagnostics': diag, 'cpu_object_observer_seconds': time.perf_counter() - started})
        if d is not None:
            self.last_qualified[rank] = dict(d)
            cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_object_rank{rank}_mask.png'), mask * 255)
            return d
        perception = self.read()
        ds = perception.get('visible_dolls', perception['dolls'])
        candidates = []
        for raw in ds:
            if np.linalg.norm(np.asarray(raw['xy']) - xy) > 0.1:
                continue
            value, fit = self.object_observer.qualify(raw, rank, np.asarray(xy), gates)
            self.log({'event': 'object_semantic_qualification', 'target_rank': rank, 'diagnostics': fit})
            if value is not None:
                candidates.append(value)
        if not candidates:
            focused, diag = self.reader.focused_read(self.rgb(), xy, float(np.max(self.geometry.points[rank][:, 2])))
            self.log({'event': 'focused_rebind', 'diagnostics': diag})
            for raw in focused:
                value, fit = self.object_observer.qualify(raw, rank, np.asarray(xy), gates)
                self.log({'event': 'object_focused_qualification', 'target_rank': rank, 'diagnostics': fit})
                if value is not None:
                    candidates.append(value)
        if candidates:
            chosen = min(candidates, key=lambda d: np.linalg.norm(np.asarray(d['xy']) - xy))
            locked = self.identity_lock.accept(rank, chosen)
            self.log({'event': 'upright_identity_lock', 'target_rank': rank, 'streak': self.identity_lock.streak.get(rank, 0), 'locked': locked})
            self.identity_pending = not locked
            self.last_qualified[rank] = dict(chosen)
            return chosen if locked else None
        if self.phase == 'identity_acquire' or fresh_only:
            return None
        prior = self.last_qualified.get(rank)
        if prior is not None and np.linalg.norm(np.asarray(prior['xy']) - xy) <= 0.03:
            self.log({'event': 'bounded_identity_hold', 'target_rank': rank, 'prior_xy': prior['xy'], 'expected_xy': xy, 'bound_m': 0.03})
            return dict(prior)
        return None

    def bind(self, xy, gates=None):
        d = self.observe_target(xy, self.rank, gates, fresh_only=True)
        if d is None:
            # Give a current head-profile identity more than one frame rather
            # than treating the first local fit rejection as terminal failure.
            self.identity_pending = True
        return d

    def begin_home(self, evidence):
        self.rank = 5
        self.relay = False
        self.phase = 'return_home'
        self.home_tick = 0
        self.command = {k: list(v) for k, v in self.home.items()}
        self.tracker = None
        self.unconfirmed_reason = None if evidence.get('valid', False) else 'rgb_row_unconfirmed_remaining_action_budget'
        self.snapshot()
        self.log({'event': 'home_joint_return_begin', 'target_joints': self.home_joints,
                  'row_evidence': evidence, 'final_placement_targets_required': False})

    def final_row(self, probe=False):
        # Expected tracks locate current pixels only. They are not a required
        # final row, and cached positions never count as final evidence.
        self.phase = 'row_observe'
        self.command = {k: list(v) for k, v in self.obs['state'].items()}
        released = {'iou': .5, 'size': .35, 'displacement': .08, 'mode': 'released_or_park'}
        observed = []
        for rank in range(5):
            expected = np.asarray(self.placed.get(rank, self.ordered[rank])['xy'])
            d = self.observe_target(expected, rank, released, fresh_only=True)
            if d is None:
                self.log({'event': 'row_probe_unconfirmed', 'target_rank': rank, 'probe': probe,
                          'evidence_source': 'current_RGB_only', 'history_fallback_allowed': False})
                self.row_observation_attempts += 1
                if self.row_observation_attempts >= 3:
                    self.fail('rgb_final_row_object_unconfirmed')
                return False
            observed.append(d)
        if not unique_current_contours(observed):
            self.fail('rgb_final_row_unique_objects_unconfirmed')
            return False
        self.row_observation_attempts = 0
        xy = np.array([d['xy'] for d in observed])
        diag = row_geometry(xy)
        self.last_row_evidence = dict(diag, xy=xy.tolist(), current_rgb_step=self.step,
             unique_rank_contours=True, fixed_goal_distance_required=False,
             upright_target_deg=15.,
             upright_evidence='current_RGB_public_upright_profile_proxy_not_measured_angle')
        self.log({'event': 'global_row_observation', **self.last_row_evidence})
        for rank, d in enumerate(observed):
            self.ordered[rank] = d
            self.placed[rank] = d
        if diag['valid']:
            self.begin_home(self.last_row_evidence)
            return True
        radii = [float(np.max(np.linalg.norm(points[:, :2], axis=1))) for points in self.geometry.points]
        repair = select_repair(xy, self.goal_row, radii)
        if repair is None:
            self.fail('rgb_final_row_geometry_unconfirmed')
            return False
        rank = repair['rank']
        source_arm = 'left' if xy[rank, 0] < 0 else 'right'
        dest_arm = 'left' if repair['target_xy'][0] < -.04 else 'right' if repair['target_xy'][0] > .04 else source_arm
        budget = repair_budget(self.step, relay=source_arm != dest_arm)
        self.log({'event': 'relative_row_repair_plan', 'repair': repair, 'budget': budget,
                  'prior_repairs': self.repair_counts.get(rank, 0), 'per_rank_limit': None})
        if not budget['allowed']:
            # Park at home; keep the uncertainty explicit and allow official
            # evaluation to decide at the public action limit.
            evidence = dict(self.last_row_evidence, correction_deferred='remaining_action_budget')
            self.begin_home(evidence)
            return False
        self.rank = rank
        self.goal_row[rank] = repair['target_xy']
        self.repair_counts[rank] = self.repair_counts.get(rank, 0) + 1
        self.repair_mode = True
        self.relay = False
        self.prepare_release_handoff(observed[rank], source_arm)
        self.begin_pick()
        return False

    def next_pick(self):
        if self.repair_mode:
            if self.receiver != self.arm and not self.relay:
                return (self.rank, self.receiver, self.drop.copy())
            return None
        if self.receiver != self.arm and (not self.relay):
            return (self.rank, self.receiver, self.drop.copy())
        if self.pick_index == 4:
            return None
        rank = self.pick_order[self.pick_index + 1]
        xy = np.asarray(self.ordered[rank]['xy'])
        return (rank, 'left' if xy[0] < 0 else 'right', xy)

    def begin_observation_park(self):
        self.preposition = None
        self.prepared_pick = None
        following = self.next_pick()
        if following is not None and following[1] != self.arm:
            rank, arm, xy = following
            d = self.observe_target(xy, rank)
            if d is not None:
                q = QUAT[arm] / np.linalg.norm(QUAT[arm])
                height = float(np.max(self.geometry.points[rank][:, 2]))
                offset = rotation(q).apply([TIP, 0, 0])
                xyz = np.r_[d['xy'], TABLE + (0.65 if rank == 4 else 0.45) * height] - offset + [0, 0, 0.14]
                self.preposition = (arm, np.asarray(self.obs['state'][arm + '_ee_pose']), np.r_[xyz, q])
                self.prepared_pick = {'rank': rank, 'arm': arm, 'object': d, 'observation_step': self.step, 'prototype': self.reader.appearance_prototype(self.rgb(), d['contour'])}
        self.enter('park', self.observation_park[self.arm + '_ee_pose'][:3], 1.0, 18, self.q)
        if self.preposition is not None:
            arm, start, target = self.preposition
            distance = float(np.linalg.norm(target[:3] - start[:3]))
            angle = float((rotation(start[3:]).inv() * rotation(target[3:])).magnitude())
            self.duration = max(self.duration, int(np.ceil(1.875 * distance / 0.03)) + 5, int(np.ceil(1.875 * angle / 0.18)) + 5)
            self.log({'event': 'unladen_other_arm_preposition', 'arm': arm, 'target': target, 'source': self.prepared_pick})

    def prepare_release_handoff(self, placed, arm):
        self.prepared_pick = {'rank': self.rank, 'arm': arm, 'object': placed, 'observation_step': self.step, 'prototype': self.reader.appearance_prototype(self.rgb(), placed['contour']), 'release_handoff': True}
        self.log({'event': 'release_evidence_handoff', 'receiver': arm, 'object': placed, 'evidence_step': self.step})

    def complete_placement(self, placed):
        self.command[self.arm + '_ee_pose'] = list(self.obs['state'][self.arm + '_ee_pose'])
        self.command[self.arm + '_ee_joint_state'] = [1.0]
        self.log({'event': 'released_object_observed', 'object': placed, 'intended_xy': self.drop,
                  'fixed_goal_error_used_as_success_requirement': False})
        self.snapshot()
        if self.receiver != self.arm and not self.relay:
            self.relay = True
            self.relay_xy = np.array(placed['xy'])
            self.prepare_release_handoff(placed, self.receiver)
            self.begin_pick()
            return
        self.ordered[self.rank] = placed
        self.placed[self.rank] = placed
        self.relay = False
        # The intended drop is a motion plan, not an official success condition.
        # Finish the row, then inspect relative order/spacing/alignment in RGB.
        if self.repair_mode:
            self.final_row()
            return
        self.pick_index += 1
        if self.pick_index < 5:
            self.rank = self.pick_order[self.pick_index]
            self.begin_pick()
        else:
            self.final_row()

    def begin_pick(self):
        if not self.relay:
            self.source = self.ordered[self.rank]
            self.height = self.source.get('calibrated_height_m', self.source['height_m'])
            self.xy = np.asarray(self.source['xy'])
            self.final = self.goal_row[self.rank].copy()
            self.arm = 'left' if self.xy[0] < 0 else 'right'
            self.receiver = 'left' if self.final[0] < -0.04 else 'right' if self.final[0] > 0.04 else self.arm
        else:
            self.arm = self.receiver
            self.xy = self.relay_xy.copy()
        prepared = self.prepared_pick
        self.prepared_pick = None
        self.preposition = None
        used_preposition = prepared is not None and prepared['rank'] == self.rank and (prepared['arm'] == self.arm) and (not prepared.get('release_handoff') or prepared['observation_step'] == self.step)
        if used_preposition:
            d = prepared['object']
            self.log({'event': 'release_identity_history' if prepared.get('release_handoff') else 'preposition_identity_history', 'observed_at_step': prepared['observation_step'], 'object': d, 'next_check': 'current wrist geometry and lift attachment'})
        else:
            self.identity_resume = 'pick'
            d = self.bind(self.xy)
        if d is None:
            if self.identity_pending:
                self.identity_resume = 'pick'
                self.identity_attempts = 1
                self.phase = 'identity_acquire'
            return
        fitted, diag = self.reader.head_size.locate(d, self.rank)
        self.log({'event': 'precontact_head_xy', 'fit': diag})
        if fitted is not None:
            d = dict(d, xy=fitted.tolist())
        self.wrist_prototype = prepared['prototype'] if used_preposition else self.reader.appearance_prototype(self.rgb(), d['contour'])
        self.log({'event': 'appearance_history', 'prototype': self.wrist_prototype})
        self.xy = np.asarray(d['xy'])
        self.q = QUAT[self.arm]
        self.q = self.q / np.linalg.norm(self.q)
        self.offset = rotation(self.q).apply([TIP, 0, 0])
        self.grasp = np.r_[self.xy, TABLE + (0.65 if self.rank == 4 else 0.45) * self.height] - self.offset
        self.drop = self.final if self.relay or self.receiver == self.arm else np.array([0.0, -0.3])
        self.dest = self.grasp + np.r_[self.drop - self.xy, 0.01]
        self.tracker = None
        self.temporal_attachment = TemporalAttachmentPersistence(max_dropout=5, centroid_tol_m=0.012)
        self.missing = 0
        self.servo_done = False
        self.prior_wrist = None
        self.reconfirmations = 0
        self.pick_identity = {'rank': self.rank, 'xy': self.xy.tolist(), 'object': d}
        self.empty_release_qualified = False
        self.enter('approach', self.grasp + [0, 0, 0.14], 1.0, 18)

    def empty_clearance(self):
        if self.done or self.phase not in ('retreat', 'park') or self.tracker is not None or (self.grip < 0.99):
            return None
        floor = TABLE + max((float(np.max(points[:, 2])) for points in self.geometry.points)) + 0.035
        ee = np.asarray(self.obs['state'][self.arm + '_ee_pose'])
        tip = ee[:3] + rotation(ee[3:]).apply([TIP, 0, 0])
        if tip[2] < floor:
            return None
        if not self.empty_release_qualified and self.step % 4 == 0:
            started = time.perf_counter()
            d, mask, diag = self.object_observer.observe(self.rgb(), self.drop, self.rank, {'iou': 0.5, 'size': 0.35, 'displacement': 0.08, 'mode': 'released_or_park'})
            self.log({'event': 'empty_release_qualification', 'diagnostics': diag, 'cpu_object_observer_seconds': time.perf_counter() - started, 'tip_xyz': tip, 'required_tip_floor': floor})
            if d is not None:
                self.empty_release_qualified = True
                self.log({'event': 'released_empty_orientation_available', 'object': d, 'tip_floor': floor})
        return {self.arm: floor} if self.empty_release_qualified else None

    def wrist_tracker(self, rgb, mask, ee):
        tracker = WristAttachment(rgb, mask, ee)
        if tracker.valid:
            return tracker
        silhouette = SilhouetteAttachment(rgb, mask, ee, self.wrist_prototype, self.geometry, self.rank, self.xy)
        self.log({'event': 'texture_independent_initialization', 'klt_corners': tracker.initial_count, 'silhouette_valid': silhouette.valid, 'support_pixels': int(silhouette.mask.sum()), 'confirmed': False})
        return silhouette

    def attachment_support(self):
        if getattr(self.tracker, 'silhouette', False):
            return {'support_pixels': int(self.tracker.mask.sum()), 'tracker': 'current_rgb_silhouette'}
        return {'features': len(self.tracker.points), 'tracker': 'KLT'}

    def advance(self):
        p = self.phase
        if p == 'approach':
            self.enter('descend', self.grasp, 1.0, 31)
        elif p in ('descend', 'wrist_center'):
            wrist_rgb = self.obs['vision']['cam_' + self.arm + '_wrist']['color']
            expected = self.geometry.center(self.xy, self.obs['state'][self.arm + '_ee_pose'], self.rank) if self.servo_done else None
            mask = None
            diag = {}
            if self.servo_done and self.prior_wrist is not None:
                mask, diag = propagate(self.prior_wrist[0], self.prior_wrist[1], wrist_rgb, expected)
                self.log({'event': 'temporal_wrist_mask', 'diagnostics': diag})
            if mask is None:
                mask, diag = self.reader.wrist_mask(wrist_rgb, self.wrist_prototype, expected)
            self.log({'event': 'wrist_preclose', 'diagnostics': diag})
            if mask is not None and (not self.servo_done):
                self.prior_wrist = (wrist_rgb.copy(), mask.copy())
                cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_prior_wrist_mask.png'), mask.astype(np.uint8) * 255)
                fitted, fit_diag = self.geometry.fit(mask, self.obs['state'][self.arm + '_ee_pose'], self.rank, self.xy)
                self.log({'event': 'wrist_geometry_fit', 'fit': fit_diag})
                self.servo_done = True
                if fitted is not None:
                    delta = np.asarray(fitted) - self.xy
                    self.xy = np.asarray(fitted)
                    self.grasp[:2] += delta
                    self.dest = self.grasp + np.r_[self.drop - self.xy, 0.01]
                    if np.linalg.norm(delta) > 0.002:
                        self.enter('wrist_center', self.grasp, 1.0, 10)
                        return
            if mask is not None:
                cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_wrist_mask.png'), mask.astype(np.uint8) * 255)
                self.tracker = self.wrist_tracker(wrist_rgb, mask, self.obs['state'][self.arm + '_ee_pose'])
                self.log({'event': 'identity_handoff', 'source': 'prior_head_rgb_selected_target', 'destination': 'unique_centered_wrist_semantic_mask'})
                if not self.tracker.valid:
                    self.fail('preclose_rgb_texture_missing')
                    return
            else:
                self.log({'event': 'wrist_mask_missing_keep_pick_identity', 'xy': self.xy, 'rank': self.rank, 'pick_identity': self.pick_identity, 'reason': 'arm_occludes_head_rgb'})
            self.enter('close', self.grasp, 0.28, 10)
        elif p == 'reconfirm':
            phase, target, remaining = self.resume_motion
            ee = np.asarray(self.obs['state'][self.arm + '_ee_pose'])
            expected_xyz = np.r_[self.xy, TABLE] + ee[:3] - self.grasp
            rgb = self.obs['vision']['cam_' + self.arm + '_wrist']['color']
            center = self.geometry.center(expected_xyz, ee, self.rank)
            mask, diag = self.reader.wrist_mask(rgb, self.wrist_prototype, center)
            fitted, fit = (None, {'valid': False, 'reason': 'no_semantic_mask'}) if mask is None else self.geometry.fit_carried(mask, ee, self.rank, expected_xyz)
            self.log({'event': 'stationary_attachment_reconfirmation', 'mask': diag, 'fit': fit, 'expected_xyz': expected_xyz})
            self.snapshot()
            if fitted is None:
                self.fail('rgb_stationary_attachment_unconfirmed')
                return
            self.tracker = self.wrist_tracker(rgb, mask, ee)
            if not self.tracker.valid:
                self.fail('stationary_semantic_texture_missing')
                return
            self.tracker.confirmed = True
            self.tracker.lock_attachment()
            self.missing = 0
            self.enter(phase, target[:3], 0.28, remaining, target[3:])
            self.duration = remaining
        elif p == 'close':
            self.lift_start_ee = np.asarray(self.obs['state'][self.arm + '_ee_pose'])
            if getattr(self.tracker, 'wrist', False):
                self.tracker.begin_lift(self.obs['state'][self.arm + '_ee_pose'])
                self.table_center_before_lift = self.geometry.center(self.xy, self.obs['state'][self.arm + '_ee_pose'], self.rank)
            self.enter('lift', self.grasp + [0, 0, 0.14], 0.28, 18)
        elif p == 'lift':
            if self.tracker is None or not self.tracker.confirmed or self.missing:
                wrist_rgb = self.obs['vision']['cam_' + self.arm + '_wrist']['color']
                ee = self.obs['state'][self.arm + '_ee_pose']
                rise = ee[2] - self.lift_start_ee[2]
                expected = self.geometry.center(np.r_[self.xy, TABLE + rise], ee, self.rank)
                mask, diag = self.reader.wrist_mask(wrist_rgb, self.wrist_prototype, expected)
                fitted, fit = (None, {'valid': False, 'reason': 'no_wrist_mask'}) if mask is None else convex_fit_lift(self.geometry, mask, ee, self.rank, self.xy, rise)
                self.log({'event': 'silhouette_lift_verification', 'mask': diag, 'fit': fit})
                if fitted is None:
                    self.fail('rgb_attachment_not_confirmed')
                    return
                cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_lift_mask.png'), mask.astype(np.uint8) * 255)
                self.tracker = self.wrist_tracker(wrist_rgb, mask, ee)
                if not self.tracker.valid:
                    self.fail('lift_semantic_texture_missing')
                    return
                self.tracker.confirmed = True
                self.missing = 0
            if getattr(self.tracker, 'wrist', False):
                self.tracker.lock_attachment()
                self.log({'event': 'attachment_anchor_locked', **self.attachment_support()})
            self.enter('transport', self.dest + [0, 0, 0.14], 0.28, 28)
        elif p == 'transport':
            if getattr(self.tracker, 'wrist', False) and self.tracker.confirmed and (self.missing == 0):
                self.tracker.lock_attachment()
                self.log({'event': 'attachment_anchor_locked', 'reason': 'settled_transport_before_place', **self.attachment_support()})
            self.enter('place', self.dest, 0.28, 20)
        elif p == 'place':
            self.enter('release', self.dest, 1.0, 8)
        elif p == 'release':
            self.tracker = None
            self.enter('retreat', self.dest + [0, 0, 0.14], 1.0, 20)
        elif p == 'retreat':
            following = self.next_pick()
            if following is None or following[1] == self.arm:
                placed, mask, diag = self.object_observer.observe(self.rgb(), self.drop, self.rank, {'iou': 0.5, 'size': 0.35, 'displacement': 0.08, 'mode': 'released_or_park'})
                self.log({'event': 'retreat_release_observation', 'diagnostics': diag, 'following_arm': None if following is None else following[1]})
                if placed is not None:
                    cv2.imwrite(str(self.output / f'{self.trace.stem}_{self.step}_direct_release_mask.png'), mask * 255)
                    if following is not None and following[0] != self.rank:
                        rank, arm, xy = following
                        next_object = self.observe_target(xy, rank)
                        self.log({'event': 'direct_transit_next_visibility', 'target_rank': rank, 'valid': next_object is not None, 'next_action': 'direct' if next_object is not None else 'observation_park'})
                        if next_object is None:
                            self.begin_observation_park()
                            return
                        self.prepared_pick = {'rank': rank, 'arm': arm, 'object': next_object, 'observation_step': self.step, 'prototype': self.reader.appearance_prototype(self.rgb(), next_object['contour'])}
                    self.log({'event': 'observed_direct_empty_transit'})
                    self.complete_placement(placed)
                    return
            self.begin_observation_park()
        elif p == 'park':
            self.identity_resume = 'placement'
            placed = self.bind(self.drop, {'iou': 0.5, 'size': 0.35, 'displacement': 0.08, 'mode': 'released_or_park'})
            if placed is None:
                if self.identity_pending:
                    self.identity_resume = 'placement'
                    self.identity_attempts = 1
                    self.phase = 'identity_acquire'
                return
            self.complete_placement(placed)

    def _memory_get_action_0(self, _frame, _context):
        if _frame.get('done'):
            return _frame
        self.operation_trace.append('sort_nesting_dolls_by_size.vision.initialize_or_acquire')
        if '_diag' in _frame['locals']:
            _diag = _frame['locals']['_diag']
        if '_diag2' in _frame['locals']:
            _diag2 = _frame['locals']['_diag2']
        if '_hist' in _frame['locals']:
            _hist = _frame['locals']['_hist']
        if '_rank' in _frame['locals']:
            _rank = _frame['locals']['_rank']
        if '_raw' in _frame['locals']:
            _raw = _frame['locals']['_raw']
        if '_value' in _frame['locals']:
            _value = _frame['locals']['_value']
        if 'a' in _frame['locals']:
            a = _frame['locals']['a']
        if 'arm' in _frame['locals']:
            arm = _frame['locals']['arm']
        if 'center' in _frame['locals']:
            center = _frame['locals']['center']
        if 'd' in _frame['locals']:
            d = _frame['locals']['d']
        if 'delta' in _frame['locals']:
            delta = _frame['locals']['delta']
        if 'early_motion_complete' in _frame['locals']:
            early_motion_complete = _frame['locals']['early_motion_complete']
        if 'ee' in _frame['locals']:
            ee = _frame['locals']['ee']
        if 'error' in _frame['locals']:
            error = _frame['locals']['error']
        if 'ev' in _frame['locals']:
            ev = _frame['locals']['ev']
        if 'first_row' in _frame['locals']:
            first_row = _frame['locals']['first_row']
        if 'gates' in _frame['locals']:
            gates = _frame['locals']['gates']
        if 'good' in _frame['locals']:
            good = _frame['locals']['good']
        if 'k' in _frame['locals']:
            k = _frame['locals']['k']
        if 'live' in _frame['locals']:
            live = _frame['locals']['live']
        if 'other' in _frame['locals']:
            other = _frame['locals']['other']
        if 'other_q' in _frame['locals']:
            other_q = _frame['locals']['other_q']
        if 'p' in _frame['locals']:
            p = _frame['locals']['p']
        if 'points' in _frame['locals']:
            points = _frame['locals']['points']
        if 'pose' in _frame['locals']:
            pose = _frame['locals']['pose']
        if 'pose_error' in _frame['locals']:
            pose_error = _frame['locals']['pose_error']
        if 'q' in _frame['locals']:
            q = _frame['locals']['q']
        if 'radii' in _frame['locals']:
            radii = _frame['locals']['radii']
        if 'relays' in _frame['locals']:
            relays = _frame['locals']['relays']
        if 'rgb' in _frame['locals']:
            rgb = _frame['locals']['rgb']
        if 'row_diag' in _frame['locals']:
            row_diag = _frame['locals']['row_diag']
        if 'second' in _frame['locals']:
            second = _frame['locals']['second']
        if 'second_order' in _frame['locals']:
            second_order = _frame['locals']['second_order']
        if 'second_row' in _frame['locals']:
            second_row = _frame['locals']['second_row']
        if 'settled' in _frame['locals']:
            settled = _frame['locals']['settled']
        if 'sign' in _frame['locals']:
            sign = _frame['locals']['sign']
        if 'start' in _frame['locals']:
            start = _frame['locals']['start']
        if 'target' in _frame['locals']:
            target = _frame['locals']['target']
        if 'temporal_ok' in _frame['locals']:
            temporal_ok = _frame['locals']['temporal_ok']
        if 'tracking_rgb' in _frame['locals']:
            tracking_rgb = _frame['locals']['tracking_rgb']
        if 'v' in _frame['locals']:
            v = _frame['locals']['v']
        if 'xy' in _frame['locals']:
            xy = _frame['locals']['xy']
        if self.home is None:
            self.home_joints = {k: np.array(v).copy() for k, v in self.joint_observation.items()}
            self.home = {k: list(v) for k, v in self.obs['state'].items()}
            self.home['left_ee_joint_state'] = [1.0]
            self.home['right_ee_joint_state'] = [1.0]
            self.command = {k: list(v) for k, v in self.home.items()}
            self.phase = 'observe'
            self.observe_tick = 0
            self.snapshot()
            self.log({'event': 'observation_begin'})
        if self.phase == 'observe':
            self.step += 1
            self.observe_tick += 1
            a = min(1.0, self.observe_tick / 30.0)
            a = 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5
            for arm, sign in [('left', -1), ('right', 1)]:
                start = np.asarray(self.home[arm + '_ee_pose'])
                target = np.array([sign * 0.45, -0.4, 1.05])
                self.command[arm + '_ee_pose'] = np.r_[start[:3] * (1 - a) + target * a, start[3:]].tolist()
                self.command[arm + '_ee_joint_state'] = [1.0]
            if self.observe_tick >= 36 and (self.observe_tick - 36) % 30 == 0:
                self.snapshot()
                rgb = self.rgb()
                live = bool(np.count_nonzero(rgb) > rgb.size * 0.01 and np.std(rgb) > 2.0)
                self.log({'event': 'rgb_liveness', 'live': live, 'std': float(np.std(rgb))})
                if live:
                    p = self.read()
                    if not p['valid']:
                        self.fail('rgb_five_contours_or_height_separation')
                    else:
                        self.observation_park = {k: list(v) for k, v in self.obs['state'].items()}
                        self.ordered = []
                        for _rank, _raw in enumerate(p['dolls']):
                            _value, _diag = self.object_observer.qualify(_raw, _rank, np.asarray(_raw['xy'], dtype=float), {'iou': 0.65, 'size': 0.24, 'displacement': 0.055, 'mode': 'precontact'})
                            self.log({'event': 'initial_profile_grounded_identity', 'target_rank': _rank, 'diagnostics': _diag})
                            self.ordered.append(_value if _value is not None else dict(_raw, identity_qualified=False))
                        radii = [float(np.max(np.linalg.norm(points[:, :2], axis=1))) for points in self.geometry.points]
                        first_row, row_diag = choose_row(self.ordered, radii)
                        second = self.read()
                        if second.get('valid'):
                            second_order = []
                            for _rank, _raw in enumerate(second['dolls']):
                                _value, _diag = self.object_observer.qualify(_raw, _rank, np.asarray(_raw['xy'], dtype=float), {'iou': 0.65, 'size': 0.24, 'displacement': 0.055, 'mode': 'precontact'})
                                second_order.append(_value if _value is not None else dict(_raw, identity_qualified=False))
                            second_row, _diag2 = choose_row(second_order, radii)
                        else:
                            second_row = first_row
                        self.goal_row, _hist = temporal_consensus([], first_row.tolist(), tol=0.012)
                        self.goal_row, _hist = temporal_consensus(_hist, second_row.tolist(), tol=0.012)
                        self.goal_row = np.asarray(self.goal_row)
                        row_diag.update({'temporal_consensus': True, 'consensus_tolerance_m': 0.012, 'first_row': first_row.tolist(), 'second_row': np.asarray(second_row).tolist()})
                        self.log({'event': 'rgb_goal_row_selected', 'goal_row': self.goal_row, 'diagnostics': row_diag})
                        self.pick_order = row_diag.get('pick_order', list(range(5)))
                        self.pick_index = 0
                        self.rank = self.pick_order[0]
                        relays = row_diag['selected_relays']
                        self.required_picks = 5 + relays
                        self.duration_scale = min(1.0, max(0.35, (930 - 24 * self.required_picks) / (171 * self.required_picks)))
                        self.log({'event': 'action_budget_frozen', 'required_picks': self.required_picks, 'duration_scale': self.duration_scale, 'official_steps': 1050})
                        self.begin_pick()
                elif self.observe_tick >= 156:
                    self.fail('head_rgb_unavailable_after_observation')
            self.log({'event': 'observation_action', 'action': self.command, 'observed': self.obs['state']})
            if self.done:
                self.phase = 'failed'
            _result = [{k: list(v) for k, v in self.command.items()}]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        self.step += 1
        if self.phase == 'evidence_wait':
            self.evidence_wait()
            _frame.update(result=[self.command], done=True)
            return _frame
        if self.phase == 'row_observe':
            self.final_row()
            _frame.update(result=[self.command], done=True)
            return _frame
        if self.phase == 'identity_acquire':
            self.identity_attempts += 1
            xy = self.xy if self.identity_resume == 'pick' else self.drop
            gates = None if self.identity_resume == 'pick' else {'iou': 0.5, 'size': 0.35, 'displacement': 0.08, 'mode': 'released_or_park'}
            d = self.observe_target(xy, self.rank, gates)
            self.log({'event': 'fresh_identity_acquisition', 'attempt': self.identity_attempts, 'qualified': d is not None})
            if d is not None:
                if self.identity_resume == 'pick':
                    self.prepared_pick = {'rank': self.rank, 'arm': self.arm, 'object': d, 'observation_step': self.step, 'prototype': self.reader.appearance_prototype(self.rgb(), d['contour'])}
                    self.begin_pick()
                else:
                    self.complete_placement(d)
            elif self.identity_attempts >= 3:
                self.fail('rgb_object_identity_or_upright_unconfirmed')
            _result = [self.command]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        if self.phase == 'return_home':
            self.home_tick += 1
            home_evidence = home_geometry(self.obs['state'], self.home)
            self.log({'event': 'official_home_geometry_estimate', **home_evidence})
            if not self.done and home_evidence['valid']:
                self.done = bool((self.last_row_evidence or {}).get('valid', False))
                self.log({'event': 'program_complete_or_awaiting_official_verdict',
                          'row_estimate_valid': bool((self.last_row_evidence or {}).get('valid', False)),
                          'evidence': 'RGB relative row and measured own end-effector home',
                          'official_success_claimed': False})
                self.snapshot()
            _result = [self.command]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        if self.done:
            self.log({'event': 'hold_action', 'action': self.command, 'observed': self.obs['state']})
            _result = [self.command]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        _values = locals()
        _frame['locals'].update({k: _values[k] for k in ['_diag', '_diag2', '_hist', '_rank', '_raw', '_value', 'a', 'arm', 'center', 'd', 'delta', 'early_motion_complete', 'ee', 'error', 'ev', 'first_row', 'gates', 'good', 'k', 'live', 'other', 'other_q', 'p', 'points', 'pose', 'pose_error', 'q', 'radii', 'relays', 'rgb', 'row_diag', 'second', 'second_order', 'second_row', 'settled', 'sign', 'start', 'target', 'temporal_ok', 'tracking_rgb', 'v', 'xy'] if k in _values})
        return _frame

    def _memory_get_action_1(self, _frame, _context):
        if _frame.get('done'):
            return _frame
        self.operation_trace.append('sort_nesting_dolls_by_size.motion.track_and_advance')
        if '_diag' in _frame['locals']:
            _diag = _frame['locals']['_diag']
        if '_diag2' in _frame['locals']:
            _diag2 = _frame['locals']['_diag2']
        if '_hist' in _frame['locals']:
            _hist = _frame['locals']['_hist']
        if '_rank' in _frame['locals']:
            _rank = _frame['locals']['_rank']
        if '_raw' in _frame['locals']:
            _raw = _frame['locals']['_raw']
        if '_value' in _frame['locals']:
            _value = _frame['locals']['_value']
        if 'a' in _frame['locals']:
            a = _frame['locals']['a']
        if 'arm' in _frame['locals']:
            arm = _frame['locals']['arm']
        if 'center' in _frame['locals']:
            center = _frame['locals']['center']
        if 'd' in _frame['locals']:
            d = _frame['locals']['d']
        if 'delta' in _frame['locals']:
            delta = _frame['locals']['delta']
        if 'early_motion_complete' in _frame['locals']:
            early_motion_complete = _frame['locals']['early_motion_complete']
        if 'ee' in _frame['locals']:
            ee = _frame['locals']['ee']
        if 'error' in _frame['locals']:
            error = _frame['locals']['error']
        if 'ev' in _frame['locals']:
            ev = _frame['locals']['ev']
        if 'first_row' in _frame['locals']:
            first_row = _frame['locals']['first_row']
        if 'gates' in _frame['locals']:
            gates = _frame['locals']['gates']
        if 'good' in _frame['locals']:
            good = _frame['locals']['good']
        if 'k' in _frame['locals']:
            k = _frame['locals']['k']
        if 'live' in _frame['locals']:
            live = _frame['locals']['live']
        if 'other' in _frame['locals']:
            other = _frame['locals']['other']
        if 'other_q' in _frame['locals']:
            other_q = _frame['locals']['other_q']
        if 'p' in _frame['locals']:
            p = _frame['locals']['p']
        if 'points' in _frame['locals']:
            points = _frame['locals']['points']
        if 'pose' in _frame['locals']:
            pose = _frame['locals']['pose']
        if 'pose_error' in _frame['locals']:
            pose_error = _frame['locals']['pose_error']
        if 'q' in _frame['locals']:
            q = _frame['locals']['q']
        if 'radii' in _frame['locals']:
            radii = _frame['locals']['radii']
        if 'relays' in _frame['locals']:
            relays = _frame['locals']['relays']
        if 'rgb' in _frame['locals']:
            rgb = _frame['locals']['rgb']
        if 'row_diag' in _frame['locals']:
            row_diag = _frame['locals']['row_diag']
        if 'second' in _frame['locals']:
            second = _frame['locals']['second']
        if 'second_order' in _frame['locals']:
            second_order = _frame['locals']['second_order']
        if 'second_row' in _frame['locals']:
            second_row = _frame['locals']['second_row']
        if 'settled' in _frame['locals']:
            settled = _frame['locals']['settled']
        if 'sign' in _frame['locals']:
            sign = _frame['locals']['sign']
        if 'start' in _frame['locals']:
            start = _frame['locals']['start']
        if 'target' in _frame['locals']:
            target = _frame['locals']['target']
        if 'temporal_ok' in _frame['locals']:
            temporal_ok = _frame['locals']['temporal_ok']
        if 'tracking_rgb' in _frame['locals']:
            tracking_rgb = _frame['locals']['tracking_rgb']
        if 'v' in _frame['locals']:
            v = _frame['locals']['v']
        if 'xy' in _frame['locals']:
            xy = _frame['locals']['xy']
        if self.tracker is not None and self.phase in ('close', 'lift', 'transport', 'place'):
            if getattr(self.tracker, 'wrist', False) and self.phase == 'lift':
                center = self.geometry.center(self.xy, self.obs['state'][self.arm + '_ee_pose'], self.rank)
                self.tracker.static_shift_px = float(np.linalg.norm(np.asarray(center) - self.table_center_before_lift)) if center is not None and self.table_center_before_lift is not None else 0.0
            tracking_rgb = self.obs['vision']['cam_' + self.arm + '_wrist']['color'] if getattr(self.tracker, 'wrist', False) else self.rgb()
            ev = self.tracker.update(tracking_rgb, self.obs['state'][self.arm + '_ee_pose'])
            temporal_ok = self.temporal_attachment.update(self.rank, self.xy, self.obs['state'][self.arm + '_ee_pose'], bool(ev.get('valid', False))) if self.temporal_attachment is not None else bool(ev.get('valid', False))
            self.log({'event': 'attachment', **ev, 'temporal_attachment_ok': temporal_ok, 'temporal_dropout': getattr(self.temporal_attachment, 'dropout', 0)})
            if self.phase in ('transport', 'place') and getattr(self.tracker, 'wrist', False):
                good = ev['valid'] and ev.get('locked_error_px') is not None and (ev['locked_error_px'] < 25) and temporal_ok
            else:
                good = ev['valid'] and ev.get('attached_error_px', 1000000000.0) < 8
            self.missing = 0 if good else self.missing + 1
            if self.phase in ('transport', 'place') and self.missing >= 3:
                ee = np.asarray(self.obs['state'][self.arm + '_ee_pose'][:3])
                delta = ee - self.dest
                if self.phase == 'place' and self.tracker.confirmed and (np.linalg.norm(delta[:2]) < 0.018) and (abs(delta[2]) < 0.035):
                    self.log({'event': 'contact_release', 'endpoint_delta': delta, 'evidence': ev})
                    self.dest = ee.copy()
                    self.tracker = None
                    self.enter('release', self.dest, 1.0, 8)
                elif self.reconfirmations < 2:
                    self.reconfirmations += 1
                    self.resume_motion = (self.phase, self.target.copy(), max(8, self.duration - self.tick))
                    self.tracker = None
                    self.enter('reconfirm', ee, 0.28, 6, self.obs['state'][self.arm + '_ee_pose'][3:])
                    self.duration = 6
                    self.command[self.arm + '_ee_pose'] = list(self.obs['state'][self.arm + '_ee_pose'])
                    _result = [self.command]
                    _frame['result'] = _result
                    _frame['done'] = True
                    return _frame
                else:
                    self.fail('rgb_attachment_lost_after_reconfirmation_budget')
                    _result = [self.command]
                    _frame['result'] = _result
                    _frame['done'] = True
                    return _frame
        if self.phase == 'close':
            self.close_samples.append(float(np.asarray(self.obs['state'][self.arm + '_ee_joint_state']).reshape(-1)[0]))
            settled = len(self.close_samples) == 4 and np.ptp(self.close_samples) <= 0.006
            self.log({'event': 'close_settling', 'samples': list(self.close_samples), 'settled': bool(settled), 'tick': self.tick})
            if self.tick >= self.duration and (not settled):
                if self.tick >= 40:
                    self.fail('gripper_did_not_settle')
                    _result = [self.command]
                    _frame['result'] = _result
                    _frame['done'] = True
                    return _frame
                self.duration = self.tick + 1
        early_motion_complete = motion_complete(self)
        if early_motion_complete:
            self.log({'event': 'observed_settled_motion_tail', 'saved_tail_actions': self.duration - self.tick})
        if self.tick >= self.duration or early_motion_complete:
            error = float(np.linalg.norm(np.asarray(self.obs['state'][self.arm + '_ee_pose'][:3]) - self.target[:3]))
            if error > 0.018 and self.phase not in ('close', 'release'):
                if self.tick >= self.duration + 12:
                    self.fail('ee_endpoint_unreached')
                    _result = [self.command]
                    _frame['result'] = _result
                    _frame['done'] = True
                    return _frame
            else:
                self.log({'event': 'stage_end', 'error': error, 'observed': self.obs['state']})
                self.advance()
        if self.done:
            self.log({'event': 'hold_action', 'action': self.command, 'observed': self.obs['state']})
            _result = [self.command]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        if self.phase in ('return_home', 'identity_acquire', 'evidence_wait', 'row_observe'):
            _result = [self.command]
            _frame['result'] = _result
            _frame['done'] = True
            return _frame
        _values = locals()
        _frame['locals'].update({k: _values[k] for k in ['_diag', '_diag2', '_hist', '_rank', '_raw', '_value', 'a', 'arm', 'center', 'd', 'delta', 'early_motion_complete', 'ee', 'error', 'ev', 'first_row', 'gates', 'good', 'k', 'live', 'other', 'other_q', 'p', 'points', 'pose', 'pose_error', 'q', 'radii', 'relays', 'rgb', 'row_diag', 'second', 'second_order', 'second_row', 'settled', 'sign', 'start', 'target', 'temporal_ok', 'tracking_rgb', 'v', 'xy'] if k in _values})
        return _frame

    def _memory_get_action_2(self, _frame, _context):
        if _frame.get('done'):
            return _frame
        self.operation_trace.append('sort_nesting_dolls_by_size.motion.interpolate')
        if '_diag' in _frame['locals']:
            _diag = _frame['locals']['_diag']
        if '_diag2' in _frame['locals']:
            _diag2 = _frame['locals']['_diag2']
        if '_hist' in _frame['locals']:
            _hist = _frame['locals']['_hist']
        if '_rank' in _frame['locals']:
            _rank = _frame['locals']['_rank']
        if '_raw' in _frame['locals']:
            _raw = _frame['locals']['_raw']
        if '_value' in _frame['locals']:
            _value = _frame['locals']['_value']
        if 'a' in _frame['locals']:
            a = _frame['locals']['a']
        if 'arm' in _frame['locals']:
            arm = _frame['locals']['arm']
        if 'center' in _frame['locals']:
            center = _frame['locals']['center']
        if 'd' in _frame['locals']:
            d = _frame['locals']['d']
        if 'delta' in _frame['locals']:
            delta = _frame['locals']['delta']
        if 'early_motion_complete' in _frame['locals']:
            early_motion_complete = _frame['locals']['early_motion_complete']
        if 'ee' in _frame['locals']:
            ee = _frame['locals']['ee']
        if 'error' in _frame['locals']:
            error = _frame['locals']['error']
        if 'ev' in _frame['locals']:
            ev = _frame['locals']['ev']
        if 'first_row' in _frame['locals']:
            first_row = _frame['locals']['first_row']
        if 'gates' in _frame['locals']:
            gates = _frame['locals']['gates']
        if 'good' in _frame['locals']:
            good = _frame['locals']['good']
        if 'k' in _frame['locals']:
            k = _frame['locals']['k']
        if 'live' in _frame['locals']:
            live = _frame['locals']['live']
        if 'other' in _frame['locals']:
            other = _frame['locals']['other']
        if 'other_q' in _frame['locals']:
            other_q = _frame['locals']['other_q']
        if 'p' in _frame['locals']:
            p = _frame['locals']['p']
        if 'points' in _frame['locals']:
            points = _frame['locals']['points']
        if 'pose' in _frame['locals']:
            pose = _frame['locals']['pose']
        if 'pose_error' in _frame['locals']:
            pose_error = _frame['locals']['pose_error']
        if 'q' in _frame['locals']:
            q = _frame['locals']['q']
        if 'radii' in _frame['locals']:
            radii = _frame['locals']['radii']
        if 'relays' in _frame['locals']:
            relays = _frame['locals']['relays']
        if 'rgb' in _frame['locals']:
            rgb = _frame['locals']['rgb']
        if 'row_diag' in _frame['locals']:
            row_diag = _frame['locals']['row_diag']
        if 'second' in _frame['locals']:
            second = _frame['locals']['second']
        if 'second_order' in _frame['locals']:
            second_order = _frame['locals']['second_order']
        if 'second_row' in _frame['locals']:
            second_row = _frame['locals']['second_row']
        if 'settled' in _frame['locals']:
            settled = _frame['locals']['settled']
        if 'sign' in _frame['locals']:
            sign = _frame['locals']['sign']
        if 'start' in _frame['locals']:
            start = _frame['locals']['start']
        if 'target' in _frame['locals']:
            target = _frame['locals']['target']
        if 'temporal_ok' in _frame['locals']:
            temporal_ok = _frame['locals']['temporal_ok']
        if 'tracking_rgb' in _frame['locals']:
            tracking_rgb = _frame['locals']['tracking_rgb']
        if 'v' in _frame['locals']:
            v = _frame['locals']['v']
        if 'xy' in _frame['locals']:
            xy = _frame['locals']['xy']
        a = min(1.0, (self.tick + 1) / max(1, self.duration - 5))
        a = 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5
        q = np.roll(Slerp([0, 1], Rotation.concatenate([rotation(self.start[3:]), rotation(self.target[3:])]))(a).as_quat(), 1)
        pose = np.r_[self.start[:3] * (1 - a) + self.target[:3] * a, q].tolist()
        self.command[self.arm + '_ee_pose'] = pose
        self.command[self.arm + '_ee_joint_state'] = [self.grip]
        self.tick += 1
        if self.phase == 'park' and self.preposition is not None:
            other, start, target = self.preposition
            other_q = np.roll(Slerp([0, 1], Rotation.concatenate([rotation(start[3:]), rotation(target[3:])]))(a).as_quat(), 1)
            self.command[other + '_ee_pose'] = np.r_[start[:3] * (1 - a) + target[:3] * a, other_q].tolist()
            self.command[other + '_ee_joint_state'] = [1.0]
        _values = locals()
        _frame['locals'].update({k: _values[k] for k in ['_diag', '_diag2', '_hist', '_rank', '_raw', '_value', 'a', 'arm', 'center', 'd', 'delta', 'early_motion_complete', 'ee', 'error', 'ev', 'first_row', 'gates', 'good', 'k', 'live', 'other', 'other_q', 'p', 'points', 'pose', 'pose_error', 'q', 'radii', 'relays', 'rgb', 'row_diag', 'second', 'second_order', 'second_row', 'settled', 'sign', 'start', 'target', 'temporal_ok', 'tracking_rgb', 'v', 'xy'] if k in _values})
        return _frame

    def _memory_get_action_3(self, _frame, _context):
        if _frame.get('done'):
            return _frame
        self.operation_trace.append('sort_nesting_dolls_by_size.action.record_and_emit')
        if '_diag' in _frame['locals']:
            _diag = _frame['locals']['_diag']
        if '_diag2' in _frame['locals']:
            _diag2 = _frame['locals']['_diag2']
        if '_hist' in _frame['locals']:
            _hist = _frame['locals']['_hist']
        if '_rank' in _frame['locals']:
            _rank = _frame['locals']['_rank']
        if '_raw' in _frame['locals']:
            _raw = _frame['locals']['_raw']
        if '_value' in _frame['locals']:
            _value = _frame['locals']['_value']
        if 'a' in _frame['locals']:
            a = _frame['locals']['a']
        if 'arm' in _frame['locals']:
            arm = _frame['locals']['arm']
        if 'center' in _frame['locals']:
            center = _frame['locals']['center']
        if 'd' in _frame['locals']:
            d = _frame['locals']['d']
        if 'delta' in _frame['locals']:
            delta = _frame['locals']['delta']
        if 'early_motion_complete' in _frame['locals']:
            early_motion_complete = _frame['locals']['early_motion_complete']
        if 'ee' in _frame['locals']:
            ee = _frame['locals']['ee']
        if 'error' in _frame['locals']:
            error = _frame['locals']['error']
        if 'ev' in _frame['locals']:
            ev = _frame['locals']['ev']
        if 'first_row' in _frame['locals']:
            first_row = _frame['locals']['first_row']
        if 'gates' in _frame['locals']:
            gates = _frame['locals']['gates']
        if 'good' in _frame['locals']:
            good = _frame['locals']['good']
        if 'k' in _frame['locals']:
            k = _frame['locals']['k']
        if 'live' in _frame['locals']:
            live = _frame['locals']['live']
        if 'other' in _frame['locals']:
            other = _frame['locals']['other']
        if 'other_q' in _frame['locals']:
            other_q = _frame['locals']['other_q']
        if 'p' in _frame['locals']:
            p = _frame['locals']['p']
        if 'points' in _frame['locals']:
            points = _frame['locals']['points']
        if 'pose' in _frame['locals']:
            pose = _frame['locals']['pose']
        if 'pose_error' in _frame['locals']:
            pose_error = _frame['locals']['pose_error']
        if 'q' in _frame['locals']:
            q = _frame['locals']['q']
        if 'radii' in _frame['locals']:
            radii = _frame['locals']['radii']
        if 'relays' in _frame['locals']:
            relays = _frame['locals']['relays']
        if 'rgb' in _frame['locals']:
            rgb = _frame['locals']['rgb']
        if 'row_diag' in _frame['locals']:
            row_diag = _frame['locals']['row_diag']
        if 'second' in _frame['locals']:
            second = _frame['locals']['second']
        if 'second_order' in _frame['locals']:
            second_order = _frame['locals']['second_order']
        if 'second_row' in _frame['locals']:
            second_row = _frame['locals']['second_row']
        if 'settled' in _frame['locals']:
            settled = _frame['locals']['settled']
        if 'sign' in _frame['locals']:
            sign = _frame['locals']['sign']
        if 'start' in _frame['locals']:
            start = _frame['locals']['start']
        if 'target' in _frame['locals']:
            target = _frame['locals']['target']
        if 'temporal_ok' in _frame['locals']:
            temporal_ok = _frame['locals']['temporal_ok']
        if 'tracking_rgb' in _frame['locals']:
            tracking_rgb = _frame['locals']['tracking_rgb']
        if 'v' in _frame['locals']:
            v = _frame['locals']['v']
        if 'xy' in _frame['locals']:
            xy = _frame['locals']['xy']
        self.log({'event': 'action', 'action': self.command, 'observed': self.obs['state']})
        _result = [{k: list(v) for k, v in self.command.items()}]
        _frame['result'] = _result
        _frame['done'] = True
        return _frame
        _values = locals()
        _frame['locals'].update({k: _values[k] for k in ['_diag', '_diag2', '_hist', '_rank', '_raw', '_value', 'a', 'arm', 'center', 'd', 'delta', 'early_motion_complete', 'ee', 'error', 'ev', 'first_row', 'gates', 'good', 'k', 'live', 'other', 'other_q', 'p', 'points', 'pose', 'pose_error', 'q', 'radii', 'relays', 'rgb', 'row_diag', 'second', 'second_order', 'second_row', 'settled', 'sign', 'start', 'target', 'temporal_ok', 'tracking_rgb', 'v', 'xy'] if k in _values})
        return _frame

    def get_action(self):
        return _execute_memory(self, __file__, 'EpisodePolicy.get_action', {})

class Model:

    def __init__(self, model_cfg, reader=None):
        if reader is None:
            import torch
            torch.set_num_threads(1)
        cv2.setNumThreads(1)
        self.cfg = model_cfg
        self.reader = Reader() if reader is None else reader
        self.reset()

    def reset(self):
        for env in getattr(self, 'envs', {}).values():
            env.trace_file.close()
        self.envs = {}

    def set_output(self, path):
        self.cfg['trace_dir'] = path
        self.reset()

    def prepare_case(self, case_meta=None):
        self.reset()

    def on_trial_end(self, result=None):
        pass

    def update_obs_batch(self, observations):
        for obs in observations:
            idx = int(obs['env_idx'])
            if idx not in self.envs:
                self.envs[idx] = EpisodePolicy(self.cfg, self.reader)
            self.envs[idx].update_obs(obs)

    def get_action_batch(self, indices):
        actions = []
        for i in indices:
            env = self.envs[int(i)]
            started = time.perf_counter()
            poses = env.get_action()[0]
            policy_time = time.perf_counter() - started
            started = time.perf_counter()
            if env.phase in ('evidence_wait', 'row_observe'):
                joints = {arm + '_arm_joint_state': env.joint_observation[arm].tolist()
                          for arm in ('left', 'right')}
                joints.update({arm + '_ee_joint_state': list(env.obs['state'][arm + '_ee_joint_state'])
                               for arm in ('left', 'right')})
                diag = {'source': 'measured_own_joint_hold_under_visual_uncertainty',
                        'object_motion_inferred': False}
            elif env.phase == 'return_home':
                joints = {}
                diag = {}
                for arm in ('left', 'right'):
                    current = env.joint_observation[arm]
                    target = current + np.clip(env.home_joints[arm] - current, -0.1, 0.1)
                    joints[arm + '_arm_joint_state'] = target.tolist()
                    joints[arm + '_ee_joint_state'] = [1.0]
                    diag[arm] = {'source': 'initial_official_proprio', 'max_joint_delta': float(np.max(np.abs(target - current)))}
            else:
                joints, diag = env.joint_control.convert(poses, env.obs['state'], env.joint_observation, env.empty_clearance())
            env.log({'event': 'joint_control', 'diagnostics': diag, 'joint_action': joints, 'cpu_ik_seconds': time.perf_counter() - started, 'policy_wall_seconds': policy_time})
            actions.append([joints])
        return actions

    def infer(self, observations):
        self.update_obs_batch(observations)
        return self.get_action_batch([o['env_idx'] for o in observations])

    def mark_natural_end(self, indices):
        for i in indices:
            self.envs[int(i)].mark_natural_end()
