"""Compare deterministic perception and geometry probes in separate processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(args):
    import numpy as np
    old = json.loads((args.original / 'task_registry.json').read_text())['tasks'][args.task]
    family = old['family']
    original = args.original / old['source']
    if family == 'coap_four':
        original = original / 'locked' / ('qv049' if args.task.endswith('_random') else 'qv065')
    if family == 'step':
        package = old['module'].rsplit('.', 1)[0]
        names = [package + '.perception']
        if args.task == 'sort_nesting_dolls_by_size':
            names = [package + '.nesting_rgb.perception', package + '.nesting_rgb.profile_geometry']
        if args.task == 'swap_T':
            names = [package + '.geometry']
    else:
        package = 'coap' if family in ('coap_zjx', 'coap_zyr', 'coap_four') else 'b88r42wq'
        names = [package + '.perception', package + ('.geometry' if family in ('wgw44wsg', 'coap_zyr') else '.motion')]
        if args.task.startswith('arrange_largest_number'):
            names[0] = 'coap.number_perception'
        if args.task == 'pick_from_conveyor_by_image':
            names[0] = 'coap.vision'
    if args.mode == 'original':
        sys.path.insert(0, str(original))
        load = importlib.import_module
    else:
        sys.path.insert(0, str(args.candidate.parent.parent.parent))
        scope = importlib.import_module('XPolicyLab.policy.' + args.candidate.name + '.core.modules').TaskModules(args.task)
        load = scope.import_module
    generator = np.random.default_rng(8241)
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[210:250, 200:260] = [230, 25, 20]
    rgb[150:220, 360:400] = [30, 200, 30]
    crop = generator.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    quaternion = np.array([.8, .2, -.1, .3]); quaternion /= np.linalg.norm(quaternion)
    probes = {
        'rx': (.31,), 'ry': (-.24,), 'rz': (.51,),
        'normalize_quat': (quaternion,), 'quat_matrix': (quaternion,),
        'matrix_quat': (np.eye(3),), 'down_quat': (.35,),
        'pose_matrix': (np.r_[.1, .2, .3, quaternion],),
        'recognize_digits': (rgb,), 'toy_face_pattern': (crop,),
        'red_mask': (rgb,), '_rgb': (rgb,),
        'box_iou': ([1., 2., 30., 40.], [10., 15., 45., 50.]),
        'masked_iou': (np.eye(6, dtype=bool), np.eye(6, dtype=bool), np.ones((6, 6), bool)),
        'inside_table': ([.1, -.1],), 'rotation': (.37,),
        'normalized': (np.array([1., 2., 3.]),), 'color_masks': (rgb,),
        'wrist_rim_evidence': (rgb,),
        '_rectangle_distance': (np.array([[.0, .0], [.3, -.2]]), [0., 0.], [.1, .1]),
        'interpolate_pose': (np.r_[.1, .2, .3, quaternion], np.r_[.3, .4, .5, 1., 0., 0., 0.], .25),
    }
    def plain(value):
        if isinstance(value, np.ndarray):
            return dict(dtype=str(value.dtype), shape=list(value.shape), value=value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    results = {}
    for name in names:
        module = load(name)
        for function, values in probes.items():
            method = getattr(module, function, None)
            if not callable(method):
                continue
            key = name + ':' + function
            try:
                results[key] = dict(result=plain(method(*values)))
            except Exception as error:
                results[key] = dict(error=type(error).__name__)
    print('PROBE_RESULT=' + json.dumps(results, sort_keys=True, allow_nan=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--mode', choices=['original', 'candidate'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode:
        return worker(args)
    registry = json.loads((args.original / 'task_registry.json').read_text())['tasks']
    tasks = [t for t in registry if t not in {'insert_key', 'insert_tubes', 'plug_in_charger', 'solve_equation'}]
    if args.task:
        tasks = [args.task]
    def compare(task):
        values = {}
        for mode in ('original', 'candidate'):
            command = [sys.executable, str(Path(__file__).resolve()), '--original', str(args.original), '--candidate', str(args.candidate), '--task', task, '--mode', mode]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=90, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            if proc.returncode:
                return task, dict(passed=False, mode=mode, error=proc.stderr[-2500:])
            output = next(line.removeprefix('PROBE_RESULT=') for line in proc.stdout.splitlines() if line.startswith('PROBE_RESULT='))
            values[mode] = json.loads(output)
        first, second = values['original'], values['candidate']
        successful = sum('result' in value for value in first.values())
        return task, dict(passed=first == second and successful > 0, probes=len(first), successful_probes=successful, differences=[key for key in first.keys() | second.keys() if first.get(key) != second.get(key)])
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = dict(executor.map(compare, tasks))
    report = dict(passed=all(value['passed'] for value in result.values()), tasks=len(result), probes=sum(value.get('probes', 0) for value in result.values()), results=result, scope='Deterministic perception/geometry helper probes; excludes GPU models and full simulator behavior')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
