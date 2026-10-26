"""Validate adapter structure, selected source identities, and task routes."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))


def audit():
    from routing import resolve
    registry = json.loads((ROOT / 'tasks/registry.json').read_text())['tasks']
    release = json.loads((ROOT / 'tasks/release.json').read_text())
    errors, code = [], set()
    for task in release:
        route = resolve(task)
        if route['canonical_task'] not in registry:
            errors.append('Unregistered task: ' + task)
    references = 0
    for task, route in registry.items():
        if route['driver'] == 'vla':
            continue
        modules = json.loads((ROOT / route['modules']).read_text())
        for module, item in modules.items():
            if item.get('namespace'):
                continue
            path = ROOT / item['file']
            raw = path.read_bytes()
            references += 1
            code.add(item['file'])
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                errors.append('Source identity changed: ' + task + ':' + module)
            ast.parse(raw, filename=str(path))
        for key in ('model', 'deploy', 'service'):
            value = route.get(key)
            if value and value.split(':')[0] not in modules:
                errors.append('Missing entry module: ' + task + ':' + value)
    for path in ROOT.rglob('*'):
        if path.is_symlink() and not path.exists():
            errors.append('Broken resource link: ' + str(path.relative_to(ROOT)))
    for path in ROOT.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for name in ['model.py', 'deploy.py', 'adaptor.py']:
        ast.parse((ROOT / name).read_text())
    required = {'policy_name', 'protocol', 'host', 'port', 'bench_name', 'task_name', 'ckpt_name', 'env_cfg_type', 'seed', 'action_type', 'gpu_id', 'eval_batch'}
    config = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in (ROOT / 'deploy.yml').read_text().splitlines() if ':' in line and not line.startswith('#')}
    if required - config.keys():
        errors.append('Missing deploy keys: ' + str(sorted(required - config.keys())))
    if config['policy_name'] != ROOT.name:
        errors.append('policy_name must match directory name')
    parity = json.loads((ROOT / 'validation/source_parity.json').read_text())
    if not all(item['algorithm_unchanged'] for entries in parity.values() for item in entries):
        errors.append('Algorithm parity failed')
    return dict(passed=not errors, tasks=len(release), scripted_routes=sum(r['driver'] != 'vla' for r in registry.values()), unique_modules=len(code), module_references=references, errors=errors)


def check_runtime(task, runtime=True):
    from routing import resolve
    from core.runtime import apply
    import importlib.util
    import os
    route=resolve(task);issues=[];cfg=apply({'task_name':task})
    def path(value,label):
        if not value or not Path(value).expanduser().exists():issues.append(label+': missing or unconfigured '+str(value))
    if not runtime:return {'task':task,'source_verified':True,'runtime_checked':False,'issues':issues}
    def setting(key,env):return cfg.get(key) or os.environ.get(env)
    for module in ['numpy','scipy','cv2','PIL','yaml','XPolicyLab']:
        if importlib.util.find_spec(module) is None:issues.append('Missing Python module: '+module)
    family=route['driver']
    if family=='vla':
        path(cfg.get('model_path'),'VLA model_path in runtime config')
    if family=='step':
        path(setting('robot_static_path','ME_BRAIN_ROBOT_STATIC'),'robot static JSON')
    if task in ('stack_blocks_random','stack_bowls_random','sort_nesting_dolls_by_size','sort_nesting_dolls_by_size_random'):
        root=setting('stack_assets_root','ME_BRAIN_STACK_ASSETS') if task.startswith('stack_') else setting('nesting_assets_root','ME_BRAIN_NESTING_ASSETS')
        path(root,'SAM3 assets root')
        if root:
            weight=Path(root)/'assets/sam3.pt';path(weight,'SAM3 checkpoint')
            if task.startswith('stack_') and weight.is_file():
                h=hashlib.sha256()
                with weight.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                if h.hexdigest()!='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e':issues.append('SAM3 checkpoint hash mismatch')
    if task in ('pack_objects_into_box', 'pack_objects_into_box_random', 'make_toast', 'make_toast_random'):path(os.environ.get('COAP_PUBLIC_RGB_MODELS'),'COAP public RGB models')
    if task in ('classify_objects','classify_objects_by_language'):
        for key,default in [('CLASSIFY_CLIP_PATH','clip-vit-base-patch32'),('CLASSIFY_DINO_PATH','grounding-dino-tiny')]:
            path(os.environ.get(key, str(ROOT / 'model_assets/classification' / default)),key)
    if task in ('classify_objects','classify_objects_by_language','imitate_sorting_sequence'):
        rel='coap/assets/sorting_dinov2_small/model.safetensors'
        path(cfg.get('source_assets',{}).get(rel,ROOT/'resources'/route['canonical_task']/rel),'DINOv2-small source asset')
    if task in ('general_pickup','pick_from_conveyor_by_image'):
        root=setting('perception_assets_root','ME_BRAIN_PERCEPTION_ASSETS') or str(ROOT / 'model_assets/perception')
        models=['clip-vit-base-patch32/5be8cdefa7ef9c95/clip-vit-base-patch32']
        models+=['grounding-dino-tiny/074cad48ab8387c7/grounding-dino-tiny','depth-anything-v2-metric-indoor-small/1bf8ab4268a28b9b/depth-anything-v2-metric-indoor-small'] if task=='general_pickup' else ['grounding-dino-tiny/b94087d6c78e04d4/grounding-dino-tiny']
        for model in models:path(Path(root)/model,'public model')
    if family=='equation':
        root=setting('equation_assets_root','ME_BRAIN_EQUATION_ASSETS');path(root,'complete equation code-skills')
        if root:
            for rel in ['resources/68f08b08211ab2e3/assets/sam3.pt','shared','memory/646cffedcabbf94ae9f31b6b4610c97ca794df1c1be983eb7314fcfdde7c9427.json','programs.json','program-templates.json','physicalrsi_skill_library']:
                path(Path(root)/rel,'equation resource')
        for module in ['torch','pinocchio','timm','ftfy','iopath']:
            if importlib.util.find_spec(module) is None:issues.append('Missing equation module: '+module)
    if task=='plug_in_charger':path(setting('charger_assets_root','ME_BRAIN_CHARGER_ASSETS'),'charger code-skills assets')
    return {'task':task,'source_verified':True,'runtime_checked':True,'issues':issues,
            'limit':'File/module presence only; does not certify GPU compatibility, all transitive assets, simulator or scores'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task')
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    result = audit()
    if args.runtime:
        tasks = [args.task] if args.task else json.loads((ROOT / 'tasks/release.json').read_text())
        result['runtime_checks'] = [check_runtime(task) for task in tasks]
        result['passed'] = result['passed'] and all(not item['issues'] for item in result['runtime_checks'])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
