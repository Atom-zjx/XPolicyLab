"""Prepare native dependencies while preserving their selected model classes."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from ..core.runtime import ROOT


def _external_dependencies(config, route):
    paths = []
    for template in route.get('dependency_paths', []):
        path = Path(template.format(**config)).expanduser().resolve()
        if path.is_dir():
            paths.append(str(path))
    sys.path[:0] = paths
    return paths


def create(config, route, scope, run):
    if route['canonical_task'] == 'deposit_coin':
        from .coin import create as create_coin
        return create_coin(config, route, scope, run)
    driver = route['driver']
    if driver == 'native':
        os.environ.update(RDJ_OUTPUT_DIR=str(run), RDJ_V1_OUTPUT=str(run / 'runtime'), RDJ_CONFIG=route['canonical_task'], RDJ_SEED=str(route.get('defaults', {}).get('seed', 0)), IK_RESULT=str(run))
        factory = scope.resolve(route['model'])
        return factory() if route.get('constructor') == 'empty' else factory(dict(route.get('defaults', {})))
    if driver == 'charger':
        _prepare_charger(config, run)
        return scope.resolve(route['model'])(config)
    config = dict(config, equation_assets_root=config.get('equation_assets_root') or os.environ.get('ME_BRAIN_EQUATION_ASSETS'))
    _external_dependencies(config, route)
    return EquationModel(config, route, scope, run)


def _prepare_charger(config, run):
    assets = config.get('charger_assets_root') or os.environ.get('ME_BRAIN_CHARGER_ASSETS')
    if not assets:
        raise ValueError('charger_assets_root or ME_BRAIN_CHARGER_ASSETS is required')
    assets = Path(assets).expanduser().resolve(strict=True)
    pinned = json.loads((ROOT / 'tasks/plug_in_charger/dependencies.json').read_text())
    robot_relative = 'implementations/rdj_plug_policy/eb0036595f3b1f25/rdj_plug_policy/assets/x5'
    curobo_relative = 'dependencies/curobo/9077c9afff1c3fda/curobo'
    for relative, expected in pinned.items():
        if hashlib.sha256((assets / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Charger dependency checksum mismatch: ' + relative)
    private = run / 'native'
    private.mkdir()
    for relative, name in [(robot_relative, 'x5'), (curobo_relative, 'curobo')]:
        shutil.copytree(assets / relative, private / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('torch_extensions', 'cache', 'tmp'):
        (private / name).mkdir()
    evidence = run / 'native_evidence'
    evidence.mkdir()
    os.environ.update(B88R42WQ_RSI_ROBOT_ASSETS=str(private / 'x5'), RDJ_RGB_EVIDENCE=str(evidence), RDJ_OUTPUT_DIR=str(private), RDJ_V1_OUTPUT=str(private), TORCH_EXTENSIONS_DIR=str(private / 'torch_extensions'), XDG_CACHE_HOME=str(private / 'cache'), TMPDIR=str(private / 'tmp'))
    sys.path.insert(0, str(private / 'curobo'))


class EquationModel:
    def __init__(self, config, route, scope, run):
        self.process, self.stream, self.core = None, None, None
        atexit.register(self.close)
        assets = config.get('equation_assets_root') or os.environ.get('ME_BRAIN_EQUATION_ASSETS')
        if not assets:
            raise ValueError('equation_assets_root or ME_BRAIN_EQUATION_ASSETS is required')
        config = dict(config, equation_assets_root=assets)
        package = Path(route['package_resource'].format(**config)).expanduser().resolve(strict=True)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        os.environ.update(PHYSICALRSI_PROGRAM_ROOT=str(ROOT / 'resources/solve_equation'), RDJ_OUTPUT_DIR=str(run), RDJ_V1_OUTPUT=str(run / 'runtime'), RDJ_CONFIG='solve_equation', RDJ_SEED=str(config.get('seed', 0)), RDJ_EQUATION_PACKAGE_ROOT=str(package), EQUATION_RESULTS=str(run / 'equation'), ROBODOJO_SAM3_URL='http://127.0.0.1:' + str(port))
        bypass = ','.join(dict.fromkeys(['127.0.0.1', 'localhost', '::1'] + os.environ.get('NO_PROXY', '').split(',')))
        os.environ.update(NO_PROXY=bypass, no_proxy=bypass)
        environment = dict(os.environ, ME_BRAIN_EQUATION_DEPENDENCIES=json.dumps(route.get('dependency_paths', [])), ME_BRAIN_EQUATION_ASSETS=str(assets))
        self.stream = (run / 'equation_sam.log').open('w')
        command = [sys.executable, '-u', '-m', __package__.split('.backends')[0] + '.services', '--task', 'solve_equation', '--module', route['service'], '--host', '127.0.0.1', '--port', str(port)]
        try:
            self.process = subprocess.Popen(command, cwd=package, env=environment, stdout=self.stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + float(config.get('equation_service_timeout', 1200))
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('Equation SAM service exited; see ' + str(run / 'equation_sam.log'))
                try:
                    with urllib.request.urlopen(os.environ['ROBODOJO_SAM3_URL'] + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Equation SAM service startup timed out')
                    time.sleep(.2)
            self.core = scope.resolve(route['model'])(config)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self.core, name)

    def close(self):
        core = self.core
        if core is not None and getattr(core, 'pool', None) is not None:
            core.pool.shutdown(wait=False)
        self.core = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.stream is not None:
            self.stream.close()
