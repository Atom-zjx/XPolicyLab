"""Verify configuration resolution and relocatable asset bindings."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parents[2]))
PACKAGE = 'XPolicyLab.policy.' + ROOT.name
paths = importlib.import_module(PACKAGE + '.core.paths')
runtime = importlib.import_module(PACKAGE + '.core.runtime')


class RelativePathTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'policy'
        self.root.mkdir()
        self.stack = []
        for target, name, value in [(paths, 'ROOT', self.root), (runtime, 'ROOT', self.root)]:
            manager = patch.object(target, name, value)
            manager.start()
            self.addCleanup(manager.stop)
        manager = patch.dict(os.environ, {}, clear=True)
        manager.start()
        self.addCleanup(manager.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_relative_configuration_ignores_current_directory(self):
        self.write('config.json', {'common': {'config': {'model_path': 'model_assets/model'}}})
        os.environ['ME_BRAIN_RUNTIME_CONFIG'] = 'config.json'
        original = Path.cwd()
        try:
            os.chdir(self.root.parent)
            result = runtime.apply({'task_name': 'press_by_number'})
        finally:
            os.chdir(original)
        self.assertEqual(Path(result['model_path']), self.root / 'model_assets/model')

    def test_include_base_and_override_order(self):
        self.write('private/base.json', {'common': {'config': {'tokenizer_path': 'model_assets/tokenizer', 'history_mode': 'old'}}})
        self.write('config.json', {'includes': ['private/base.json'], 'common': {'config': {'history_mode': 'current'}}, 'tasks': {'press_by_number': {'config': {'max_token_len': 400}}}})
        os.environ['ME_BRAIN_RUNTIME_CONFIG'] = 'config.json'
        result = runtime.apply({'task_name': 'press_by_number'})
        self.assertEqual(result['history_mode'], 'current')
        self.assertEqual(result['max_token_len'], 400)
        self.assertEqual(Path(result['tokenizer_path']), self.root / 'model_assets/tokenizer')

    def test_environment_resource_path_is_policy_relative(self):
        self.write('config.json', {'common': {'environment': {'CLASSIFY_CLIP_PATH': 'model_assets/clip', 'ME_BRAIN_API_URL': 'https://example.invalid/api'}}})
        os.environ['ME_BRAIN_RUNTIME_CONFIG'] = 'config.json'
        runtime.apply({'task_name': 'classify_objects'})
        self.assertEqual(Path(os.environ['CLASSIFY_CLIP_PATH']), self.root / 'model_assets/clip')
        self.assertEqual(os.environ['ME_BRAIN_API_URL'], 'https://example.invalid/api')

    def test_include_cycle_is_rejected(self):
        self.write('first.json', {'includes': ['second.json']})
        self.write('second.json', {'includes': ['first.json']})
        os.environ['ME_BRAIN_RUNTIME_CONFIG'] = 'first.json'
        with self.assertRaisesRegex(ValueError, 'cycle'):
            runtime.apply({'task_name': 'press_by_number'})

    def test_private_include_cannot_change_task_routing(self):
        self.write('private.json', {'common': {'config': {'action_type': 'joint'}}})
        self.write('config.json', {'includes': ['private.json']})
        os.environ['ME_BRAIN_RUNTIME_CONFIG'] = 'config.json'
        with self.assertRaisesRegex(ValueError, 'routing'):
            runtime.apply({'task_name': 'press_by_number'})

    def test_source_asset_link_stays_relative_and_is_idempotent(self):
        external = self.root.parent / 'private_assets' / 'encoder.bin'
        external.parent.mkdir()
        external.write_bytes(b'fixture-data')
        config = {'source_assets': {'coap/assets/encoder.bin': '../private_assets/encoder.bin'}, 'strategy_evidence_dir': '../runs'}
        route = {'canonical_task': 'classify_objects'}
        runtime.prepare(config, route)
        target = self.root / 'resources/classify_objects/coap/assets/encoder.bin'
        self.assertTrue(target.is_symlink())
        self.assertFalse(Path(os.readlink(target)).is_absolute())
        self.assertTrue(target.samefile(external))
        runtime.prepare(config, route)
        self.assertTrue(target.samefile(external))

    def test_runtime_configuration_uses_no_absolute_path_literals(self):
        filename = ROOT / 'runtime_config.local.json'
        if not filename.exists():
            self.skipTest('A machine configuration is optional')
        def check(value):
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertFalse(value.startswith('/'))
        check(json.loads(filename.read_text()))

    def test_cad_and_usd_geometry_metadata_remains_byte_identical(self):
        identifiers = {
            'acf66e8d0ab20408efa72449d62aad0108919635d84c1627997f728c42b37288',
            'fd8d0deeb70ea6bf76039b12767e8061bcd53f9cfec474ee2dc975fd50e18a68',
            'c46f47d0137dcad6b701d57505b53cf57264d6e48f9c74fe74c030491ee3d894',
            '9633d615aa97564c23c8aa5c9ecf3253d120297979858acc047db0458af18e88',
        }
        for identifier in identifiers:
            path = ROOT / 'assets' / identifier[:2] / (identifier + '.json')
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identifier)


if __name__ == '__main__':
    unittest.main()
