"""Regression coverage for shared code with task-local execution semantics."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
PACKAGE = 'XPolicyLab.policy.' + ROOT.name
routing = importlib.import_module(PACKAGE + '.routing')
modules = importlib.import_module(PACKAGE + '.core.modules')


class AdapterContractTests(unittest.TestCase):
    def test_all_release_tasks_keep_their_protocol(self):
        release = json.loads((ROOT / 'tasks/release.json').read_text())
        self.assertEqual(len(release), 54)
        for task in release:
            config, route = routing.configure(dict(task_name=task, env_cfg_type='arx_x5', action_type='auto'))
            self.assertEqual(config['action_type'], route['action_type'])
            self.assertEqual(config['eval_batch'], route['eval_batch'])
        self.assertEqual(routing.resolve('sort_nesting_dolls_by_size_random')['canonical_task'], 'sort_nesting_dolls_by_size')
        self.assertEqual(routing.resolve('unregistered_custom_task')['driver'], 'vla')
        with self.assertRaises(ValueError):
            routing.configure(dict(task_name='press_by_number', env_cfg_type='arx_x5', action_type='joint'))

    def test_registry_has_no_author_or_revision_dispatch(self):
        for route in routing.REGISTRY.values():
            self.assertFalse({'family', 'version', 'source', 'author', 'owner'} & route.keys())

    def test_all_implementation_bytes_match_the_extraction(self):
        for filename in ROOT.glob('tasks/*/modules.json'):
            for record in json.loads(filename.read_text()).values():
                if not record.get('namespace'):
                    self.assertEqual(hashlib.sha256((ROOT / record['file']).read_bytes()).hexdigest(), record['sha256'])

    def test_identical_code_keeps_separate_task_globals(self):
        first = modules.TaskModules('play_Xylophone')
        second = modules.TaskModules('pour_liquid_into_cup')
        left = first.import_module('coap.geometry')
        right = second.import_module('coap.geometry')
        self.assertIsNot(left, right)
        self.assertNotEqual(left.__package__, right.__package__)
        left._isolation_probe = object()
        self.assertFalse(hasattr(right, '_isolation_probe'))
        self.assertIs(first.importlib.import_module('coap.geometry'), left)
        self.assertNotIn('coap.geometry', sys.modules)

    def test_distinct_perception_implementations_stay_distinct(self):
        first = modules.TaskModules('press_by_number').import_module('b88r42wq.perception')
        second = modules.TaskModules('general_pickup').import_module('coap.perception')
        self.assertNotEqual(first.__file__, second.__file__)
        self.assertTrue(hasattr(first, 'recognize_digits'))
        self.assertTrue(hasattr(second, 'description'))

    def test_resource_resolution_preserves_adjacent_geometry(self):
        scope = modules.TaskModules('insert_key')
        model = scope.import_module('insert_key.model')
        resource = Path(model.__resource_file__).resolve()
        self.assertTrue(resource.with_name('wrist_calibration.json').is_file())
        self.assertEqual(model.WRIST_CAL, json.loads(resource.with_name('wrist_calibration.json').read_text()))

    def test_source_introspection_keeps_the_original_imports(self):
        scope = modules.TaskModules('imitate_sorting_sequence')
        module = scope.import_module('coap.tasks.sorting_demo_fragments')
        source = inspect.getsource(module.prepare_core)
        self.assertIn("node.module == 'memory_initial'", source)
        self.assertIn('inspect.getsource(replay)', source)

    def test_dynamic_perception_module_is_bundled(self):
        scope = modules.TaskModules('classify_objects_by_language')
        name = 'coap.tasks.classify_r32_language_source_material_hexa'
        self.assertIn(name, scope.modules)
        self.assertIsNotNone(scope.importlib.import_module(name))

    def test_absolute_checkpoint_keeps_cli_precedence(self):
        model = importlib.import_module(PACKAGE + '.vla_model')
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'configured', Path(directory) / 'requested'
            for path in (first, second):
                path.mkdir()
                (path / 'model.safetensors').touch()
            result = model._resolve_checkpoint(dict(model_path=str(first), ckpt_name=str(second)))
            self.assertEqual(result, second)

    def test_named_checkpoint_keeps_explicit_path_precedence(self):
        model = importlib.import_module(PACKAGE + '.vla_model')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / 'model.safetensors').touch()
            result = model._resolve_checkpoint(dict(model_path=str(checkpoint), ckpt_name='named-checkpoint'))
            self.assertEqual(result, checkpoint)


if __name__ == '__main__':
    unittest.main()
