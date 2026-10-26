"""Keep task resource contexts canonical without changing upstream packages."""
import json
import os
from pathlib import Path
import re
import unittest


ROOT = Path(os.environ.get('ME_BRAIN_TEST_POLICY_ROOT', Path(__file__).resolve().parents[1])).resolve()
RESOURCES = ROOT / 'resources'
REVISION = re.compile(r'(?:^|[_-])(?:v|r|qv)\d+(?:[_-]\d+)*(?:$|[_-])', re.I)
OPAQUE_CODE = re.compile(r'^(?=.{7,12}$)(?=.*[a-z])(?=.*\d)[a-z\d]+$', re.I)
SEMANTIC_NUMBER = re.compile(r'^(?:joint|link|camera|mesh)\d+$', re.I)


class ResourceLayoutTests(unittest.TestCase):
    def require_resources(self):
        if not RESOURCES.is_dir():
            self.skipTest('Code-only review snapshot: physical resources are not included')

    def test_module_resource_context_belongs_to_its_task(self):
        manifests = sorted((ROOT / 'tasks').glob('*/modules.json'))
        self.assertTrue(manifests, 'No task module manifests found in the selected policy root')
        references = 0
        for filename in manifests:
            task = filename.parent.name
            for record in json.loads(filename.read_text()).values():
                if record.get('namespace'):
                    continue
                relative = Path(record['resource'])
                references += 1
                self.assertFalse(relative.is_absolute())
                self.assertNotIn('..', relative.parts)
                self.assertEqual(relative.parts[:2], ('resources', task))
        self.assertGreater(references, 0)

    def test_every_physical_task_context_has_coap(self):
        self.require_resources()
        tasks = [path for path in RESOURCES.iterdir() if path.is_dir()]
        self.assertTrue(tasks)
        for task in tasks:
            self.assertTrue((task / 'coap').is_dir(), 'Missing canonical context: ' + task.name)

    def test_task_owned_directories_have_no_historical_namespace(self):
        self.require_resources()
        for current, directories, _ in os.walk(RESOURCES, followlinks=False):
            relative = Path(current).relative_to(RESOURCES)
            if 'vendor' in relative.parts:
                directories[:] = []
                continue
            for name in directories:
                if name == 'vendor':
                    continue
                self.assertIsNone(REVISION.search(name), 'Revision directory: ' + str(relative / name))
                self.assertFalse(OPAQUE_CODE.fullmatch(name) and not SEMANTIC_NUMBER.fullmatch(name),
                    'Opaque task namespace: ' + str(relative / name))
                if relative.parts and relative.parts[-1] in {'adapters', 'policy'}:
                    self.assertIn(name, {'coap', 'shared', 'tasks'}, 'Unnormalized wrapper: ' + str(relative / name))

    def test_resource_links_are_relative_and_valid(self):
        self.require_resources()
        links = 0
        for current, directories, filenames in os.walk(RESOURCES, followlinks=False):
            for name in [*directories, *filenames]:
                path = Path(current) / name
                if not path.is_symlink():
                    continue
                links += 1
                label = str(path.relative_to(RESOURCES))
                self.assertFalse(Path(os.readlink(path)).is_absolute(), 'Absolute resource link: ' + label)
                self.assertTrue(path.exists(), 'Broken resource link: ' + label)
        self.assertGreater(links, 0)

    def test_resource_python_contexts_share_capability_inodes(self):
        self.require_resources()
        code = {(path.stat().st_dev, path.stat().st_ino) for path in (ROOT / 'capabilities').rglob('*.py')}
        contexts = list(RESOURCES.rglob('*.py'))
        self.assertEqual(len(contexts), 194)
        for path in contexts:
            self.assertFalse(path.is_symlink())
            identity = (path.stat().st_dev, path.stat().st_ino)
            self.assertIn(identity, code, 'Detached resource code: ' + str(path.relative_to(RESOURCES)))

    def test_external_asset_links_keep_neutral_lexical_bindings(self):
        self.require_resources()
        assets = ROOT / 'assets'
        self.assertTrue(assets.is_dir(), 'Physical deployment is missing its asset pool')
        for tree in (RESOURCES, assets):
            for current, directories, filenames in os.walk(tree, followlinks=False):
                for name in [*directories, *filenames]:
                    path = Path(current) / name
                    if not path.is_symlink():
                        continue
                    target = Path(os.readlink(path))
                    label = str(path.relative_to(ROOT))
                    self.assertFalse(target.is_absolute(), 'Absolute asset binding: ' + label)
                    lexical = Path(os.path.abspath(path.parent / target))
                    if lexical.is_relative_to(ROOT):
                        continue
                    # Preserve the relocatable binding without resolving its machine-specific target.
                    self.assertIn('.me_brain_runtime', lexical.parts, 'External binding bypasses runtime: ' + label)
                    boundary = lexical.parts.index('.me_brain_runtime')
                    self.assertIn('assets', lexical.parts[boundary + 1:], 'External binding bypasses asset pool: ' + label)
                    self.assertTrue(path.exists(), 'Broken external asset binding: ' + label)


if __name__ == '__main__':
    unittest.main()
