"""Verify optional gateway configuration without issuing network requests."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'capabilities' / 'control'
SERVICE_KEYS = (
    'ME_BRAIN_API_URL', 'ME_BRAIN_API_MODEL', 'ME_BRAIN_API_TOKEN_FILE',
    'ME_BRAIN_API_AUTH_HEADER', 'ME_BRAIN_API_REQUEST_ID_HEADER',
    'ME_BRAIN_API_TOKEN', 'COAP_ASTRA_TOKEN', 'COAP_ASTRA_TOKEN_FILE',
    'COAP_X32_GPT', 'COAP_X33_GPT',
)


def empty_configuration():
    return {key: '' for key in SERVICE_KEYS}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CONTROL / filename)
    module = importlib.util.module_from_spec(spec)
    module.__policy_root__ = str(ROOT)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unique_package():
    name = '_gateway_configuration_' + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(CONTROL)]
    sys.modules[name] = package
    return name


class FakeReply:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, *args):
        return json.dumps(self.value).encode()


class GatewayConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.modules_before = set(sys.modules)

    def tearDown(self):
        for name in set(sys.modules) - self.modules_before:
            if name.startswith('_gateway_configuration_'):
                sys.modules.pop(name, None)

    def unexpected_request(self, *args, **kwargs):
        self.fail('An unconfigured or disabled gateway attempted a request')

    def check_supervisor_without_configuration(self, stem):
        with patch.dict(os.environ, empty_configuration()):
            name = unique_package() + '.' + stem
            module = load_module(name, stem + '.py')
            supervisor = module.Supervisor(opener=self.unexpected_request)
            result = supervisor.review('initial', 0, {}, [], ['hold'])
            self.assertIsNone(supervisor.token_path)
            self.assertEqual(result['error'], 'missing_endpoint')

    def check_supervisor_with_configuration(self, stem):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / 'credential'
            token.write_text('dummy-protocol-credential')
            token.chmod(0o600)
            settings = empty_configuration()
            settings.update(
                ME_BRAIN_API_URL='http://127.0.0.1/responses',
                ME_BRAIN_API_MODEL='configured-model',
                ME_BRAIN_API_TOKEN_FILE=str(token),
                ME_BRAIN_API_AUTH_HEADER='X-Custom-Token',
                ME_BRAIN_API_REQUEST_ID_HEADER='X-Custom-Request',
            )
            with patch.dict(os.environ, settings):
                module = load_module(unique_package() + '.' + stem, stem + '.py')
                calls = []

                def opener(request, **kwargs):
                    calls.append(request)
                    payload = json.loads(request.data)
                    query = json.loads(payload['input'][1]['content'][0]['text'])
                    self.assertEqual(payload['model'], 'configured-model')
                    self.assertEqual(request.headers['Authorization'], 'Bearer dummy-protocol-credential')
                    self.assertEqual(request.headers['X-custom-token'], 'dummy-protocol-credential')
                    self.assertIn('X-custom-request', request.headers)
                    result = dict(
                        request_id=query['request_id'], stage=query['stage'], action='hold',
                        possession='unknown', placement='unknown', confidence=.5, evidence='test',
                    )
                    return FakeReply(dict(status='completed', output_text=json.dumps(result)))

                supervisor = module.Supervisor(opener=opener)
                result = supervisor.review(
                    'initial', 0, {}, [('head', np.zeros((480, 640, 3), np.uint8))], ['hold'],
                )
                self.assertEqual(result['action'], 'hold')
                self.assertEqual(len(calls), 1)
                self.assertEqual(supervisor.records[-1]['status'], 'completed')

    def advisor_modules(self):
        package = unique_package()
        base = load_module(package + '.xylophone_astra', 'xylophone_astra.py')
        temporal = load_module(package + '.xylophone_astra33', 'xylophone_astra33.py')
        return base, temporal

    def check_advisor_without_configuration(self, index):
        with patch.dict(os.environ, empty_configuration()):
            module = self.advisor_modules()[index]
            advisor = module.Advisor(post=self.unexpected_request)
            self.assertFalse(advisor.enabled)
            self.assertEqual(advisor.ask('test', {}, ['wait'], {}), {})
            self.assertEqual(advisor.events, [])
            online = module.Advisor(enabled=True, post=self.unexpected_request)
            self.assertEqual(online.ask('test', {}, ['wait'], {}), {})
            self.assertEqual(online.events[-1]['error_type'], 'missing_gateway_configuration')

    def check_advisor_with_configuration(self, index):
        settings = empty_configuration()
        settings.update(
            ME_BRAIN_API_URL='http://127.0.0.1/responses',
            ME_BRAIN_API_MODEL='configured-model',
            ME_BRAIN_API_AUTH_HEADER='X-Custom-Token',
            ME_BRAIN_API_REQUEST_ID_HEADER='X-Custom-Request',
            COAP_ASTRA_TOKEN='dummy-explicit-credential',
            ME_BRAIN_API_TOKEN='dummy-fallback-credential',
        )
        with patch.dict(os.environ, settings):
            module = self.advisor_modules()[index]
            calls = []

            def post(url, headers, json, **kwargs):
                calls.append(headers)
                self.assertEqual(url, 'http://127.0.0.1/responses')
                self.assertEqual(json['model'], 'configured-model')
                self.assertEqual(headers['Authorization'], 'Bearer dummy-explicit-credential')
                self.assertEqual(headers['X-Custom-Token'], 'dummy-explicit-credential')

                class Reply:
                    status_code = 200

                    def json(self):
                        return dict(status='completed', output_text='{"choice":"wait","diagnosis":"test","rois":{}}')

                return Reply()

            advisor = module.Advisor(enabled=True, post=post)
            vision = {'cam_head': {'color': np.zeros((480, 640, 3), np.uint8)}}
            result = advisor.ask('test', vision, ['wait'], {})
            self.assertEqual(result['choice'], 'wait')
            self.assertEqual(len(calls), 1)

    def test_disabled_identity_checker_needs_no_credential(self):
        with patch.dict(os.environ, empty_configuration()):
            package = unique_package()
            load_module(package + '.ans7_supervisor', 'ans7_supervisor.py')
            module = load_module(package + '.identity_check', 'identity_check.py')
            checker = module.IdentityChecker(opener=self.unexpected_request)
            self.assertIsNone(checker.check(0, [], []))
            self.assertIsNone(checker.token_path)


def add_case(name, method, argument):
    def test(self):
        getattr(self, method)(argument)
    test.__name__ = name
    setattr(GatewayConfigurationTests, name, test)


for stem in ('astra_supervisor', 'ans6_supervisor', 'ans7_supervisor'):
    add_case('test_' + stem + '_without_configuration', 'check_supervisor_without_configuration', stem)
    add_case('test_' + stem + '_configured_protocol', 'check_supervisor_with_configuration', stem)

for index, stem in enumerate(('xylophone_astra', 'xylophone_astra33')):
    add_case('test_' + stem + '_without_configuration', 'check_advisor_without_configuration', index)
    add_case('test_' + stem + '_configured_credential_precedence', 'check_advisor_with_configuration', index)


if __name__ == '__main__':
    unittest.main()
