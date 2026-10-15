"""Redirected plugin output remains UTF-8 on Windows and other locales."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Cozter.agent_tools.base import AgentTool


_CLI_SOURCE = """
import sys
sys.argv[0] = '-m'
from Cozter.agent_tools.base import AgentTool
class FixtureTool(AgentTool):
    name = 'unicode_fixture'
    description = 'Unicode description: \\uac80\\uc0ac \\U0001f642'
    parameters = {'type': 'object'}
    async def run(self, workspace_path, args):
        if args.get('fail'):
            raise RuntimeError(args['text'])
        return args.get('text', '')
FixtureTool.run_as_script()
"""


class PluginCliUnicodeTests(unittest.TestCase):
    def run_cli(self, args):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        # Reproduce an ordinary Windows redirected stream even on UTF-8 CI.
        env['PYTHONIOENCODING'] = 'cp1252'
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [sys.executable, '-B', '-c', _CLI_SOURCE, args],
                cwd=directory, env=env, capture_output=True, timeout=10,
            )

    def test_unicode_result_survives_redirected_legacy_encoding(self):
        text = '검사/파일-🙂-e\u0301.txt'
        result = self.run_cli(json.dumps({'text': text}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode('utf-8').rstrip('\r\n'), text)
        self.assertEqual(result.stderr, b'')

    def test_unicode_help_schema_uses_utf8(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['description'], 'Unicode description: 검사 🙂')

    def test_unicode_error_retains_exit_code_and_stderr(self):
        result = self.run_cli(json.dumps({'fail': True, 'text': '실패 🙂'}))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b'')
        self.assertEqual(
            result.stderr.decode('utf-8').rstrip('\r\n'),
            'Error: tool FixtureTool failed: 실패 🙂',
        )

    def test_stderr_keeps_its_backslashreplace_policy(self):
        result = self.run_cli(json.dumps({'fail': True, 'text': 'bad surrogate \ud800'}))
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'bad surrogate \\ud800', result.stderr)
        self.assertNotIn(b'UnicodeEncodeError', result.stderr)

    def test_embedded_text_streams_need_no_binary_or_reconfigure_api(self):
        class FixtureTool(AgentTool):
            async def run(self, workspace_path, args):
                return '검사 🙂'

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', ['fixture']), redirect_stdout(stdout), redirect_stderr(stderr):
            FixtureTool.run_as_script()
        self.assertEqual(stdout.getvalue(), '검사 🙂\n')
        self.assertEqual(stderr.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
