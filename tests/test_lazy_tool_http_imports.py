"""Cold CLI imports retain the full tool surface without loading HTTP eagerly."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class LazyToolHttpImportTests(unittest.TestCase):
    def run_fresh(self, source):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-c", textwrap.dedent(source)],
                cwd=directory, env=env, capture_output=True, text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_cli_mode_keeps_all_builtins_schemas_dispatch_and_permissions(self):
        self.run_fresh("""
            import asyncio
            import pathlib
            import sys
            sys.argv[0] = '-m'
            from Cozter import agent_tools as tools
            expected = ('apply_patch', 'tree', 'bash', 'copy_file', 'delete_file',
                        'edit_file', 'glob', 'grep', 'list_dir', 'make_dir',
                        'move_file', 'multi_edit', 'read_file', 'read_files',
                        'web_fetch', 'web_search', 'write_file')
            assert tuple(tools._BY_NAME) == expected
            assert [row['function'] for row in tools.TOOL_SCHEMA] == [tool.schema for tool in tools._TOOLS]
            assert not any(tool.is_plugin for tool in tools._TOOLS)
            assert 'aiohttp' not in sys.modules
            pathlib.Path('fixture.txt').write_text('synthetic evidence', encoding='utf-8')
            async def run():
                result = await tools.execute_tool('read_file', {'path': 'fixture.txt'}, '.', 'confirm', lambda event: None)
                assert 'synthetic evidence' in result
                result = await tools.execute_tool('write_file', {'path': 'blocked.txt', 'content': 'x'}, '.', 'confirm', lambda event: None)
                assert not pathlib.Path('blocked.txt').exists()
                assert isinstance(result, str)
            asyncio.run(run())
            assert 'aiohttp' not in sys.modules
        """)

    def test_plugin_help_keeps_required_imports_and_calculator_execution_stays_cold(self):
        for plugin in ('calculator', 'current_time', 'git_info', 'git_ops', 'git_sync', 'memory', 'notes'):
            with self.subTest(plugin=plugin):
                output = self.run_fresh(f"""
                    import runpy
                    import sys
                    sys.argv[:] = ['-m', '--help']
                    runpy.run_module('Cozter.agent_tools.plugins.{plugin}', run_name='__main__')
                    # Git helpers share the lightweight process launcher and
                    # do not initialize the provider registry or HTTP stack.
                    assert 'aiohttp' not in sys.modules
                """)
                self.assertEqual(json.loads(output)["name"], plugin)
        output = self.run_fresh("""
            import runpy
            import sys
            sys.argv[:] = ['-m', '{"expression":"1+1"}']
            runpy.run_module('Cozter.agent_tools.plugins.calculator', run_name='__main__')
            assert 'aiohttp' not in sys.modules
        """)
        self.assertEqual(output.strip(), "2")

    def test_unused_worker_pool_and_url_validation_need_no_http_import(self):
        self.run_fresh("""
            import asyncio
            import sys
            sys.argv[0] = '-m'
            from Cozter.agent_tools import base
            assert base.validate_public_url('https://example.com/path') is None
            assert base.validate_public_url('http://127.0.0.1/private') is not None
            async def run():
                async with base.public_http_pool_scope():
                    assert base._PUBLIC_HTTP_POOL.get()._connector is None
            asyncio.run(run())
            assert 'aiohttp' not in sys.modules
            assert {'aiohttp', 'AbstractResolver', 'DefaultResolver', 'PublicResolver'} <= set(dir(base))
            assert 'aiohttp' not in sys.modules
            assert not hasattr(base, 'not_an_http_export')
        """)

    def test_annotations_and_public_exports_are_the_real_stable_types(self):
        self.run_fresh("""
            from concurrent.futures import ThreadPoolExecutor
            from collections.abc import AsyncIterator
            import inspect
            import sys
            import threading
            import typing
            sys.argv[0] = '-m'
            from Cozter.agent_tools import base
            from Cozter.agent_tools.builtin import web_fetch
            assert 'aiohttp' not in sys.modules
            barrier = threading.Barrier(8)
            def first_use(index):
                barrier.wait()
                return base.PublicResolver, base.AbstractResolver, base.DefaultResolver, base.aiohttp
            with ThreadPoolExecutor(max_workers=8) as workers:
                values = list(workers.map(first_use, range(8)))
            assert all(all(a is b for a, b in zip(value, values[0])) for value in values)
            import aiohttp
            from aiohttp.abc import AbstractResolver
            from aiohttp.resolver import DefaultResolver
            from Cozter.agent_tools.base import PublicResolver
            assert base.aiohttp is aiohttp
            assert base.AbstractResolver is AbstractResolver
            assert base.DefaultResolver is DefaultResolver
            assert base.PublicResolver is PublicResolver
            assert issubclass(PublicResolver, AbstractResolver)
            assert not inspect.isabstract(PublicResolver)
            assert typing.get_type_hints(base.read_bounded_text)['resp'] is aiohttp.ClientResponse
            assert typing.get_type_hints(base.open_http_response)['return'] == AsyncIterator[aiohttp.ClientResponse]
            assert typing.get_type_hints(base.open_public_http_session)['return'] == AsyncIterator[aiohttp.ClientSession]
            assert typing.get_type_hints(base._PublicHttpPool.connector)['return'] is aiohttp.TCPConnector
            assert typing.get_type_hints(web_fetch._fetch_following_redirects)['session'] is aiohttp.ClientSession
            namespace = {}
            exec('from Cozter.agent_tools.base import *', namespace)
            for name in ('aiohttp', 'AbstractResolver', 'DefaultResolver', 'PublicResolver', 'AgentTool', 'NonPublicAddressError'):
                assert namespace[name] is getattr(base, name)
        """)

    def test_type_introspection_itself_can_be_the_first_http_use(self):
        self.run_fresh("""
            import sys
            import typing
            sys.argv[0] = '-m'
            from Cozter.agent_tools import base
            assert 'aiohttp' not in sys.modules
            hint = typing.get_type_hints(base.read_bounded_text)['resp']
            import aiohttp
            assert hint is aiohttp.ClientResponse
            namespace = {}
            exec('from Cozter.agent_tools.base import *', namespace)
            assert namespace['aiohttp'] is aiohttp
        """)
