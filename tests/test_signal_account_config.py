"""Account routing configuration without network access or native accounts."""
import unittest
from unittest import mock

from Cozter import config
from Cozter.backends_bot import create_platforms
from Cozter.tests.helpers import temporary_config


class SignalAccountConfigTests(unittest.TestCase):
    base = {"signal_group_urls": ["https://signal.group/#fixture"],
            "signal_jsonrpc_socket": "tcp://127.0.0.1:7584"}

    def test_configured_account_is_trimmed_and_passed_to_signal(self):
        for account in ("+12025550123", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"):
            with self.subTest(account=account), temporary_config({
                **self.base, "signal_account": "  " + account + "  ",
            }):
                loaded = config.load_config()
                self.assertEqual(loaded["signal_account"], account)
                with mock.patch("Cozter.backends_bot.signal.SignalBot") as constructor:
                    create_platforms(loaded)
                self.assertEqual(constructor.call_args.kwargs["account"], account)

    def test_unconfigured_single_account_daemon_needs_no_identity(self):
        with temporary_config(self.base):
            loaded = config.load_config()
        self.assertEqual(loaded["signal_account"], "")
        with mock.patch("Cozter.backends_bot.signal.SignalBot") as constructor:
            create_platforms(loaded)
        self.assertEqual(constructor.call_args.kwargs["account"], "")

    def test_invalid_account_values_are_not_silently_discarded(self):
        for value in (None, True, False, 123, [], {}):
            with self.subTest(value=value), temporary_config({**self.base, "signal_account": value}):
                with mock.patch("builtins.print") as output, self.assertRaises(SystemExit):
                    config.load_config()
                self.assertIn("signal_account", str(output.call_args))

    def test_account_without_group_and_socket_is_rejected(self):
        with temporary_config({"signal_account": "+12025550123"}):
            with mock.patch("builtins.print") as output, self.assertRaises(SystemExit):
                config.load_config()
            self.assertIn("signal_group_urls", str(output.call_args_list))


if __name__ == "__main__":
    unittest.main()
