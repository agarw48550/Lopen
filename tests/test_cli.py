"""Tests for the interactive CLI module."""

from __future__ import annotations

import sys
import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestCLIImport(unittest.TestCase):
    """Verify the CLI module imports cleanly."""

    def test_import_cli(self) -> None:
        import cli  # noqa: F401
        self.assertTrue(hasattr(cli, "CLIState"))
        self.assertTrue(hasattr(cli, "repl"))
        self.assertTrue(hasattr(cli, "main"))

    def test_cli_state_defaults(self) -> None:
        from cli import CLIState
        state = CLIState(host="localhost", port=8000, debug=False)
        self.assertEqual(state.host, "localhost")
        self.assertEqual(state.port, 8000)
        self.assertFalse(state.debug)
        self.assertEqual(state.base_url, "http://localhost:8000")

    def test_cli_state_debug_on(self) -> None:
        from cli import CLIState
        state = CLIState(host="localhost", port=8000, debug=True)
        self.assertTrue(state.debug)

    def test_session_id_format(self) -> None:
        from cli import CLIState
        state = CLIState("localhost", 8000, False)
        self.assertTrue(state.session_id.startswith("cli-"))

    def test_colour_helpers(self) -> None:
        from cli import cyan, green, yellow, red, bold, dim
        # These should return at least the original text
        self.assertIn("hello", cyan("hello"))
        self.assertIn("world", green("world"))
        self.assertIn("warn", yellow("warn"))
        self.assertIn("err", red("err"))
        self.assertIn("bold", bold("bold"))
        self.assertIn("dim", dim("dim"))


class TestCLIEasterEggs(unittest.TestCase):
    """Verify easter egg detection logic."""

    def _make_state(self) -> "object":
        from cli import CLIState
        return CLIState("localhost", 8000, False)

    def test_easter_egg_joke(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        buf = StringIO()
        with patch("sys.stdout", buf):
            result = _easter_egg(state, "", "joke")
        self.assertTrue(result)

    def test_easter_egg_haiku(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "haiku")
        self.assertTrue(result)

    def test_easter_egg_about(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "about")
        self.assertTrue(result)

    def test_easter_egg_sing(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "sing")
        self.assertTrue(result)

    def test_easter_egg_quote(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "quote")
        self.assertTrue(result)

    def test_no_easter_egg_for_normal_input(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "what is the weather today")
        self.assertFalse(result)

    def test_no_easter_egg_for_help(self) -> None:
        from cli import _easter_egg
        state = self._make_state()
        result = _easter_egg(state, "", "help")
        self.assertFalse(result)


class TestCLIOfflineBehavior(unittest.TestCase):
    """Verify CLI handles offline orchestrator gracefully."""

    def _make_state(self) -> "object":
        from cli import CLIState
        return CLIState("localhost", 19999, False)  # port 19999 should not be running

    def test_get_returns_none_when_offline(self) -> None:
        from cli import _get
        state = self._make_state()
        result = _get(state, "/health", timeout=1.0)
        self.assertIsNone(result)

    def test_post_returns_none_when_offline(self) -> None:
        from cli import _post
        state = self._make_state()
        result = _post(state, "/chat", {"message": "hello"}, timeout=1.0)
        self.assertIsNone(result)

    def test_cmd_chat_offline_does_not_raise(self) -> None:
        from cli import _cmd_chat, CLIState
        state = CLIState("localhost", 19999, False)
        # Should print a message and return gracefully
        _cmd_chat(state, "hello world")

    def test_cmd_status_offline_does_not_raise(self) -> None:
        from cli import _cmd_status, CLIState
        state = CLIState("localhost", 19999, False)
        _cmd_status(state, "")

    def test_cmd_plugins_offline_does_not_raise(self) -> None:
        from cli import _cmd_plugins, CLIState
        state = CLIState("localhost", 19999, False)
        _cmd_plugins(state, "")

    def test_cmd_history_offline_does_not_raise(self) -> None:
        from cli import _cmd_history, CLIState
        state = CLIState("localhost", 19999, False)
        _cmd_history(state, "")


class TestCLICommandDispatch(unittest.TestCase):
    """Verify command dispatching."""

    def test_dispatch_table_has_required_commands(self) -> None:
        from cli import _DISPATCH
        required = {"help", "status", "plugins", "history", "config", "debug", "benchmark", "chat"}
        for cmd in required:
            self.assertIn(cmd, _DISPATCH, f"Command '{cmd}' missing from dispatch table")

    def test_quit_commands_set(self) -> None:
        from cli import _QUIT_CMDS
        self.assertIn("quit", _QUIT_CMDS)
        self.assertIn("exit", _QUIT_CMDS)
        self.assertIn("q", _QUIT_CMDS)

    def test_debug_toggle_on(self) -> None:
        from cli import _cmd_debug, CLIState
        state = CLIState("localhost", 8000, False)
        _cmd_debug(state, "on")
        self.assertTrue(state.debug)

    def test_debug_toggle_off(self) -> None:
        from cli import _cmd_debug, CLIState
        state = CLIState("localhost", 8000, True)
        _cmd_debug(state, "off")
        self.assertFalse(state.debug)

    def test_cmd_help_does_not_raise(self) -> None:
        from cli import _cmd_help, CLIState
        state = CLIState("localhost", 8000, False)
        _cmd_help(state, "")

    def test_cmd_config_offline_does_not_raise(self) -> None:
        from cli import _cmd_config, CLIState
        state = CLIState("localhost", 19999, False)
        _cmd_config(state, "")


class TestBannerAndTaglines(unittest.TestCase):
    """Verify banner text constants are populated."""

    def test_banner_non_empty(self) -> None:
        from cli import _BANNER
        self.assertTrue(len(_BANNER.strip()) > 0)

    def test_taglines_non_empty(self) -> None:
        from cli import _TAGLINES
        self.assertGreater(len(_TAGLINES), 0)
        for t in _TAGLINES:
            self.assertIsInstance(t, str)
            self.assertGreater(len(t), 10)

    def test_jokes_list(self) -> None:
        from cli import _JOKES
        self.assertGreater(len(_JOKES), 2)

    def test_haiku_format(self) -> None:
        from cli import _HAIKU
        self.assertGreater(len(_HAIKU), 0)
        for h in _HAIKU:
            self.assertEqual(len(h), 3)


if __name__ == "__main__":
    unittest.main()
