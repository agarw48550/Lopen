"""Unit tests for ArgumentComposer argument extraction."""

from __future__ import annotations

import pytest
from agent_core.argument_composer import ArgumentComposer


@pytest.fixture
def composer() -> ArgumentComposer:
    return ArgumentComposer(llm_adapter=None)


class TestArgumentComposer:
    def test_always_returns_query(self, composer: ArgumentComposer) -> None:
        args = composer.compose("some query", "any_tool")
        assert "query" in args
        assert args["query"] == "some query"

    def test_extracts_absolute_file_path(self, composer: ArgumentComposer) -> None:
        args = composer.compose("read the file /home/user/documents/notes.txt", "file_ops")
        assert "file_path" in args
        assert args["file_path"] == "/home/user/documents/notes.txt"

    def test_extracts_url(self, composer: ArgumentComposer) -> None:
        args = composer.compose("open https://example.com/page in the browser", "browser_automation")
        assert "url" in args
        assert args["url"] == "https://example.com/page"

    def test_extracts_programming_language(self, composer: ArgumentComposer) -> None:
        args = composer.compose("write a Python function to parse JSON", "coder_assist")
        assert "language" in args
        assert args["language"].lower() == "python"

    def test_extracts_language_javascript(self, composer: ArgumentComposer) -> None:
        args = composer.compose("debug this JavaScript code", "coder_assist")
        assert "language" in args
        assert args["language"].lower() == "javascript"

    def test_extracts_code_snippet(self, composer: ArgumentComposer) -> None:
        query = "fix this code:\n```python\ndef foo():\n    pass\n```"
        args = composer.compose(query, "coder_assist")
        assert "code_snippet" in args
        assert "def foo" in args["code_snippet"]

    def test_no_false_positives_on_plain_query(self, composer: ArgumentComposer) -> None:
        args = composer.compose("explain quantum entanglement", "homework_tutor")
        assert "query" in args
        assert "file_path" not in args
        assert "url" not in args

    def test_researcher_extracts_quoted_term(self, composer: ArgumentComposer) -> None:
        args = composer.compose('search for "machine learning basics"', "researcher")
        assert "search_term" in args
        assert "machine learning" in args["search_term"]

    def test_empty_query(self, composer: ArgumentComposer) -> None:
        args = composer.compose("", "any_tool")
        assert args["query"] == ""

    def test_no_tool_name(self, composer: ArgumentComposer) -> None:
        args = composer.compose("find the file /tmp/test.py")
        assert "file_path" in args
