"""Unit tests for Planner intent classification and task decomposition."""

import pytest
from agent_core.planner import Planner, Intent, TaskPlan


@pytest.fixture
def planner() -> Planner:
    return Planner(llm_adapter=None)


class TestIntentClassification:
    def test_homework_derivatives(self, planner: Planner) -> None:
        intent = planner.classify_intent("explain derivatives to me")
        assert intent == Intent.HOMEWORK

    def test_homework_math(self, planner: Planner) -> None:
        assert planner.classify_intent("solve this math equation") == Intent.HOMEWORK

    def test_homework_science(self, planner: Planner) -> None:
        assert planner.classify_intent("explain how photosynthesis works") == Intent.HOMEWORK

    def test_research_climate(self, planner: Planner) -> None:
        intent = planner.classify_intent("find info about climate change")
        assert intent == Intent.RESEARCH

    def test_research_lookup(self, planner: Planner) -> None:
        assert planner.classify_intent("look up the latest news about AI") == Intent.RESEARCH

    def test_coding_python_function(self, planner: Planner) -> None:
        intent = planner.classify_intent("write a python function to sort a list")
        assert intent == Intent.CODING

    def test_coding_debug(self, planner: Planner) -> None:
        assert planner.classify_intent("debug this python code") == Intent.CODING

    def test_desktop_organize(self, planner: Planner) -> None:
        intent = planner.classify_intent("organize my desktop files")
        assert intent == Intent.DESKTOP

    def test_desktop_clean(self, planner: Planner) -> None:
        assert planner.classify_intent("clean up my desktop") == Intent.DESKTOP

    def test_general_unknown(self, planner: Planner) -> None:
        intent = planner.classify_intent("zxqyw abc123 gibberish nothing here")
        assert intent == Intent.GENERAL

    def test_communication_whatsapp(self, planner: Planner) -> None:
        intent = planner.classify_intent("send a whatsapp message to Alice")
        assert intent == Intent.COMMUNICATION

    def test_file_ops_read(self, planner: Planner) -> None:
        intent = planner.classify_intent("read file documents/notes.txt")
        assert intent == Intent.FILE_OPS


class TestTaskDecomposition:
    def test_decompose_returns_task_plan(self, planner: Planner) -> None:
        plan = planner.decompose("explain derivatives to me")
        assert isinstance(plan, TaskPlan)
        assert plan.intent == Intent.HOMEWORK
        assert len(plan.steps) > 0

    def test_decompose_with_explicit_intent(self, planner: Planner) -> None:
        plan = planner.decompose("anything", intent=Intent.RESEARCH)
        assert plan.intent == Intent.RESEARCH

    def test_decompose_preserves_query(self, planner: Planner) -> None:
        query = "what is the speed of light"
        plan = planner.decompose(query)
        assert plan.original_query == query

    def test_decompose_general_has_steps(self, planner: Planner) -> None:
        plan = planner.decompose("zxqyw gibberish")
        assert len(plan.steps) >= 1
