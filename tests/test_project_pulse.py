"""Tests for ProjectPulse task tracker and burndown chart."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta

import pytest

from tools.project_pulse import (
    ProjectPulse,
    Task,
    TaskStatus,
    _map_notion_status,
    _PulseDB,
)


# ---------------------------------------------------------------------------
# _PulseDB direct tests
# ---------------------------------------------------------------------------

class TestPulseDB:
    def _make_db(self) -> _PulseDB:
        return _PulseDB(tempfile.mktemp(suffix=".db"))

    def test_add_and_get_task(self) -> None:
        db = self._make_db()
        task = Task(id=0, title="Write essay", subject="English")
        new_id = db.add_task(task)
        assert new_id > 0
        tasks = db.get_all()
        assert len(tasks) == 1
        assert tasks[0].title == "Write essay"
        assert tasks[0].id == new_id

    def test_update_status(self) -> None:
        db = self._make_db()
        task = Task(id=0, title="Maths homework")
        task_id = db.add_task(task)
        db.update_status(task_id, TaskStatus.IN_PROGRESS)
        tasks = db.get_all()
        assert tasks[0].status == TaskStatus.IN_PROGRESS

    def test_mark_done_sets_completed_at(self) -> None:
        db = self._make_db()
        task = Task(id=0, title="Done task")
        task_id = db.add_task(task)
        db.update_status(task_id, TaskStatus.DONE)
        tasks = db.get_all()
        assert tasks[0].status == TaskStatus.DONE
        assert tasks[0].completed_at is not None

    def test_get_by_status(self) -> None:
        db = self._make_db()
        db.add_task(Task(id=0, title="A", status=TaskStatus.BACKLOG))
        db.add_task(Task(id=0, title="B", status=TaskStatus.IN_PROGRESS))
        backlog = db.get_by_status(TaskStatus.BACKLOG)
        assert len(backlog) == 1
        assert backlog[0].title == "A"

    def test_upsert_from_notion_inserts_new(self) -> None:
        db = self._make_db()
        task = Task(id=0, title="From Notion", notion_id="notion-123", source="notion")
        db.upsert_from_notion(task)
        all_tasks = db.get_all()
        assert len(all_tasks) == 1
        assert all_tasks[0].notion_id == "notion-123"

    def test_upsert_from_notion_updates_existing(self) -> None:
        db = self._make_db()
        task = Task(id=0, title="Old title", notion_id="notion-456", source="notion")
        db.upsert_from_notion(task)
        task2 = Task(id=0, title="New title", notion_id="notion-456", source="notion")
        db.upsert_from_notion(task2)
        all_tasks = db.get_all()
        assert len(all_tasks) == 1
        assert all_tasks[0].title == "New title"


# ---------------------------------------------------------------------------
# ProjectPulse high-level tests
# ---------------------------------------------------------------------------

class TestProjectPulse:
    def _make_pulse(self) -> ProjectPulse:
        return ProjectPulse(db_path=tempfile.mktemp(suffix=".db"))

    def test_add_task_returns_task_with_id(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("Essay", subject="English")
        assert task.id > 0
        assert task.title == "Essay"
        assert task.status == TaskStatus.BACKLOG

    def test_move_to_in_progress(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("Task A")
        pulse.move_to_in_progress(task.id)
        active = pulse.active_tasks()
        assert any(t.id == task.id and t.status == TaskStatus.IN_PROGRESS for t in active)

    def test_mark_done(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("Task B")
        pulse.mark_done(task.id)
        all_tasks = pulse.all_tasks()
        done = [t for t in all_tasks if t.id == task.id]
        assert len(done) == 1
        assert done[0].status == TaskStatus.DONE

    def test_overdue_tasks(self) -> None:
        pulse = self._make_pulse()
        pulse.add_task("Late", due_date=date.today() - timedelta(days=2))
        pulse.add_task("Future", due_date=date.today() + timedelta(days=5))
        overdue = pulse.overdue_tasks()
        assert len(overdue) == 1
        assert overdue[0].title == "Late"

    def test_overdue_does_not_include_done(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("Late done", due_date=date.today() - timedelta(days=1))
        pulse.mark_done(task.id)
        assert len(pulse.overdue_tasks()) == 0

    def test_task_board_returns_string(self) -> None:
        pulse = self._make_pulse()
        pulse.add_task("A")
        board = pulse.task_board()
        assert isinstance(board, str)
        assert "TASK BOARD" in board
        assert "Backlog" in board

    def test_deadline_summary_no_tasks(self) -> None:
        pulse = self._make_pulse()
        summary = pulse.deadline_summary()
        assert "No active tasks" in summary

    def test_deadline_summary_with_tasks(self) -> None:
        pulse = self._make_pulse()
        pulse.add_task("Upcoming", due_date=date.today() + timedelta(days=1))
        summary = pulse.deadline_summary()
        assert "Upcoming" in summary

    def test_burndown_chart_returns_string(self) -> None:
        pulse = self._make_pulse()
        pulse.add_task("Old task", due_date=date.today() - timedelta(days=7))
        chart = pulse.burndown_chart(weeks=4)
        assert isinstance(chart, str)

    def test_burndown_chart_empty(self) -> None:
        pulse = self._make_pulse()
        chart = pulse.burndown_chart(weeks=2)
        assert "Burndown" in chart

    def test_weekly_summary_returns_string(self) -> None:
        pulse = self._make_pulse()
        pulse.add_task("Current")
        summary = pulse.weekly_summary()
        assert "WEEKLY SUMMARY" in summary

    def test_socratic_prompts_by_subject(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("Derivative", subject="math")
        questions = pulse.socratic_prompts(task)
        assert len(questions) == 3
        assert all(isinstance(q, str) for q in questions)

    def test_socratic_prompts_default_subject(self) -> None:
        pulse = self._make_pulse()
        task = pulse.add_task("General task", subject="")
        questions = pulse.socratic_prompts(task)
        assert len(questions) == 3

    def test_sync_from_notion(self) -> None:
        from tools.notion_integration import Assignment
        from datetime import date

        pulse = self._make_pulse()

        class _FakeAssignment:
            id = "notion-abc"
            title = "Notion task"
            status = "in_progress"
            subject = "Science"
            due_date = date.today() + timedelta(days=4)

        count = pulse.sync_from_notion([_FakeAssignment()])
        assert count == 1
        tasks = pulse.all_tasks()
        assert len(tasks) == 1
        assert tasks[0].source == "notion"
        assert tasks[0].notion_id == "notion-abc"


# ---------------------------------------------------------------------------
# Task model tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_urgency_label_overdue(self) -> None:
        t = Task(id=1, title="x", due_date=date.today() - timedelta(days=3))
        label = t.urgency_label()
        assert "OVERDUE" in label

    def test_urgency_label_today(self) -> None:
        t = Task(id=1, title="x", due_date=date.today())
        assert "TODAY" in t.urgency_label()

    def test_urgency_label_tomorrow(self) -> None:
        t = Task(id=1, title="x", due_date=date.today() + timedelta(days=1))
        assert "TOMORROW" in t.urgency_label()

    def test_urgency_label_done(self) -> None:
        t = Task(id=1, title="x", status=TaskStatus.DONE, due_date=date.today() - timedelta(days=1))
        assert "✅" in t.urgency_label()

    def test_urgency_label_no_date(self) -> None:
        t = Task(id=1, title="x", due_date=None)
        assert t.urgency_label() == ""


# ---------------------------------------------------------------------------
# _map_notion_status tests
# ---------------------------------------------------------------------------

class TestMapNotionStatus:
    def test_done_mappings(self) -> None:
        for s in ("done", "complete", "completed", "finished"):
            assert _map_notion_status(s) == TaskStatus.DONE

    def test_in_progress_mappings(self) -> None:
        for s in ("in_progress", "in progress", "doing", "working"):
            assert _map_notion_status(s) == TaskStatus.IN_PROGRESS

    def test_blocked_mappings(self) -> None:
        for s in ("blocked", "waiting"):
            assert _map_notion_status(s) == TaskStatus.BLOCKED

    def test_unknown_falls_back_to_backlog(self) -> None:
        assert _map_notion_status("some_random_status") == TaskStatus.BACKLOG
