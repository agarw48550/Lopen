"""Project Pulse: student task tracking with ASCII burndown charts and deadline alerts.

Features:
  - Task management with Backlog → In Progress → Done workflow
  - ASCII burndown chart (no external rendering required)
  - Smart deadline alerts (overdue, due today, due tomorrow, due in N days)
  - Weekly progress summaries
  - Integration with NotionIntegration (optional — also works standalone)
  - Homework tutor integration: Socratic Q&A for tasks in progress
  - Persistent storage via SQLite (storage/project_pulse.db)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task status constants
# ---------------------------------------------------------------------------

class TaskStatus:
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"

    ALL = (BACKLOG, IN_PROGRESS, DONE, BLOCKED)
    ACTIVE = (BACKLOG, IN_PROGRESS, BLOCKED)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: int
    title: str
    status: str = TaskStatus.BACKLOG
    subject: str = ""
    due_date: Optional[date] = None
    created_at: Optional[date] = None
    completed_at: Optional[date] = None
    notes: str = ""
    source: str = "local"           # "local" | "notion"
    notion_id: str = ""

    # ------------------------------------------------------------------
    def days_until_due(self) -> Optional[int]:
        if self.due_date is None:
            return None
        return (self.due_date - date.today()).days

    def is_overdue(self) -> bool:
        d = self.days_until_due()
        return d is not None and d < 0 and self.status != TaskStatus.DONE

    def urgency_label(self) -> str:
        d = self.days_until_due()
        if d is None:
            return ""
        if self.status == TaskStatus.DONE:
            return "✅"
        if d < 0:
            return f"🔴 OVERDUE ({-d}d)"
        if d == 0:
            return "🔴 TODAY"
        if d == 1:
            return "🟡 TOMORROW"
        if d <= 3:
            return f"🟠 {d}d"
        return f"⚪ {d}d"


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class _PulseDB:
    def __init__(self, db_path: str = "storage/project_pulse.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                status TEXT DEFAULT 'backlog',
                subject TEXT DEFAULT '',
                due_date TEXT,
                created_at TEXT,
                completed_at TEXT,
                notes TEXT DEFAULT '',
                source TEXT DEFAULT 'local',
                notion_id TEXT DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def add_task(self, task: Task) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO tasks (title, status, subject, due_date, created_at, notes, source, notion_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.title,
                task.status,
                task.subject,
                task.due_date.isoformat() if task.due_date else None,
                (task.created_at or date.today()).isoformat(),
                task.notes,
                task.source,
                task.notion_id,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_status(self, task_id: int, status: str) -> None:
        completed_at = date.today().isoformat() if status == TaskStatus.DONE else None
        self._conn.execute(
            "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
            (status, completed_at, task_id),
        )
        self._conn.commit()

    def update_notes(self, task_id: int, notes: str) -> None:
        self._conn.execute("UPDATE tasks SET notes = ? WHERE id = ?", (notes, task_id))
        self._conn.commit()

    def get_all(self) -> list[Task]:
        rows = self._conn.execute(
            """
            SELECT id, title, status, subject, due_date, created_at, completed_at, notes, source, notion_id
            FROM tasks ORDER BY due_date ASC NULLS LAST, id ASC
            """
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_by_status(self, status: str) -> list[Task]:
        rows = self._conn.execute(
            """
            SELECT id, title, status, subject, due_date, created_at, completed_at, notes, source, notion_id
            FROM tasks WHERE status = ? ORDER BY due_date ASC NULLS LAST
            """,
            (status,),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def upsert_from_notion(self, task: Task) -> None:
        """Insert or update a task synced from Notion."""
        existing = self._conn.execute(
            "SELECT id FROM tasks WHERE notion_id = ?", (task.notion_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                """
                UPDATE tasks SET title=?, status=?, subject=?, due_date=?, notes=?
                WHERE notion_id=?
                """,
                (
                    task.title,
                    task.status,
                    task.subject,
                    task.due_date.isoformat() if task.due_date else None,
                    task.notes,
                    task.notion_id,
                ),
            )
        else:
            self.add_task(task)
        self._conn.commit()

    @staticmethod
    def _row_to_task(r: tuple) -> Task:
        return Task(
            id=r[0],
            title=r[1],
            status=r[2],
            subject=r[3] or "",
            due_date=date.fromisoformat(r[4]) if r[4] else None,
            created_at=date.fromisoformat(r[5]) if r[5] else None,
            completed_at=date.fromisoformat(r[6]) if r[6] else None,
            notes=r[7] or "",
            source=r[8] or "local",
            notion_id=r[9] or "",
        )


# ---------------------------------------------------------------------------
# Main ProjectPulse class
# ---------------------------------------------------------------------------

class ProjectPulse:
    """Student task tracker with ASCII burndown and deadline management.

    Usage::

        pulse = ProjectPulse()
        pulse.add_task("Write history essay", subject="History", due_date=date(2026, 4, 10))
        print(pulse.burndown_chart())
        print(pulse.deadline_summary())
    """

    def __init__(self, db_path: str = "storage/project_pulse.db") -> None:
        self._db = _PulseDB(db_path)
        logger.info("ProjectPulse initialised (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        subject: str = "",
        due_date: Optional[date] = None,
        notes: str = "",
        source: str = "local",
        notion_id: str = "",
    ) -> Task:
        task = Task(
            id=0,
            title=title,
            status=TaskStatus.BACKLOG,
            subject=subject,
            due_date=due_date,
            created_at=date.today(),
            notes=notes,
            source=source,
            notion_id=notion_id,
        )
        new_id = self._db.add_task(task)
        task.id = new_id
        logger.info("Task added: #%d %r", new_id, title)
        return task

    def move_to_in_progress(self, task_id: int) -> None:
        self._db.update_status(task_id, TaskStatus.IN_PROGRESS)
        logger.info("Task #%d → in_progress", task_id)

    def mark_done(self, task_id: int) -> None:
        self._db.update_status(task_id, TaskStatus.DONE)
        logger.info("Task #%d → done", task_id)

    def mark_blocked(self, task_id: int) -> None:
        self._db.update_status(task_id, TaskStatus.BLOCKED)

    def add_note(self, task_id: int, note: str) -> None:
        self._db.update_notes(task_id, note)

    def sync_from_notion(self, assignments: list[Any]) -> int:
        """Import Assignment objects from NotionIntegration into ProjectPulse."""
        count = 0
        for a in assignments:
            task = Task(
                id=0,
                title=a.title,
                status=_map_notion_status(a.status),
                subject=a.subject,
                due_date=a.due_date,
                notion_id=a.id,
                source="notion",
            )
            self._db.upsert_from_notion(task)
            count += 1
        logger.info("Synced %d tasks from Notion", count)
        return count

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def all_tasks(self) -> list[Task]:
        return self._db.get_all()

    def active_tasks(self) -> list[Task]:
        tasks = self._db.get_all()
        return [t for t in tasks if t.status in TaskStatus.ACTIVE]

    def overdue_tasks(self) -> list[Task]:
        return [t for t in self.all_tasks() if t.is_overdue()]

    def task_board(self) -> str:
        """Return a formatted ASCII Kanban board."""
        all_tasks = self._db.get_all()
        sections = {
            TaskStatus.BACKLOG: [],
            TaskStatus.IN_PROGRESS: [],
            TaskStatus.BLOCKED: [],
            TaskStatus.DONE: [],
        }
        for t in all_tasks:
            sections.get(t.status, sections[TaskStatus.BACKLOG]).append(t)

        lines = ["╔══════════════════ TASK BOARD ══════════════════╗"]
        labels = {
            TaskStatus.BACKLOG: "📋 Backlog",
            TaskStatus.IN_PROGRESS: "🔄 In Progress",
            TaskStatus.BLOCKED: "🚧 Blocked",
            TaskStatus.DONE: "✅ Done",
        }
        for status, label in labels.items():
            lines.append(f"  {label}")
            tasks = sections[status]
            if tasks:
                for t in tasks[:10]:  # show max 10 per section
                    urgency = t.urgency_label()
                    suffix = f" [{urgency}]" if urgency else ""
                    lines.append(f"    #{t.id:03d} {t.title[:40]}{suffix}")
            else:
                lines.append("    (empty)")
            lines.append("")
        lines.append("╚════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def deadline_summary(self) -> str:
        """Return a sorted deadline alert string for all active tasks."""
        tasks = [t for t in self.all_tasks() if t.status != TaskStatus.DONE]
        if not tasks:
            return "✅ No active tasks."

        lines = ["📅 UPCOMING DEADLINES"]
        lines.append("─" * 40)
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.due_date or date.max, t.id),
        )
        for t in sorted_tasks:
            label = t.urgency_label() or "  no date"
            subj = f"[{t.subject}] " if t.subject else ""
            lines.append(f"  {label:20s} {subj}{t.title[:35]}")
        return "\n".join(lines)

    def burndown_chart(self, weeks: int = 4) -> str:
        """Return a simple ASCII burndown chart for the last N weeks.

        Shows remaining (active) task count per week-end date.
        """
        all_tasks = self._db.get_all()
        today = date.today()

        # Build weekly series
        week_labels = []
        remaining_counts = []
        for w in range(weeks):
            week_end = today - timedelta(days=today.weekday()) + timedelta(weeks=w - weeks + 1)
            week_labels.append(week_end.strftime("%m/%d"))
            active = sum(
                1
                for t in all_tasks
                if (
                    t.created_at is not None
                    and t.created_at <= week_end
                    and (t.completed_at is None or t.completed_at > week_end)
                )
            )
            remaining_counts.append(active)

        if not any(remaining_counts):
            return "📊 Burndown: no task history yet.\n   Add tasks with due dates to track progress."

        max_count = max(remaining_counts) or 1
        chart_height = 8
        lines = [f"📊 BURNDOWN CHART (last {weeks} weeks)"]
        lines.append("─" * (weeks * 7 + 4))

        for row in range(chart_height, 0, -1):
            threshold = max_count * row / chart_height
            bar_row = ""
            for count in remaining_counts:
                if count >= threshold:
                    bar_row += "  ██  "
                else:
                    bar_row += "      "
            label = f"{int(threshold):3d}│"
            lines.append(label + bar_row)

        # X-axis
        lines.append("   └" + "──────" * weeks)
        lines.append("    " + "".join(f"{lbl:6s}" for lbl in week_labels))
        lines.append("    (week ending)")
        return "\n".join(lines)

    def weekly_summary(self) -> str:
        """Return a human-readable weekly progress summary."""
        all_tasks = self._db.get_all()
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

        done_this_week = [
            t for t in all_tasks
            if t.completed_at is not None and t.completed_at >= week_start
        ]
        in_progress = [t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]
        overdue = [t for t in all_tasks if t.is_overdue()]

        lines = [
            f"📈 WEEKLY SUMMARY  (week of {week_start.strftime('%d %b %Y')})",
            "─" * 40,
            f"  ✅ Completed this week : {len(done_this_week)}",
            f"  🔄 In progress         : {len(in_progress)}",
            f"  🔴 Overdue             : {len(overdue)}",
        ]
        if overdue:
            lines.append("\n  Overdue tasks:")
            for t in overdue[:5]:
                lines.append(f"    • {t.title[:45]}")
        return "\n".join(lines)

    def socratic_prompts(self, task: Task) -> list[str]:
        """Generate Socratic questions to help a student think through a task."""
        subject_prompts: dict[str, list[str]] = {
            "math": [
                "What formula or theorem applies here?",
                "Can you work through a simpler example first?",
                "What happens if you change one variable?",
            ],
            "science": [
                "What is the underlying principle or law?",
                "How would you test this hypothesis?",
                "What evidence supports your answer?",
            ],
            "history": [
                "What were the causes and effects of this event?",
                "How does this connect to events before and after?",
                "Whose perspective is missing from this account?",
            ],
            "coding": [
                "Can you break this problem into smaller sub-problems?",
                "What edge cases should your solution handle?",
                "How would you test that your solution is correct?",
            ],
            "default": [
                "What do you already know about this topic?",
                "What is the core question being asked here?",
                "How would you check if your answer is correct?",
            ],
        }
        subject_key = task.subject.lower() if task.subject else "default"
        prompts = subject_prompts.get(subject_key, subject_prompts["default"])
        return [
            f"[{task.title[:30]}] {p}"
            for p in prompts
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_notion_status(notion_status: str) -> str:
    """Map a Notion status string to a ProjectPulse TaskStatus constant."""
    mapping = {
        "done": TaskStatus.DONE,
        "complete": TaskStatus.DONE,
        "completed": TaskStatus.DONE,
        "finished": TaskStatus.DONE,
        "in_progress": TaskStatus.IN_PROGRESS,
        "in progress": TaskStatus.IN_PROGRESS,
        "doing": TaskStatus.IN_PROGRESS,
        "working": TaskStatus.IN_PROGRESS,
        "blocked": TaskStatus.BLOCKED,
        "waiting": TaskStatus.BLOCKED,
    }
    return mapping.get(notion_status.lower(), TaskStatus.BACKLOG)
