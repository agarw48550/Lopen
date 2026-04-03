"""Smoke tests: verify all modules can be imported without external services."""

import pytest


def test_import_agent_core_planner() -> None:
    from agent_core.planner import Planner, Intent
    assert Intent.HOMEWORK is not None


def test_import_agent_core_router() -> None:
    from agent_core.router import Router
    r = Router()
    assert r is not None


def test_import_agent_core_tool_registry() -> None:
    from agent_core.tool_registry import ToolRegistry, ToolMeta
    reg = ToolRegistry()
    assert len(reg) == 0


def test_import_agent_core_permissions() -> None:
    from agent_core.permissions import PermissionLevel, check_permission
    assert check_permission("read", PermissionLevel.LOW)


def test_import_agent_core_task_queue() -> None:
    from agent_core.task_queue import TaskQueue, Task, TaskStatus
    q = TaskQueue()
    assert q.size == 0


def test_import_agent_core_memory() -> None:
    from agent_core.memory import ConversationMemory
    mem = ConversationMemory()
    assert mem.turn_count == 0


def test_import_llm_adapter() -> None:
    from llm.llm_adapter import LLMAdapter
    llm = LLMAdapter()
    assert llm.mode == "mock"


def test_import_storage_database() -> None:
    from storage.database import SQLiteDB
    assert SQLiteDB is not None


def test_import_storage_vector_cache() -> None:
    from storage.vector_cache import VectorCache
    vc = VectorCache()
    assert vc.size == 0


def test_import_tools_base() -> None:
    from tools.base_tool import BaseTool
    assert BaseTool is not None


def test_import_tools_homework() -> None:
    from tools.homework_tutor import HomeworkTutor
    t = HomeworkTutor()
    assert t.name == "homework_tutor"


def test_import_tools_researcher() -> None:
    from tools.researcher import Researcher
    t = Researcher()
    assert t.name == "researcher"


def test_import_tools_coder_assist() -> None:
    from tools.coder_assist import CoderAssist
    t = CoderAssist()
    assert t.name == "coder_assist"


def test_import_tools_desktop_organizer() -> None:
    from tools.desktop_organizer import DesktopOrganizer
    t = DesktopOrganizer()
    assert t.name == "desktop_organizer"


def test_import_tools_file_ops() -> None:
    from tools.file_ops import FileOps
    t = FileOps()
    assert t.name == "file_ops"


def test_import_system_health_ram_watchdog() -> None:
    from system_health.ram_watchdog import RamWatchdog
    w = RamWatchdog()
    assert w is not None


def test_import_system_health_disk_check() -> None:
    from system_health.disk_check import DiskCheck
    d = DiskCheck()
    assert d is not None


def test_import_system_health_log_rotation() -> None:
    from system_health.log_rotation import LogRotation
    lr = LogRotation()
    assert lr is not None


def test_import_system_health_cache_cleanup() -> None:
    from system_health.cache_cleanup import CacheCleanup
    cc = CacheCleanup()
    assert cc is not None


def test_import_system_health_heartbeat() -> None:
    from system_health.heartbeat import Heartbeat
    h = Heartbeat()
    assert h is not None


def test_import_voice_wake_word() -> None:
    from interfaces.voice_service.wake_word import WakeWordDetector
    w = WakeWordDetector()
    assert w is not None


def test_import_voice_asr_adapter() -> None:
    from interfaces.voice_service.asr_adapter import ASRAdapter
    a = ASRAdapter()
    assert a.is_mock is True  # no binary in CI


def test_import_voice_tts_adapter() -> None:
    from interfaces.voice_service.tts_adapter import TTSAdapter
    t = TTSAdapter()
    assert t.mode in ("piper", "say", "mock")


def test_import_whatsapp_bridge() -> None:
    from interfaces.whatsapp_service.bridge import WhatsAppBridge
    b = WhatsAppBridge()
    assert b is not None


def test_import_web_dashboard_app() -> None:
    from interfaces.web_dashboard.app import create_dashboard_app
    app = create_dashboard_app()
    assert app is not None


def test_import_orchestrator() -> None:
    from orchestrator import app
    assert app is not None
