"""Unit tests for config loading."""

import pytest
from pathlib import Path
import yaml


CONFIG_DIR = Path(__file__).parent.parent / "config"


class TestSettingsYaml:
    def test_settings_file_exists(self) -> None:
        assert (CONFIG_DIR / "settings.yaml").is_file()

    def test_settings_loads(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_top_level_keys(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        for key in ("lopen", "llm", "memory", "voice", "whatsapp", "web_dashboard", "orchestrator", "health"):
            assert key in cfg, f"Missing top-level key: {key}"

    def test_llm_defaults(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        llm = cfg["llm"]
        assert llm["context_window"] == 2048
        assert llm["temperature"] == 0.7
        assert llm["max_tokens"] == 512
        assert llm["memory_conservative"] is True

    def test_memory_defaults(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        mem = cfg["memory"]
        assert mem["max_turns"] == 20
        assert mem["summary_threshold"] == 15

    def test_health_thresholds(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        health = cfg["health"]
        assert health["ram_threshold_gb"] == 4.0
        assert health["disk_free_threshold_gb"] == 5.0

    def test_orchestrator_port(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["orchestrator"]["port"] == 8000

    def test_dashboard_port(self) -> None:
        with open(CONFIG_DIR / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["web_dashboard"]["port"] == 8080


class TestModelsYaml:
    def test_models_file_exists(self) -> None:
        assert (CONFIG_DIR / "models.yaml").is_file()

    def test_models_loads(self) -> None:
        with open(CONFIG_DIR / "models.yaml") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)
        assert "models" in cfg

    def test_llm_model_spec(self) -> None:
        with open(CONFIG_DIR / "models.yaml") as f:
            cfg = yaml.safe_load(f)
        llm_section = cfg["models"]["llm"]
        # New structure: llm.active points to the active model entry
        active_key = llm_section.get("active")
        if active_key:
            llm = llm_section[active_key]
        else:
            llm = llm_section  # legacy flat structure
        assert "filename" in llm
        assert "url" in llm
        assert llm["size_gb"] > 0

    def test_asr_model_spec(self) -> None:
        with open(CONFIG_DIR / "models.yaml") as f:
            cfg = yaml.safe_load(f)
        asr_section = cfg["models"]["asr"]
        active_key = asr_section.get("active")
        if active_key:
            asr = asr_section[active_key]
        else:
            asr = asr_section
        assert "model_file" in asr
        assert "url" in asr

    def test_tts_model_spec(self) -> None:
        with open(CONFIG_DIR / "models.yaml") as f:
            cfg = yaml.safe_load(f)
        tts_section = cfg["models"]["tts"]
        active_key = tts_section.get("active")
        if active_key:
            tts = tts_section[active_key]
        else:
            tts = tts_section
        assert "voice" in tts
        assert "url" in tts


class TestToolsYaml:
    def test_tools_file_exists(self) -> None:
        assert (CONFIG_DIR / "tools.yaml").is_file()

    def test_expected_tools_present(self) -> None:
        with open(CONFIG_DIR / "tools.yaml") as f:
            cfg = yaml.safe_load(f)
        tools = cfg.get("tools", {})
        for name in ("homework_tutor", "researcher", "coder_assist", "desktop_organizer"):
            assert name in tools, f"Missing tool config: {name}"
