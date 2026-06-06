"""Tests for the CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from tattletots.cli import main


class TestCLI:
    def test_basic_invocation(self) -> None:
        """CLI runs a short simulation without crashing."""
        result = main(["--steps", "5", "--population", "5", "--seed", "42"])
        assert result == 0

    def test_verbose_mode(self) -> None:
        result = main(["--steps", "10", "--population", "5", "--seed", "42", "--verbose"])
        assert result == 0

    def test_output_file(self, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        result = main(
            ["--steps", "5", "--population", "5", "--seed", "42", "--output", str(output)]
        )
        assert result == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert "summary" in data
        assert "config" in data
        assert "population_history" in data

    def test_config_file(self, tmp_path: Path) -> None:
        config = {
            "simulation": {
                "initial_population": 5,
                "max_steps": 5,
                "seed": 42,
            },
            "scenario": {
                "n_components": 5,
                "dimensionality": 20,
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        result = main(["--config", str(config_path), "--steps", "5"])
        assert result == 0

    def test_output_contains_scenario_config(self, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        main(["--steps", "5", "--population", "5", "--seed", "42", "--output", str(output)])
        data = json.loads(output.read_text())
        assert "scenario" in data
        assert data["scenario"]["scenario"] == "gaussian_shift"
