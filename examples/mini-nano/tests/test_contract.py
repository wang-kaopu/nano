import subprocess
import sys
from pathlib import Path

import mini_nano


def test_mini_nano_module_and_public_exports():
    assert mini_nano.Nano is not None
    assert mini_nano.FakeModelClient is not None
    assert not hasattr(mini_nano, "MiniAgent")
    result = subprocess.run([sys.executable, "-m", "mini_nano", "--help"], capture_output=True, text=True, check=True)
    assert "Teaching-sized Nano agent harness" in result.stdout


def test_readme_main_mapping_points_to_existing_files():
    repo_root = Path(__file__).resolve().parents[3]
    main_files = [
        "nano/cli.py",
        "nano/runtime.py",
        "nano/agent_loop.py",
        "nano/context_manager.py",
        "nano/providers/clients.py",
        "nano/tool_executor.py",
        "nano/tools.py",
        "nano/task_state.py",
        "nano/run_store.py",
        "nano/workspace.py",
    ]
    for path in main_files:
        assert (repo_root / path).exists()
