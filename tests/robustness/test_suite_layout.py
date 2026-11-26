"""Collection target for project CPU tests.

Add unit tests beside this file, under the suite packages below. Keep those
tests free of GPU, simulator, and checkpoint dependencies so the project CI
can run them.
"""

from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent
PROJECT_SUITES = ("perturbations", "rollouts", "config", "evaluation")


def test_project_suites_exist():
    missing = [name for name in PROJECT_SUITES if not (SUITE_ROOT / name).is_dir()]
    assert missing == []
