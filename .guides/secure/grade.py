#!/usr/bin/env python3
"""
grade.py — Resilient Codio Autograder for ICE 5: Model Selection (Exhaustive & Forward Stepwise)
================================================================================================
Supports dual execution modes:
  1. Standard pytest execution (when pytest is installed in container / venv).
  2. Native Python unittest fallback (when pytest is missing, ensuring zero crash failures).

Points Breakdown:
  - Question 1 (Exhaustive Best Subset): 10.0 points
  - Question 2 (Forward Stepwise Selection): 15.0 points
  Total: 25.0 points
"""

import contextlib
import importlib
import io
import os
import re
import site
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

# Configure writable cache directory for Matplotlib (critical for Codio / sandboxes)
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------------------------
# Dynamic Path & Workspace Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()
CODIO_WORKSPACE = Path("/home/codio/workspace")

SEARCH_DIRS = [
    SCRIPT_DIR,
    CWD,
    CODIO_WORKSPACE,
] + list(SCRIPT_DIR.parents) + list(CWD.parents)

for d in SEARCH_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

# ---------------------------------------------------------------------------
# Virtual Environment Resolution & Discovery
# ---------------------------------------------------------------------------
import shutil

SYSTEM_ROOTS = {Path("/"), Path("/usr"), Path("/usr/local"), Path("/etc")}

def discover_virtualenvs():
    """Scans all standard and custom virtual environment locations in the Codio stack."""
    found = []

    if "VIRTUAL_ENV" in os.environ:
        found.append(Path(os.environ["VIRTUAL_ENV"]))

    for prog in ("pytest", "python3", "python"):
        prog_path = shutil.which(prog)
        if prog_path:
            p = Path(prog_path).resolve()
            if p.parent.name in ("bin", "Scripts"):
                parent_dir = p.parent.parent
                if parent_dir not in SYSTEM_ROOTS and (parent_dir / "pyvenv.cfg").exists():
                    found.append(parent_dir)

    found.extend([
        Path("/home/codio/cs5821_venv"),
        Path("/home/codio/workspace/cs5821_venv"),
        Path("/home/codio/workspace/.venv"),
        Path("/home/codio/workspace/venv"),
        Path("/home/codio/.venv"),
        Path("/home/codio/venv"),
        Path("/home/codio/env"),
        Path.home() / "cs5821_venv",
        Path.home() / ".venv",
        Path.home() / "venv",
        Path("/opt/cs5821_venv"),
        Path("/opt/venv"),
        Path("/opt/conda"),
        Path("/usr/local/venv"),
        SCRIPT_DIR.parent.parent / ".venv",
        SCRIPT_DIR.parent / ".venv",
        SCRIPT_DIR / ".venv",
        CWD / ".venv",
        CODIO_WORKSPACE / ".venv",
    ])

    for base in [Path("/home/codio"), Path("/opt"), Path("/var")]:
        if base.exists():
            try:
                for cfg in base.glob("*/pyvenv.cfg"):
                    found.append(cfg.parent)
                for cfg in base.glob("*/*/pyvenv.cfg"):
                    found.append(cfg.parent)
                for act in base.glob("*/bin/activate"):
                    found.append(act.parent.parent)
                for act in base.glob("*/*/bin/activate"):
                    found.append(act.parent.parent)
            except Exception:
                pass

    seen = set()
    unique = []
    for c in found:
        try:
            resolved = c.resolve()
            if resolved not in SYSTEM_ROOTS and resolved not in seen and resolved.is_dir():
                seen.add(resolved)
                unique.append(resolved)
        except Exception:
            pass
    return unique

VENV_CANDIDATES = discover_virtualenvs()

HAS_PYTEST = False
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

current_exe = Path(sys.executable).resolve()
for venv_dir in VENV_CANDIDATES:
    for py_name in ("python3", "python"):
        venv_py = venv_dir / "bin" / py_name
        if venv_py.exists():
            try:
                resolved_py = venv_py.resolve()
                if resolved_py != current_exe:
                    res = subprocess.run(
                        [str(resolved_py), "-c", "import pytest"],
                        capture_output=True,
                        timeout=5
                    )
                    if res.returncode == 0:
                        os.environ["VIRTUAL_ENV"] = str(venv_dir)
                        os.environ["PATH"] = f"{venv_dir / 'bin'}:{os.environ.get('PATH', '')}"
                        os.execv(str(resolved_py), [str(resolved_py)] + sys.argv)
            except Exception:
                pass

py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for venv_dir in VENV_CANDIDATES:
    if venv_dir.exists():
        for sp in venv_dir.glob(f"lib/{py_ver}/site-packages"):
            if str(sp) not in sys.path:
                sys.path.insert(0, str(sp))

try:
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if sp and Path(sp).exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
except Exception:
    pass

if not HAS_PYTEST:
    try:
        import pytest
        HAS_PYTEST = True
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location", "pytest"],
                capture_output=True,
                timeout=25
            )
            importlib.invalidate_caches()
            import pytest
            HAS_PYTEST = True
        except Exception:
            HAS_PYTEST = False


# ---------------------------------------------------------------------------
# Test File Discovery
# ---------------------------------------------------------------------------
def locate_test_file() -> Path:
    candidates = [
        SCRIPT_DIR / "tests.py",
        SCRIPT_DIR.parent / "tests.py",
        SCRIPT_DIR.parent.parent / "tests.py",
        CODIO_WORKSPACE / "tests.py",
        CODIO_WORKSPACE / ".guides" / "secure" / "tests.py",
        CWD / "tests.py",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    for d in SEARCH_DIRS:
        if d.exists():
            for m in d.rglob("tests.py"):
                if m.is_file():
                    return m.resolve()

    raise FileNotFoundError("Could not locate 'tests.py' in workspace or .guides directories.")


# ---------------------------------------------------------------------------
# Question Configuration & Point Allocation
# ---------------------------------------------------------------------------
QUESTION_METADATA = {
    "q1": {"title": "Question 1: Exhaustive Best Subset Selection", "points": 10.0, "class_name": "TestQ1"},
    "q2": {"title": "Question 2: Forward Stepwise Selection", "points": 15.0, "class_name": "TestQ2"},
}


# ---------------------------------------------------------------------------
# Pytest Runner
# ---------------------------------------------------------------------------
class CodioPytestGrader:
    def __init__(self):
        self.reports = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.reports.append(report)
        elif report.when in ("setup", "teardown") and report.failed:
            self.reports.append(report)


def run_with_pytest(test_file: Path) -> dict:
    grader = CodioPytestGrader()
    pytest.main([str(test_file), "-q", "--tb=no"], plugins=[grader])

    q_results = {q_id: {"passed": [], "failed": []} for q_id in QUESTION_METADATA}
    for rep in grader.reports:
        test_name = rep.nodeid.split("::")[-1]
        parent_class = rep.nodeid.split("::")[-2] if len(rep.nodeid.split("::")) > 1 else ""

        matched_q = None
        for q_id, meta in QUESTION_METADATA.items():
            if meta["class_name"] in parent_class or f"test_{q_id}_" in test_name:
                matched_q = q_id
                break

        if matched_q:
            if rep.passed:
                q_results[matched_q]["passed"].append(test_name)
            else:
                msg = ""
                if hasattr(rep, "longreprtext"):
                    lines = [ln.strip() for ln in rep.longreprtext.splitlines() if ln.strip()]
                    msg = lines[-1] if lines else "Test failed"
                q_results[matched_q]["failed"].append((test_name, msg))

    return q_results


# ---------------------------------------------------------------------------
# Unittest Fallback Runner (Zero External Dependencies)
# ---------------------------------------------------------------------------
class UnittestCollector(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.reports = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.reports.append((test.id().split(".")[-1], test.__class__.__name__, True, ""))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.reports.append((test.id().split(".")[-1], test.__class__.__name__, False, str(err[1])))

    def addError(self, test, err):
        super().addError(test, err)
        self.reports.append((test.id().split(".")[-1], test.__class__.__name__, False, str(err[1])))


def run_with_unittest(test_file: Path) -> dict:
    tests_dir = str(test_file.parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    import tests as tests_module

    suite = unittest.defaultTestLoader.loadTestsFromModule(tests_module)
    collector = UnittestCollector()
    suite.run(collector)

    q_results = {q_id: {"passed": [], "failed": []} for q_id in QUESTION_METADATA}
    for test_name, class_name, passed, msg in collector.reports:
        matched_q = None
        for q_id, meta in QUESTION_METADATA.items():
            if meta["class_name"] in class_name or f"test_{q_id}_" in test_name:
                matched_q = q_id
                break

        if matched_q:
            if passed:
                q_results[matched_q]["passed"].append(test_name)
            else:
                q_results[matched_q]["failed"].append((test_name, msg))

    return q_results


# ---------------------------------------------------------------------------
# Grade Reporting to Codio LMS
# ---------------------------------------------------------------------------
def send_to_codio(score_pct: float, feedback_md: str):
    try:
        sys.path.append("/usr/share/codio/assessments")
        from lib.grade import send_partial_v2, FORMAT_V2_MD
        send_partial_v2(round(score_pct, 2), feedback_md, FORMAT_V2_MD)
    except Exception:
        pass

    codio_url = os.environ.get("CODIO_PARTIAL_POINTS_V2_URL")
    if codio_url:
        try:
            data = urllib.parse.urlencode({
                "points": score_pct,
                "format": "md",
                "feedback": feedback_md
            }).encode("utf-8")
            req = urllib.request.Request(codio_url, data=data)
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass


def main():
    test_file = locate_test_file()

    if HAS_PYTEST:
        q_results = run_with_pytest(test_file)
    else:
        q_results = run_with_unittest(test_file)

    total_earned = 0.0
    total_possible = sum(m["points"] for m in QUESTION_METADATA.values())
    feedback_lines = [
        "# ICE 5: Model Selection — Autograder Results\n"
    ]

    for q_id, meta in sorted(QUESTION_METADATA.items()):
        q_title = meta["title"]
        q_points = meta["points"]
        res = q_results[q_id]

        n_passed = len(res["passed"])
        n_total = n_passed + len(res["failed"])
        earned = (n_passed / n_total) * q_points if n_total > 0 else 0.0
        total_earned += earned

        status_emoji = "✅" if earned == q_points else ("⚠️" if earned > 0 else "❌")
        feedback_lines.append(f"### {q_title}: **{earned:.1f} / {q_points:.1f} pts** {status_emoji}")
        feedback_lines.append(f"- Passed {n_passed} of {n_total} test cases.")

        for p_name in res["passed"]:
            feedback_lines.append(f"  - ✅ `{p_name}`")

        for f_name, msg in res["failed"]:
            feedback_lines.append(f"  - ❌ `{f_name}`")
            if msg:
                short_msg = msg[:120] + "..." if len(msg) > 120 else msg
                feedback_lines.append(f"    - *Details:* `{short_msg}`")

        feedback_lines.append("")

    overall_pct = (total_earned / total_possible) * 100 if total_possible > 0 else 0.0
    feedback_lines.append("---")
    feedback_lines.append(f"## **Overall Score: {total_earned:.1f} / {total_possible:.1f} ({overall_pct:.1f}%)**\n")

    full_feedback = "\n".join(feedback_lines)
    print(full_feedback)

    send_to_codio(overall_pct, full_feedback)
    sys.exit(0)


if __name__ == "__main__":
    main()
