"""Shared helpers for the aiter custom all-reduce nightly perf tests.

Imported by the per-world-size test files
(``test_aiter_custom_allreduce_perf_amd*.py``).  This module is registered
with ``disabled="helper module..."`` so the suite runner discovers the
``register_amd_ci`` marker (sanity check) but never invokes it as a test.
"""

import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import write_github_step_summary

# Helper module: the registry marker is required by the suite runner's sanity
# check (every .py under test/registered/ must register at least once); the
# `disabled` reason keeps the file from being executed as a test on its own.
register_amd_ci(
    est_time=0,
    suite="nightly-amd-perf-allreduce-2-gpu",
    nightly=True,
    disabled="helper module, no tests",
)

# Matches lines like:
#   [SGLang] 32K: 0.123 ms (avg across ranks)
#   [Aiter] 1M: 1.234 ms (avg across ranks)
#   [SGLangCustom] 64M: 5.678 ms (avg across ranks)
#   [TorchSymmMem] 16M: 3.210 ms (avg across ranks)
_TIMING_RE = re.compile(
    r"^\[(?P<impl>[A-Za-z0-9_]+)\]\s+(?P<size>\d+[BKMG])\s*:\s*"
    r"(?P<ms>[\d.]+)\s*ms\s*\(avg across ranks\)\s*$"
)


def repo_root() -> Path:
    # test/registered/amd/perf/mi30x/<this file>  ->  repo root is parents[5]
    return Path(__file__).resolve().parents[5]


def benchmark_dir() -> Path:
    return repo_root() / "benchmark" / "kernels" / "all_reduce"


def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def parse_timings(stdout: str) -> Dict[str, Dict[str, float]]:
    """Parse the per-implementation, per-size timings from benchmark stdout.

    Returns a nested dict ``{impl: {size: ms}}``.  Unknown / unparsable
    lines are ignored so the parser remains tolerant to format tweaks.
    """
    timings: Dict[str, Dict[str, float]] = {}
    for line in stdout.splitlines():
        m = _TIMING_RE.match(line.strip())
        if not m:
            continue
        impl = m.group("impl")
        size = m.group("size")
        try:
            ms = float(m.group("ms"))
        except ValueError:
            continue
        timings.setdefault(impl, {})[size] = ms
    return timings


def _size_to_bytes(s: str) -> int:
    unit = s[-1]
    magnitude = int(s[:-1])
    return magnitude * {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]


def format_markdown_table(
    title: str,
    timings: Dict[str, Dict[str, float]],
    nproc: int,
    elapsed_s: float,
) -> str:
    """Render the parsed timings as a Markdown table for the step summary.

    Columns are the implementations seen in stdout (in iteration order).
    Rows are the message sizes, sorted from smallest to largest.
    """
    if not timings:
        return (
            f"### {title} (TP={nproc})\n"
            "_(no timings parsed; benchmark may have skipped or aborted)_\n\n"
        )

    impls = list(timings.keys())

    all_sizes = set()
    for impl_timings in timings.values():
        all_sizes.update(impl_timings.keys())
    sizes_sorted = sorted(all_sizes, key=_size_to_bytes)

    header = "| size | " + " | ".join(f"{impl} (ms)" for impl in impls) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(impls)) + " |"
    lines = [
        f"### {title} (TP={nproc}, wall={elapsed_s:.1f}s)",
        "",
        header,
        sep,
    ]
    for size in sizes_sorted:
        cells = []
        for impl in impls:
            v = timings[impl].get(size)
            cells.append(f"{v:.3f}" if v is not None else "N/A")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_torchrun_benchmark(
    script: Path,
    nproc: int,
    extra_args: Optional[List[str]] = None,
    timeout: int = 1200,
) -> Tuple[str, float]:
    """Run a benchmark script under torchrun and return (stdout, elapsed_s).

    Raises ``AssertionError`` if torchrun exits non-zero so the unit test
    surfaces the failure with the captured output attached.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(script),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    # Make the aiter dispatch path explicit so the nightly is self-documenting
    # (the runtime default on AMD is already "true").
    env.setdefault("SGLANG_USE_AITER_AR", "1")

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise AssertionError(
            f"Benchmark {script.name} (nproc={nproc}) failed with exit code "
            f"{result.returncode}.\nCommand: {' '.join(cmd)}\n"
            f"Output:\n{result.stdout}"
        )

    print(f"\n===== {script.name} (TP={nproc}) stdout =====", flush=True)
    print(result.stdout, flush=True)
    print(f"===== end {script.name} (TP={nproc}) stdout =====\n", flush=True)

    return result.stdout, elapsed


def run_perf_suite(test_case: unittest.TestCase, nproc: int) -> None:
    """Execute both micro-benchmarks at ``nproc`` and report into the step summary.

    Designed to be called from a single ``unittest`` test method per file so
    that the test runner reports one file = one suite of work.  Failures in
    individual benchmarks short-circuit via ``run_torchrun_benchmark`` raising
    ``AssertionError``.
    """
    if gpu_count() < nproc:
        test_case.skipTest(
            f"This test requires at least {nproc} GPUs, found {gpu_count()}."
        )

    bdir = benchmark_dir()
    benchmark_aiter = bdir / "benchmark_aiter.py"
    benchmark_all_reduce = bdir / "benchmark_all_reduce.py"
    for script in (benchmark_aiter, benchmark_all_reduce):
        if not script.exists():
            test_case.skipTest(f"Required benchmark script missing: {script}")

    summary_chunks: List[str] = [
        "## Aiter Custom AllReduce Nightly Benchmark\n",
        f"Runner GPU count: {gpu_count()}, TP size: {nproc}\n\n",
    ]

    try:
        # Primary benchmark: SGLang vs Aiter -- this is what catches
        # regressions in aiter's CustomAllreduce kernel directly.
        stdout, elapsed = run_torchrun_benchmark(benchmark_aiter, nproc=nproc)
        timings = parse_timings(stdout)
        summary_chunks.append(
            format_markdown_table(
                "benchmark_aiter.py (SGLang vs Aiter)", timings, nproc, elapsed
            )
        )
        test_case.assertTrue(
            any(impl in timings for impl in ("Aiter", "SGLang")),
            "benchmark_aiter.py produced no parsable timings for either "
            f"SGLang or Aiter. Parsed: {sorted(timings.keys())}",
        )
        if "Aiter" not in timings:
            print(
                "[WARN] Aiter implementation produced no timings; the kernel "
                "may have disabled itself for this topology.",
                flush=True,
            )

        # Secondary benchmark: SGLang vs Torch SymmMem (the script the user
        # originally referenced).  Useful as a control to disambiguate
        # aiter-side regressions from broader stack regressions.
        stdout, elapsed = run_torchrun_benchmark(benchmark_all_reduce, nproc=nproc)
        timings = parse_timings(stdout)
        summary_chunks.append(
            format_markdown_table(
                "benchmark_all_reduce.py (SGLang vs TorchSymmMem)",
                timings,
                nproc,
                elapsed,
            )
        )
        test_case.assertTrue(
            any(impl in timings for impl in ("SGLangCustom", "TorchSymmMem")),
            "benchmark_all_reduce.py produced no parsable timings for "
            f"either SGLangCustom or TorchSymmMem. "
            f"Parsed: {sorted(timings.keys())}",
        )
    finally:
        # Append everything in one go so the section is contiguous in the
        # GitHub Actions UI even if the test fails partway through.
        write_github_step_summary("".join(summary_chunks))
