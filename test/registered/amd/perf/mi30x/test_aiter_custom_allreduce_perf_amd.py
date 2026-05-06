"""AMD nightly performance benchmark for aiter's custom all-reduce kernel (2-GPU).

This test launches the standalone all-reduce micro-benchmarks under
``benchmark/kernels/all_reduce/`` via ``torchrun`` and parses the per-message-
size timings on rank 0.  Its purpose is to detect performance regressions in
``aiter``'s ``CustomAllreduce`` kernel
(see https://github.com/ROCm/aiter/commits/main/aiter/ops/custom_all_reduce.py)
which is the implementation sglang dispatches to on AMD when
``SGLANG_USE_AITER_AR=1`` (the default).

Two benchmark scripts are exercised:

* ``benchmark_aiter.py`` (primary): times
  ``aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce``
  directly alongside sglang's own ``CustomAllreduce``.  Regressions in the
  aiter kernel show up here.
* ``benchmark_all_reduce.py`` (baseline): times sglang's ``CustomAllreduce``
  versus ``TorchSymmMemCommunicator``.  Useful as a control to distinguish
  aiter-specific regressions from broader stack regressions.

The numeric tables produced by the benchmarks are echoed verbatim into the
GitHub Step Summary so the nightly history is easy to eyeball over time.

Registry: nightly-amd-perf-allreduce-2-gpu suite
"""

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_amd_ci

# Two micro-benchmarks at a few message sizes are quick (single-digit minutes
# even with warmup), but est_time also has to absorb torchrun spin-up,
# AITER kernel JIT compilation on first invocation, and CI-side overhead.
register_amd_ci(est_time=900, suite="nightly-amd-perf-allreduce-2-gpu", nightly=True)

# test/ has no __init__.py; add this dir so the sibling helpers module is
# importable when this file is invoked directly via `python3 <path>`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _aiter_custom_allreduce_perf_helpers import run_perf_suite  # noqa: E402


class TestAiterCustomAllreducePerfAMD2Gpu(unittest.TestCase):
    """Run both all-reduce micro-benchmarks at TP=2."""

    def test_allreduce_microbench(self):
        run_perf_suite(self, nproc=2)


if __name__ == "__main__":
    unittest.main()
