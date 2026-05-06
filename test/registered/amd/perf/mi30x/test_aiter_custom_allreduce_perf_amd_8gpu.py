"""AMD nightly performance benchmark for aiter's custom all-reduce kernel (8-GPU).

This is the 8-GPU companion to ``test_aiter_custom_allreduce_perf_amd.py``;
see that file for full background.  An 8-way all-reduce is the topology used
by 8x DeepSeek/Grok deployments on MI30x, so a regression in aiter's
``CustomAllreduce`` at TP=8 is directly user-impacting.

Registry: nightly-amd-perf-allreduce-8-gpu suite
"""

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=900, suite="nightly-amd-perf-allreduce-8-gpu", nightly=True)

# test/ has no __init__.py; add this dir so the sibling helpers module is
# importable when this file is invoked directly via `python3 <path>`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _aiter_custom_allreduce_perf_helpers import run_perf_suite  # noqa: E402


class TestAiterCustomAllreducePerfAMD8Gpu(unittest.TestCase):
    """Run both all-reduce micro-benchmarks at TP=8."""

    def test_allreduce_microbench(self):
        run_perf_suite(self, nproc=8)


if __name__ == "__main__":
    unittest.main()
