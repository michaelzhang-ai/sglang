"""MI35x Qwen 3.5 MGSM-EN Chat Evaluation Test (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B (MoE, Hybrid Attention with Gated Delta Networks)
with MGSM-EN chat completion benchmark on MI35x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35", nightly=True
)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
SERVER_LAUNCH_TIMEOUT = 3600
ACCURACY_THRESHOLD = 0.90
TP_SIZE = 8


class TestQwen35EvalMI35x(CustomTestCase):
    """Qwen 3.5 MGSM-EN Chat Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_qwen35_mgsm_accuracy(self):
        """Test Qwen 3.5 with MGSM-EN chat completion benchmark."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "triton",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        process = popen_launch_server(
            QWEN35_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

            args = SimpleNamespace(
                base_url=self.base_url,
                model=QWEN35_MODEL_PATH,
                eval_name="mgsm_en",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            score = metrics["score"]

            passed = score >= ACCURACY_THRESHOLD
            status = "PASS" if passed else "FAIL"
            print(f"  score={score:.3f} threshold={ACCURACY_THRESHOLD} {status}")

            if is_in_ci():
                summary = "### Qwen 3.5 Model (MI35x)\n\n"
                summary += "| Model | TP | Score | Threshold | Status |\n"
                summary += "| ----- | -- | ----- | --------- | ------ |\n"
                summary += f"| {QWEN35_MODEL_PATH} | {TP_SIZE} | {score:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                score,
                ACCURACY_THRESHOLD,
                f"Qwen 3.5 score {score:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
