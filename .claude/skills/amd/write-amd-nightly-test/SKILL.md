---
name: write-amd-nightly-test
description: Write AMD nightly accuracy and performance tests for MI30x and MI35x platforms. Covers GSM8K completion benchmarks, MMMU VLM evaluations, performance benchmarks with NightlyBenchmarkRunner, CI suite registration with register_amd_ci, and cross-platform test variants. Use when creating AMD test files, adding models to AMD nightly CI, or writing accuracy/performance tests for AMD GPUs.
---

# Write AMD Nightly Test

Guide for writing nightly CI tests that run on AMD MI30x (MI300X/MI325X) and MI35x (MI355X) hardware.

## Test Types and Templates

| Type | Evaluation | Template file |
|---|---|---|
| Text accuracy (GSM8K standalone) | Inline 5-shot completion benchmark | `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` |
| Text accuracy (shared evaluator) | `sglang.test.few_shot_gsm8k.run_eval` | `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` |
| Text accuracy (LMEvalMixin) | `LMEvalMixin` + `CustomTestCase` | `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` |
| VLM accuracy (MMMU) | `run_eval` with `eval_name="mmmu"` | `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` |
| Performance | `NightlyBenchmarkRunner` | `test/registered/amd/perf/mi30x/test_deepseek_v32_basic_perf_amd.py` |
| Diffusion | Custom server + generation | `test/registered/amd/test_wan2_2_i2v_a14b.py` |

## CI Registration

Every test file **must** call `register_amd_ci` at module level:

```python
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-{model}",
    nightly=True,
)
```

The `suite` name must match the CI workflow YAML job that invokes `run_suite.py --suite {suite}`.

### Suite naming (from actual codebase)

Suite names are **not strictly uniform** — follow the naming style closest to existing tests of the same type:

| Pattern | Examples |
|---|---|
| `nightly-amd-accuracy-8-gpu-{model}` | `nightly-amd-accuracy-8-gpu-minimax-m25`, `nightly-amd-accuracy-8-gpu-glm5` |
| `nightly-amd-8-gpu-{feature}` | `nightly-amd-8-gpu-grok`, `nightly-amd-8-gpu-deepseek-v3-kv-fp8` |
| `nightly-amd-8-gpu-mi35x-{model}` | `nightly-amd-8-gpu-mi35x-glm5`, `nightly-amd-8-gpu-mi35x-minimax-m25` |
| `nightly-amd-accuracy-8-gpu-mi35x-{model}` | `nightly-amd-accuracy-8-gpu-mi35x-qwen35`, `nightly-amd-accuracy-8-gpu-mi35x-kimi-k25` |
| `nightly-perf-8-gpu-{model}` | `nightly-perf-8-gpu-deepseek-v32-basic`, `nightly-perf-8-gpu-grok2` |
| `nightly-perf-8-gpu-mi35x-{model}` | `nightly-perf-8-gpu-mi35x-deepseek-v32-basic` |
| `nightly-8-gpu-{model}` | `nightly-8-gpu-qwen3-235b` |
| `nightly-amd` | Shared 2-GPU GSM8K accuracy suite |
| `nightly-amd-accuracy-2-gpu-vlm` | VLM MMMU 2-GPU suite |
| `stage-b-test-small-1-gpu-amd` | Per-PR unit tests on AMD |
| `stage-c-test-large-8-gpu-amd` | Per-PR 8-GPU tests on AMD |

### Runners

| Platform | GPUs | Runner label |
|---|---|---|
| MI30x | 1 | `linux-mi325-1gpu-sglang` |
| MI30x | 2 | `linux-mi325-2gpu-sglang` |
| MI30x | 8 | `linux-mi325-8gpu-sglang` |
| MI35x | 1 | `linux-mi35x-gpu-1` |
| MI35x | 8 | `linux-mi35x-gpu-8` |
| MI35x (disagg) | 8 | `linux-mi35x-gpu-8.fabric` |

## Text Accuracy Test Patterns

### Pattern A: Standalone GSM8K (inline benchmark, `unittest.TestCase`)

For models needing custom server args or per-model iteration. Most accuracy tests use this pattern.

Key structure from `test_minimax_m25_eval_amd.py`:

```python
from sglang.test.ci.ci_register import register_amd_ci
register_amd_ci(est_time=3600, suite="nightly-amd-accuracy-8-gpu-{model}", nightly=True)

@dataclass
class ModelConfig:
    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None
    variant: Optional[str] = None

MODELS = [
    ModelConfig(
        model_path="{ORG}/{MODEL}",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="TP8",
        other_args=["--attention-backend", "aiter", ...],
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
]

class TestModelEvalAMD(unittest.TestCase):
    def test_accuracy(self):
        for config in self.models:
            process = popen_launch_server(config.model_path, ...)
            try:
                acc, invalid, latency = run_gsm8k_benchmark(...)
            finally:
                kill_process_tree(process.pid)
```

### Pattern B: Shared evaluator (`CustomTestCase` + `few_shot_gsm8k`)

For simpler models with a long-lived server. Used by Kimi-K2.5, Kimi-K2, DeepSeek-V3.2 variants.

```python
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import CustomTestCase

class TestModelEvalAMD(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(MODEL_PATH, cls.base_url, ...)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_accuracy(self):
        args = SimpleNamespace(num_shots=8, num_questions=1319, parallel=1319, ...)
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(metrics["accuracy"], THRESHOLD)
```

### Pattern C: LMEvalMixin (`CustomTestCase` + mixin)

For models that use the `lm-eval` harness. Used by Qwen3.5.

```python
from sglang.test.test_utils import CustomTestCase, LMEvalMixin

class TestModelEvalAMD(LMEvalMixin, CustomTestCase):
    ...
```

## Backend Args by Architecture

### MHA models (standard attention)

```python
other_args=[
    "--attention-backend", "aiter",  # or omit to use auto-selection
    "--trust-remote-code",
    "--mem-fraction-static", "0.85",
],
env_vars={"SGLANG_USE_AITER": "1"},
```

### MLA models (DeepSeek-style)

```python
other_args=[
    "--attention-backend", "aiter",  # auto-selects when head_num is 16 or 128
    "--chunked-prefill-size", "131072",
    "--trust-remote-code",
    "--mem-fraction-static", "0.85",
    "--model-loader-extra-config", '{"enable_multithread_load": true}',
    "--watchdog-timeout", "1200",
],
env_vars={"SGLANG_USE_AITER": "1"},
```

### MoE models

Add `--ep-size 8` for models with many experts (e.g., MiniMax-M2.5 with 256 experts).

### NSA models (GLM-5 style)

```python
other_args=[
    "--trust-remote-code",
    "--nsa-prefill-backend", "tilelang",  # default on HIP, can omit
    "--nsa-decode-backend", "tilelang",   # default on HIP, can omit
    "--chunked-prefill-size", "131072",
    "--mem-fraction-static", "0.80",
    "--model-loader-extra-config", '{"enable_multithread_load": true}',
    "--watchdog-timeout", "1200",
],
env_vars={"SGLANG_USE_AITER": "1"},
```

### Mixed prefill/decode (Kimi-K2.5)

```python
other_args=[
    "--decode-attention-backend", "triton",
    "--prefill-attention-backend", "aiter",
],
env_vars={"SGLANG_USE_AITER": "1", "SGLANG_ROCM_FUSED_DECODE_MLA": "0"},
```

## VLM Accuracy Test Pattern

VLM tests use MMMU evaluation instead of GSM8K:

```python
args = SimpleNamespace(
    base_url=self.base_url,
    model=model_path,
    eval_name="mmmu",
    num_examples=100,
    num_threads=64,
    max_tokens=30,
)
metrics = run_eval(args)
```

VLM tests typically:
- Use TP=1 or TP=2 (smaller models)
- Support retries (up to 3 attempts)
- Track startup/eval/total times in summaries
- Exclude known-failing models via `AMD_FAILING_VLM_MODELS` list

## Performance Test Pattern

Performance tests use `NightlyBenchmarkRunner`:

```python
from sglang.test.nightly_utils import NightlyBenchmarkRunner

runner = NightlyBenchmarkRunner(model_path, variant_config, ...)
runner.run_benchmark_for_model(batch_sizes=[1, 32], input_lens=[...], output_lens=[...])
```

## MI30x vs MI35x Differences

When creating a MI35x variant, apply these changes:

```python
# Add at TOP of MI35x file (before other imports)
import os
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
```

| Aspect | MI30x | MI35x |
|---|---|---|
| Suite name | `nightly-amd-accuracy-8-gpu-{model}` | `nightly-amd-8-gpu-mi35x-{model}` |
| `est_time` | 3600 | 5400 |
| Class name | `Test{Model}EvalAMD` | `Test{Model}EvalMI35x` |
| `timeout` | 3600 | 5400 |

## GitHub Step Summary

```python
from sglang.test.test_utils import is_in_ci, write_github_step_summary

if is_in_ci():
    summary = "### {Model} ({Platform})\n\n"
    summary += "| Model | TP | Accuracy | Threshold | Status |\n"
    summary += "| ----- | -- | -------- | --------- | ------ |\n"
    summary += f"| {path} | {tp} | {acc:.3f} | {threshold} | {status} |\n"
    write_github_step_summary(summary)
```

## Accuracy Thresholds

Always measure on real AMD hardware first. Set threshold ~2-3% below measured accuracy.

## Checklist

- [ ] Test class inherits from `unittest.TestCase` or `CustomTestCase`
- [ ] `register_amd_ci(...)` called at module level
- [ ] Suite name matches CI workflow YAML job
- [ ] `SGLANG_USE_AITER=1` set in env vars
- [ ] Server killed in teardown via `kill_process_tree`
- [ ] GitHub step summary generated
- [ ] Has `if __name__ == "__main__": unittest.main()`
- [ ] MI35x variant created with HF cache env, adjusted est_time/timeout, updated suite name
- [ ] Accuracy threshold validated on real hardware

## References

- `.claude/skills/write-sglang-test/SKILL.md` — general SGLang test writing guide
- `.claude/skills/amd/enable-amd-model/SKILL.md` — full enablement workflow including CI YAML
- `python/sglang/srt/server_args.py` — `ATTENTION_BACKEND_CHOICES`, `NSA_CHOICES`, auto-selection
- `python/sglang/test/few_shot_gsm8k.py` — shared GSM8K evaluator
- `python/sglang/test/test_utils.py` — `CustomTestCase`, `LMEvalMixin`, `popen_launch_server`
- `python/sglang/test/nightly_utils.py` — `NightlyBenchmarkRunner` for perf tests
- `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` — standalone MHA template
- `test/registered/amd/accuracy/mi30x/test_deepseek_v32_eval_amd.py` — standalone MLA template
- `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` — shared evaluator template
- `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` — NSA backend template
- `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` — LMEvalMixin template
- `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` — VLM template
- `test/registered/amd/perf/mi30x/test_deepseek_v32_basic_perf_amd.py` — perf template
