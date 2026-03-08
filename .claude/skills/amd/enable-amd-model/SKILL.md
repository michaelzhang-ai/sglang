---
name: enable-amd-model
description: End-to-end workflow for enabling a new model on AMD GPUs in SGLang. Covers HuggingFace architecture research, AMD backend selection (aiter/triton/wave/NSA with auto-selection logic), accuracy test file creation for MI30x and MI35x, CI workflow YAML updates, and documentation. Use when enabling a model on AMD, adding a model to AMD CI, or when the user mentions AMD model enablement.
---

# Enable a Model on AMD GPUs

End-to-end workflow: architecture research, test files (MI30x + MI35x), CI YAML (2 workflow files x 3 edit locations each), documentation, and local validation.

## Step 1: Architecture Research

Fetch the model's HuggingFace config and SGLang model implementation.

```bash
curl -s https://huggingface.co/{MODEL_PATH}/raw/main/config.json | python3 -m json.tool
```

Key fields to extract:

| Field | What it tells you |
|---|---|
| `architectures` | Maps to `python/sglang/srt/models/` via `ModelRegistry` |
| `num_hidden_layers` | Total layer count |
| `num_attention_heads` / `num_key_value_heads` | GQA ratio |
| `kv_lora_rank` or `q_lora_rank` | MLA architecture indicator |
| `num_experts` / `num_experts_per_tok` | MoE configuration |
| `quantization_config` | FP8/INT4/MXFP4 format |

### Determine AttentionArch

SGLang has two attention architectures (defined in `python/sglang/srt/configs/model_config.py`):

| AttentionArch | Models | Detection |
|---|---|---|
| **MLA** | DeepSeek-V2/V3/V3.2, Kimi-K2/K2.5, MiniCPM3, GLM-MoE-DSA, MistralLarge3, Pixtral, BailingMoe, SarvamMLA | Has `kv_lora_rank` in config, or architecture class in MLA list in `model_config.py` |
| **MHA** | Everything else (Llama, Mistral, Mixtral, Qwen, MiniMax, GLM-5, Grok, etc.) | Default |

Read the model source in `python/sglang/srt/models/` and `python/sglang/srt/configs/model_config.py` to confirm.

## Step 2: Backend Selection

### Auto-selection (preferred)

SGLang **auto-selects** the attention backend when `--attention-backend` is not set. On AMD (`is_hip()` is true), the logic in `server_args.py::_get_default_attn_backend()` is:

| AttentionArch | Auto-selected backend | Condition |
|---|---|---|
| MHA | `aiter` | Default on HIP |
| MLA | `aiter` | When `num_kv_heads / tp_size` is 16 or 128 |
| MLA | `triton` | When head count is not 16 or 128 |
| NSA models (GLM-5) | `nsa` with `tilelang` prefill+decode | Auto-detected on HIP |

**In most cases, just set `SGLANG_USE_AITER=1` and let auto-selection work.** Only override `--attention-backend` when you need a specific backend.

### AMD-compatible backends

From `ATTENTION_BACKEND_CHOICES` in `server_args.py`:

| Backend | AMD support | When to use |
|---|---|---|
| `aiter` | AMD-specific | Default for most MHA and MLA models |
| `triton` | Cross-platform | Fallback when aiter doesn't support head count; also used for some base models |
| `wave` | AMD-specific | Wave kernel backend (experimental) |
| `nsa` | Yes (with tilelang) | For NSA models (GLM-5 style); prefill/decode default to `tilelang` on HIP |
| `torch_native` | Cross-platform | Generic fallback |
| `flex_attention` | Cross-platform | Torch flex attention |

NSA sub-backends (`NSA_CHOICES`): `tilelang` (default on HIP), `aiter`, `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `trtllm`.

### Special cases

- **Llama4**: auto-selects `aiter` on HIP
- **Diffusion models**: forces `triton` on HIP
- **Mixed prefill/decode**: use `--prefill-attention-backend` and `--decode-attention-backend` to override separately (e.g., Kimi-K2.5 uses `aiter` prefill + `triton` decode)
- **NSA models**: use `--nsa-prefill-backend tilelang --nsa-decode-backend tilelang` (or let auto-selection handle it)

### Common server args by model type

| Model characteristic | Additional args |
|---|---|
| MoE with many experts | `--ep-size 8` |
| Large models (>100B) | `--mem-fraction-static 0.85 --watchdog-timeout 1200` |
| MLA models | `--chunked-prefill-size 131072` |
| Models needing trust | `--trust-remote-code` |
| Fast loading | `--model-loader-extra-config '{"enable_multithread_load": true}'` |

## Step 3: Create Test Files

Create **two** test files — one for MI30x, one for MI35x. See the `write-amd-nightly-test` skill for detailed templates and patterns.

### File locations

- MI30x: `test/registered/amd/accuracy/mi30x/test_{model}_eval_amd.py`
- MI35x: `test/registered/amd/accuracy/mi35x/test_{model}_eval_mi35x.py`

### Key MI30x vs MI35x differences

| | MI30x | MI35x |
|---|---|---|
| HF cache | (system default) | `os.environ.setdefault("HF_HOME", "/data2/models/huggingface")` and `os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")` |
| `est_time` | 3600 | 5400 |
| Suite name prefix | `nightly-amd-accuracy-8-gpu-` | `nightly-amd-8-gpu-mi35x-` or `nightly-amd-accuracy-8-gpu-mi35x-` |
| Class name suffix | `EvalAMD` | `EvalMI35x` |
| Summary header | `(MI325)` | `(MI35x)` |
| `timeout` in ModelConfig | 3600 | 5400 |

## Step 4: Update CI Workflow YAML

Edit **both** workflow files. Each requires changes in **three** places.

### Runners

| Platform | GPUs | Runner label |
|---|---|---|
| MI30x (MI300X/MI325X) | 1 | `linux-mi325-1gpu-sglang` |
| MI30x | 2 | `linux-mi325-2gpu-sglang` |
| MI30x | 8 | `linux-mi325-8gpu-sglang` |
| MI35x (MI355X) | 1 | `linux-mi35x-gpu-1` |
| MI35x | 8 | `linux-mi35x-gpu-8` |
| MI35x (disagg/RDMA) | 8 | `linux-mi35x-gpu-8.fabric` |

### File 1: `.github/workflows/nightly-test-amd.yml`

**Place 1** — Add to `job_select` options list:
```yaml
          - nightly-8-gpu-{model}           # MI30x
          - nightly-8-gpu-mi35x-{model}     # MI35x
```

**Place 2** — Add job definition block (MI30x template):
```yaml
  nightly-8-gpu-{model}:
    if: >-
      (github.repository == 'sgl-project/sglang' || github.event_name == 'pull_request')
      && (!(inputs.job_filter || inputs.job_select) || (inputs.job_filter || inputs.job_select) == 'all'
      || contains(format(',{0},', inputs.job_filter || inputs.job_select), ',nightly-8-gpu-{model},'))
    runs-on: linux-mi325-8gpu-sglang
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          ref: ${{ inputs.ref || github.ref }}
      - name: Setup docker
        run: |
          touch github_summary.md
          bash scripts/ci/amd/amd_ci_start_container.sh
        env:
          GITHUB_WORKSPACE: ${{ github.workspace }}
      - name: Install dependencies
        run: bash scripts/ci/amd/amd_ci_install_dependency.sh
      - name: Accuracy Test (8-GPU {MODEL_DISPLAY})
        timeout-minutes: 120
        run: |
          > github_summary.md
          bash scripts/ci/amd/amd_ci_exec.sh -w /sglang-checkout/test \
            -e SGLANG_USE_AITER=1 \
            -e GITHUB_STEP_SUMMARY="/sglang-checkout/github_summary.md" \
            python3 run_suite.py --hw amd --suite {SUITE_NAME} --nightly --timeout-per-file 3600 ${{ inputs.continue_on_error && '--continue-on-error' || '' }} || TEST_EXIT_CODE=$?
          echo "$(<github_summary.md )" >> $GITHUB_STEP_SUMMARY || true
          exit ${TEST_EXIT_CODE:-0}
```

MI35x variant: `runs-on: linux-mi35x-gpu-8`, job name prefix `nightly-8-gpu-mi35x-{model}`.

**Place 3** — Add to `check-all-jobs.needs` list:
```yaml
      - nightly-8-gpu-{model}
      - nightly-8-gpu-mi35x-{model}
```

### File 2: `.github/workflows/nightly-test-amd-rocm720.yml`

Same three places, but:
- Job names get `-rocm720` suffix
- Docker setup adds `--rocm-version rocm720`
- Install step adds `--skip-test-time-deps`
- Suite name stays the same (test file is shared)

## Step 5: Update Documentation

If the model has a docs page under `docs/basic_usage/`, add an AMD GPU deployment section. If the model is new, add it to `docs/supported_models/text_generation/generative_models.md`.

## Step 6: Local Validation

```bash
docker exec -it {CONTAINER} bash
cd /sglang-checkout && pip install -e "python[all]"

SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
    --model {MODEL_PATH} --tp 8 {OTHER_ARGS} &

python3 test/registered/amd/accuracy/mi30x/test_{model}_eval_amd.py
```

Verify: accuracy meets threshold, no HIP errors, server launches within timeout.

## Checklist

- [ ] HuggingFace config analyzed (AttentionArch: MLA or MHA)
- [ ] Backend verified (auto-selection or explicit override)
- [ ] MI30x test file created in `test/registered/amd/accuracy/mi30x/`
- [ ] MI35x test file created in `test/registered/amd/accuracy/mi35x/`
- [ ] `nightly-test-amd.yml` updated (3 places: options, job block, needs)
- [ ] `nightly-test-amd-rocm720.yml` updated (3 places: options, job block, needs)
- [ ] Documentation updated (if applicable)
- [ ] Local validation passed on AMD hardware

## References

- `python/sglang/srt/server_args.py` — `ATTENTION_BACKEND_CHOICES`, `_get_default_attn_backend()`, auto-selection logic
- `python/sglang/srt/configs/model_config.py` — `AttentionArch` enum (MLA, MHA)
- `python/sglang/srt/layers/attention/attention_registry.py` — backend name → class mapping
- `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` — standalone MHA test
- `test/registered/amd/accuracy/mi30x/test_deepseek_v32_eval_amd.py` — standalone MLA test
- `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` — shared evaluator test
- `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` — NSA backend test
- `.github/workflows/nightly-test-amd.yml` — CI workflow (ROCm default)
- `.github/workflows/nightly-test-amd-rocm720.yml` — CI workflow (ROCm 7.2)
