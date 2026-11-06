# Native generation and resumable execution

Phase 3 implements generation and execution accounting. It does not implement
answer grading, fresh-editor access, candidate search, or a benchmark
experiment. The public command is deliberately limited to development smoke
profiles. A completed generation is not a correctness measurement.

The [Phase 5 replay workflow](published-patch-replay.md) separately composes
this runner with frozen benchmark data, independent graders, and paired
reporting. `run-native` remains smoke-only; full replay execution requires the
explicit `replay-full` command and its validated study freeze.

## Runtime and methodological choices

The preferred execution path is **vLLM 0.17.0**, on a compatible Linux CUDA
machine, using the complete acquired `Qwen3_5ForConditionalGeneration`
checkpoint. The [vLLM architecture registry at v0.17.0](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/models/registry.py)
includes this architecture. The [released evaluation code](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py)
used vLLM, but did not pin its package version. Our version is a new execution
pin, not a claim about the authors' installed version.

There are two explicit variations from that evaluation:

- Its [model profile](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/profiles.py)
  used a text-only export. This project preserves the acquired full release,
  including vision weights and configuration. Text requests use its language
  path without converting the architecture.
- Each scheduled sample has its own seed and generation request. The upstream
  harness grouped repeated outputs using `n` and a shared sampling seed. The
  new schedule is traceable and paired across arms; it does not reproduce the
  upstream random-number stream.

For the local Apple CPU check, **Transformers 5.3.0 / Torch 2.10.0**, eager
attention, four CPU threads, and BF16 are an additional methodological
variation. [Transformers v5.3.0](https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5)
contains the exact full architecture. This is an explicitly selected backend,
never a fallback. It retains the snapshot's F32 tensors as well as its BF16
tensors and checks the loaded values. This mixed storage/compute path must
not be assumed numerically identical to vLLM's BF16 execution. No
quantization, MPS substitution, text-only conversion, or FP32 model fallback
is performed.

Ordinary autoregressive execution does not use the release's 15 auxiliary
multi-token-prediction (MTP) tensors. The pinned Transformers class explicitly
omits them. The adapter accepts only their exact known names, records their
saved hashes and reason for nonuse, verifies all 473 active saved tensors
byte-for-byte, and checks the tied output head. All 488 tensors remain in each
complete checkpoint. Speculative MTP is disabled. CPU DeltaNet uses the
framework's Torch implementation; optional accelerated kernels are absent in
the recorded CPU environment.

Heavy dependencies are optional. The default package and CPU fixture suite
do not import them. Install the two runtime extras in **separate environments**;
vLLM 0.17.0 requires a different Transformers version:

```bash
python -m pip install -e '.[native-cpu]'
# On a separate compatible Linux CUDA machine/environment:
python -m pip install -r configs/native-vllm-requirements.txt
python -m pip install -e '.[native-vllm]'
```

The CPU extra pins NumPy 2.3.5. The vLLM extra pins NumPy 2.2.6 because its
Numba dependency requires NumPy below 2.3. The lightweight package accepts
2.2.6 through 2.3.5; the complete installed package versions are frozen in each
run, and changing them prevents resume. The measured CPU environment is
also version-locked in [native-cpu-requirements.txt](../configs/native-cpu-requirements.txt).
The [GPU requirements](../configs/native-vllm-requirements.txt) resolved for
Linux x86-64, glibc 2.31+, Python 3.12 using uv 0.13.5. This verifies package
constraints, not an installed or functioning GPU runtime. The resolved Torch
CUDA runtime is 12.8.90; a compatible driver and BF16-capable GPU are required.

Core versions are exact and checked before model loading. Runtime metadata
records the loaded classes, actual versions, generation settings, tokenizer
and template identity, and measured resource observations. Missing packages,
version mismatch, wrong architecture, unknown options, or unsupported required
settings fail; the runner never silently chooses another runtime.

## Explicit generation contract

All three arms load their own verified local model directory and tokenizer
assets from model revision `2fc06364715b967f1860aea9cf38778875588b17`.
`local_files_only` / offline operation prevents model substitution or downloads.
The template is the acquired `chat_template.jinja`, with a checked SHA256,
`add_generation_prompt=true`, and explicit `enable_thinking=true`.
The authored development task is saved in `tasks.json`, including the complete
user message. No benchmark test task is acquired or exposed.

| Setting | Tiny development smoke | Released math reference |
| --- | --- | --- |
| Sampling | Stochastic | Stochastic |
| Temperature / top-p / top-k | 0.6 / 0.95 / 20 | 0.6 / 0.95 / 20 |
| Presence penalty | 1.5, additive, generated tokens only | 1.5 |
| Repetition / frequency penalty | 1.0 / 0.0 | No additional penalty in released call |
| Thinking | Enabled | Enabled |
| Maximum new tokens | **16** | **32,768** |
| Context limit | **2,048** | **34,816** in released evaluation |
| Sample schedule | One authored task, seed 17 per arm | Benchmark-specific repeats |
| Grading / scientific score | None | Separate published evaluators |

The Transformers presence processor subtracts the penalty once from each
token ID already present in the generated suffix, independent of its count
or the sign of its logit. Prompt tokens do not incur this penalty. Repetition
penalty is never used in its place. Unsupported frequency/repetition changes,
stop strings, ignored EOS, or decoding options are rejected.

Stopping is explicit: EOS IDs `248044` (`<|endoftext|>`, model configuration)
and `248046` (`<|im_end|>`, tokenizer), or the output-token limit;
no stop strings, ignored EOS, or automatic prompt truncation. Raw decoding
retains special tokens and disables whitespace cleanup; exact generated token
IDs and rendered prompt identity accompany the output. EOS is `ok/eos`;
reaching the token limit is `truncated/length`. A truncated smoke response is
expected and does not establish a completed reasoning answer.
The EOS union is this project's explicit policy; upstream's exported generation
metadata was not fully pinned, so identical implicit EOS behavior is not claimed.

## Reproduce the three-arm smoke

First complete the [Phase 2 acquisition and reconstruction](model-artifacts.md#reproduce).
Run from the repository root with a new destination for each new run:

```bash
source .venv/bin/activate
python -m pip install -r configs/native-cpu-requirements.txt
python -m pip install -e '.[native-cpu]'
python -m weight_space_lab copy-identity \
  --acquisition artifacts/p50c-acquisition --out artifacts/p50c-identity
python -m weight_space_lab prepare-native-smoke \
  --acquisition artifacts/p50c-acquisition \
  --reconstruction artifacts/p50c-reconstruction \
  --identity artifacts/p50c-identity --backend transformers \
  --out artifacts/native-smoke-inputs

python -m weight_space_lab run-native artifacts/native-smoke-inputs/original.json \
  --model artifacts/p50c-acquisition/model \
  --tasks artifacts/native-smoke-inputs/tasks.json \
  --schedule artifacts/native-smoke-inputs/schedule.json \
  --out artifacts/native-smoke-original
python -m weight_space_lab run-native artifacts/native-smoke-inputs/identity.json \
  --model artifacts/p50c-identity/model \
  --tasks artifacts/native-smoke-inputs/tasks.json \
  --schedule artifacts/native-smoke-inputs/schedule.json \
  --out artifacts/native-smoke-identity
python -m weight_space_lab run-native artifacts/native-smoke-inputs/reconstructed.json \
  --model artifacts/p50c-reconstruction/model \
  --tasks artifacts/native-smoke-inputs/tasks.json \
  --schedule artifacts/native-smoke-inputs/schedule.json \
  --out artifacts/native-smoke-reconstructed

python -m weight_space_lab check-records artifacts/native-smoke-original/records.json
python -m weight_space_lab check-records artifacts/native-smoke-identity/records.json
python -m weight_space_lab check-records artifacts/native-smoke-reconstructed/records.json
```

For vLLM, explicitly select `--backend vllm` at preparation in its separate
environment and choose new input and output destinations. Reuse the same
three model bundles. Never edit a frozen configuration to resume a CPU run
under vLLM. No paid services or remote inference endpoints are used.

The identity control is an independently copied, reopened, byte-verified
directory. It is **not** a `save_pretrained` reserialization experiment.
The original and identity have identical complete file digests; reconstructed
weights have a different digest and the same tokenizer/template/configuration.
All three take the same adapter and generation path.

## Durability, attempts, and resume

Append `--resume` to the exact original `run-native` command. The run binds
the entire configuration, ordered schedule, task text, artifact inventory,
retry policy, budgets, backend versions, and executable project-source digest.
The CLI rehashes the actual complete checkpoint and all assets before each
run or resume. Changed source or configuration requires a new preparation and
output directory; changed configuration never silently extends old evidence.
Keep input files stable throughout execution.

The runner takes an exclusive lock and durably records an attempt start before
dispatch. Atomic, fsynced immutable events preserve every attempt. Completed
output and metadata are committed together; derived `records.json` contains
one Phase 1 record per scheduled item/sample, including unstarted `missing`
rows. A missing derived export is rebuilt from the journal. Corrupt committed
events are rejected rather than silently treated as unstarted work.

An interrupted in-flight attempt becomes a recorded failure on recovery.
No retry follows a successful or truncated generation. Retries require the
frozen `retry_errors` policy, a retryable failure, and remaining attempt,
sequence, and elapsed budgets. Smoke configurations allow exactly one
attempt. To test an intentionally different retry policy, prepare and retain a
new configuration before execution. Attempts with unknown measurements keep
nulls, not fabricated zero counts. Resource use from failed attempts remains
part of execution accounting.
The CLI exits with status 2 if any scheduled result remains missing, errored,
or timed out. A deliberately token-limited smoke generation is an accounted
truncated result, not a correctness score.

Startup and generation run in an isolated worker with bounded deadlines.
Timeouts stop the worker; a parent-liveness guard prevents an orphaned worker
from continuing indefinitely after abrupt parent death. The tiny configuration
allows 300 seconds for startup, 180 for a request, 600 total, and one attempted
sequence per arm. These are safety ceilings, not performance measurements.

Run the offline recovery tests with:

```bash
python -m unittest discover -s tests -v
python -m pip check
python -m compileall -q weight_space_lab tests scripts
```

The fixture backend is visibly synthetic. It tests errors, timeouts, crashes,
missing exports, interruption, retries, concurrent access, and incompatible
resume without loading a model. Fixture outputs cannot enter the scientific
raw-to-grade conversion. A separate fixture projection retains a synthetic
envelope and cannot be passed directly to `compare`.

## Resource interpretation and next boundary

Report model load time separately from generation time, token throughput,
total attempt time, and peak process resident memory. Short-prompt, 16-token
timings can support a rough sequential planning extrapolation, not a validated
long-context capacity or benchmark ETA. Report the assumptions with the
estimate; do not launch a longer measurement merely to obtain a favorable rate.

Native precision, CPU kernel behavior, sequence length, batching, KV cache,
and memory pressure affect scaling. A suitable Linux CUDA host is still needed
to validate the preferred vLLM path. A benchmark additionally requires pinned
tasks, independently tested graders, a complete paired schedule, reference
settings, and a separately approved execution budget. Fresh-editor work needs
the actual Astra execution interface and isolation; none is used in this phase.
