# Native development smoke, 2026-11-06

**Three real checkpoints generated successfully. No benchmark was graded and
no experiment was executed.** This validates loading and short generation,
not mathematical accuracy, weight-edit benefit, or reference-length capacity.

| Evidence category | Status |
| --- | --- |
| Implementation | Native adapters, bounded schedule runner, immutable attempts and validated resume implemented |
| Fixture checks | 166 tests pass with both NumPy 2.3.5 and isolated NumPy 2.2.6 |
| Real-model check | Original, identity-copy and reconstructed checkpoints generated through the same CPU path |
| Preferred vLLM path | Dependency resolution verified for Linux; GPU execution not attempted |
| Pilot / full experiment | Not started; no grader, editor call, score, or candidate selection |

Machine-readable evidence: [status](status.json), [build checks](build-checks.json),
[fixture checks](fixture-checks.json), and [measured report](report.json).
The raw records and immutable event journals are retained for
[original](original/records.json), [identity](identity/records.json), and
[reconstructed](reconstructed/records.json). Each directory includes its
complete configuration, task, ordered schedule and artifact identities in
`manifest.json`; the finish event contains token IDs and loaded-model metadata.

## Settings and artifact checks

All arms use `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`, full
`Qwen3_5ForConditionalGeneration`, Transformers 5.3.0, Torch 2.10.0, CPU eager
attention, four threads, and saved BF16/F32 precision. The CPU environment is
Apple arm64, Python 3.13.5, with 24 GiB host memory; process memory below is
measured separately. The complete installed package versions are recorded.

The acquired chat template and tokenizer are identical across arms. Template
SHA256 is `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`.
Thinking is enabled. The same authored development arithmetic prompt produces
33 input tokens. Each arm uses seed 17, temperature 0.6, top-p 0.95, top-k 20,
presence penalty 1.5, repetition penalty 1.0, frequency penalty 0, 2,048 context
tokens, and a **16-new-token cap**. EOS IDs are explicitly 248044 and 248046.
All outputs end `truncated/length`; none is treated as a complete answer.

All 473 active saved tensors were verified byte-for-byte after loading,
including 36 F32 tensors restored from their original saved values. The tied
output head was verified. The checkpoint's 15 auxiliary MTP tensors remain
on disk and have explicit nonuse metadata and hashes; speculative MTP is off.
The original and identity-copy have the same complete file identity; the
reconstructed checkpoint has the verified Phase 2 substitutions and a different
identity. All three emitted the same 16 output token IDs in this tiny smoke.
That observation does not show equivalent longer behavior or effectiveness
of the patch.

The [native execution guide](../../docs/native-execution.md) explains the
full-architecture, per-sample seed, EOS and CPU-runtime variations relative
to the released text-only vLLM evaluation. The reference output cap is 32,768,
not 16. This smoke is neither a pilot nor a reference reproduction.

## Measured resources and planning estimates

| Arm | Load seconds | Generation seconds | Output tokens/s | Peak process RSS, bytes |
| --- | ---: | ---: | ---: | ---: |
| Original | 4.569 | 21.492 | 0.744 | 4,088,086,528 |
| Identity-copy | 2.892 | 22.127 | 0.723 | 4,029,612,032 |
| Reconstructed | 3.593 | 21.045 | 0.760 | 3,961,405,440 |

Generation time includes prompt rendering, tokenization and generation, and
excludes loading and final text decoding. Load time includes tensor checks.
Peak RSS is the worker's whole-process high-water mark, including verification
temporaries, not incremental model memory or GPU allocation. Each arm used a
fresh worker, but file caches may be warm. These are single short observations,
not repeated timing trials or uncertainty intervals.

At the measured 0.723–0.760 output tokens/s, a **hypothetical** serial run of
100 sequences each reaching 1,024 output tokens would take approximately
**37.4–39.3 hours for the generation component alone**. Applying that same
unvalidated constant rate to 2,000 sequences at the reference cap of 32,768
tokens yields roughly 998–1,049 days per arm. Neither larger run was launched.
Longer contexts change compute and memory; these extrapolations motivate a
suitable GPU rather than promise completion times. Long-context memory fit
has not been tested. Measured worker peak RSS was approximately 3.69–3.81 GiB.

## Failure and recovery evidence

The initial original-model attempt failed closed with `loaded_tensor_missing`
before generation. Its [raw record](initial-load-failure/records.json), frozen
inputs and journal are preserved separately. A bounded load diagnostic found
the 15 MTP omissions explicitly declared by the pinned runtime. The adapter
was corrected to accept only that exact set and record it. Fresh configurations
and output directories were prepared for all three subsequent arms; the failed
run was not silently resumed under changed code. Four requests are accounted
for in the saved journals: one startup failure and three 16-token generations.
The separate diagnostic load generated no tokens.

After the successful original run, its derived `records.json` was removed
and recovered with `--resume`. The recovered bytes were identical and the
event count remained two; no generation was repeated. Resuming with a changed
output cap of 17 was rejected with exit status 2 and preserved the journal.
The process tests additionally verify hard parent interruption and worker
death, startup and generation timeouts, missing committed events, corruption,
retry histories, and configuration/task/schedule/asset incompatibility.

The CPU worker shutdown emitted a Python semaphore-cleanup warning; the
resource tracker cleaned it during process exit. No model worker was left
running. Terminal diagnostics are private working material, not raw scientific
generation records.

## Reproduction and remaining prerequisites

The [exact setup and three-arm commands](../../docs/native-execution.md#reproduce-the-three-arm-smoke)
reproduce the procedure. The successful local execution used the same commands
with `native-smoke-v2-inputs` and `native-smoke-v2-{original,identity,reconstructed}`
as destinations. Existing acquisitions and identity bundles can be reused;
choose fresh run/input destinations for a new execution. Reproduce the checks:

```bash
python -m unittest discover -s tests -v
python -m pip check
python -m compileall -q weight_space_lab tests scripts
python -m weight_space_lab check-status reports/native-smoke/status.json
python -m weight_space_lab check-records reports/native-smoke/original/records.json
python -m weight_space_lab check-records reports/native-smoke/identity/records.json
python -m weight_space_lab check-records reports/native-smoke/reconstructed/records.json
```

The preferred vLLM 0.17.0 backend still needs a real **Linux x86-64 host with
glibc 2.31 or later, Python 3.12, a BF16-capable NVIDIA GPU, and a driver
compatible with CUDA 12.8**. Package resolution was verified for that target;
Torch 2.10.0 resolves CUDA runtime 12.8.90. Neither installation on that host,
GPU inference, long-context fit, nor vLLM throughput is claimed. Install its
separate pinned environment and run the same three tiny smoke arms first.

A larger study additionally needs pinned benchmark tasks, validated independent
graders, a frozen paired schedule, reference settings and an explicit resource
budget. Fresh-editor experiments require the actual Astra interface and data
isolation. These prerequisites affect future experiments, not the completed
CPU development smoke. No upstream benchmark repository was modified.
