# Published p50c artifact verification

Verified 2026-11-06. This report concerns acquired files and saved checkpoint
bytes. It contains no model generations, benchmark scores, or fresh-editor
experiment.

| Evidence category | Status |
| --- | --- |
| Implementation | Acquisition, inventory, directory replay, and independent saved verification implemented |
| Fixtures | All 126 CPU-only tests pass, including sharded layouts, failure cleanup, and tampering |
| Real artifacts | Pinned snapshot and complete saved reconstruction verified |
| Real model execution | Not attempted; no inference runtime installed |
| Smoke run, pilot, full experiment | Not executed |

The reference is [published_p50c.json](../../configs/published_p50c.json). Its
original pre-execution status is retained as part of the frozen input, not
treated as a current result. The resolved source revision is
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; the resolved model revision is
`2fc06364715b967f1860aea9cf38778875588b17` for `Qwen/Qwen3.5-0.8B`.

## Observed reconstruction

| Target | Stored shape | Sparse substitutions | Targeted column footprint | Changed saved entries |
| --- | --- | ---: | ---: | ---: |
| Layer 15 MLP down projection | 1024 × 3584 | 65,260 | 65,536 | 65,260 |
| Layer 19 MLP down projection | 1024 × 3584 | 64,928 | 65,536 | 64,928 |
| Total | | 130,188 | 131,072 | 130,188 |

The archive contains only sparse BF16 substitutions that differ from base.
The targeted footprint was derived from the actual 64 unique column IDs in
each archive index array and all 1,024 tensor rows, and checked against the
published column-edit description. The complete column IDs are in the audit.
The 884 remaining coordinates within that footprint have unchanged stored
bits; they are not extra entries in the archive. Both published base hashes
and both published edited hashes matched the actual tensor bytes.

The full snapshot has 11 model files, one indexed Safetensors shard, and 488
stored tensors. The stored-element total is 873,438,784: 873,436,192 BF16
elements across 452 tensors and 2,592 F32 elements across 36 tensors. These
counts include buffers and stored aliases. All 486 unrelated tensors,
headers, index, configuration, tokenizer, chat template, processor assets,
and license bytes were preserved. The original acquisition is read-only and
was reverified after reconstruction.

## Evidence and identity

- [Acquisition audit](acquisition.json): every file and tensor hash, original
  configuration and templates, resolved revisions, source paths, and remote
  Git/LFS identities.
- [Reconstruction audit](reconstruction.json): the complete saved inventory,
  actual change counts, target column IDs, and input-audit binding.
- [Supplementary independent comparison](independent-verification.json):
  direct byte comparisons outside target spans using a checker that imports
  no project implementation.

| Artifact | SHA256 |
| --- | --- |
| Original weight shard | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Reconstructed weight shard | `bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5` |
| Published manifest | `83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225` |
| Published archive | `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7` |
| Acquisition audit | `d0e7830ed0012e0beb91641af2a0d5d48bb85628e95c9eb1cb94371bc3300549` |
| Reconstruction audit | `648c3661e7fcf0f36d34a6c344d5c158c549be581d9d7b7a3f05585403cfd624` |

The audits are exact copies of the finalized local bundle audits. They contain
metadata and chat-template text, not model weights. Model configuration and
template content originate from Qwen and retain the upstream
[Apache 2.0 license](LICENSE.model). Project implementation remains under its
own MIT license. The published intervention is attributed to Takehi, Teoh,
Zhang, and Khattab, with Astra (`gpt-6-astra`) as their experimental editor.
No new editor call occurred here.

## Reproduction and limits

Executed local checks included the full unittest suite, `pip check`, Python
compilation, execution diagnostics without backend probes, acquisition
reverification, saved reconstruction verification, full inventory export, and
the supplementary independent comparison. The publication-content scan ran
from temporary code outside tracked source. Large weight files were confirmed
ignored, and copied report audits and the model license were checked against
the finalized bundles.

```bash
python -m unittest discover -s tests -v
python -m pip check
python -m compileall -q weight_space_lab tests scripts
python -m weight_space_lab diagnose
```

Follow the exact [acquisition, reconstruction, and verification commands](../../docs/model-artifacts.md#reproduce).
For the additional independent whole-file comparison:

```bash
python scripts/verify_p50c_bytes.py \
  artifacts/p50c-acquisition artifacts/p50c-reconstruction \
  --reference configs/published_p50c.json
```

Acquisition ran with Python 3.13.5 and NumPy 2.2.5; its operations were network
transfer, hashing, and metadata inspection. Reconstruction, offline
reverification, the independent comparison, and the fixture suite ran with
Python 3.13.5 and the repository-pinned NumPy 2.3.5. The documented environment
uses the pinned version throughout. No Torch, Transformers, Safetensors
runtime, inference server, grader, or GPU execution was needed.

Reacquisition requires public network access and sufficient disk space;
offline verification can reuse the acquired bundles. Native model loading
and generation still require a selected compatible inference runtime,
versioned dependencies, explicit precision and decoding settings, and a
bounded development-only execution check. Benchmarks additionally require
pinned datasets, independent graders, schedules, and evaluation controls.
Those requirements affect later execution stages and do not invalidate this
byte-level reconstruction. The original checkpoint includes the vision
components of the pinned public release; no text-only model conversion was
performed.
