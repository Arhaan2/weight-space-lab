# Pinned acquisition and published-patch reconstruction

This workflow prepares the exact published p50c intervention using
`configs/published_p50c.json`. It does not run inference, call an experimental
editor, choose a candidate, or evaluate a benchmark. The reference's status
records the state when that configuration was written; verification evidence
belongs in the separate acquisition and reconstruction audits.

## Reproduce

Use Python 3.11 or later, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

python -m weight_space_lab acquire-published \
  --reference configs/published_p50c.json \
  --out artifacts/p50c-acquisition \
  --max-download-bytes 3000000000 \
  --timeout 900 --max-elapsed-seconds 1800

python -m weight_space_lab verify-acquisition \
  --reference configs/published_p50c.json \
  --bundle artifacts/p50c-acquisition

python -m weight_space_lab reconstruct-published \
  --reference configs/published_p50c.json \
  --acquisition artifacts/p50c-acquisition \
  --out artifacts/p50c-reconstruction

python -m weight_space_lab verify-reconstruction \
  --reference configs/published_p50c.json \
  --acquisition artifacts/p50c-acquisition \
  --out artifacts/p50c-reconstruction

python -m weight_space_lab inventory-model \
  --model artifacts/p50c-reconstruction/model \
  --out artifacts/p50c-reconstruction-inventory.json
```

Acquisition requires HTTPS access to the public GitHub API, raw GitHub files,
Hugging Face, and its weight-download redirect hosts. The selected snapshot is
about 1.77 GB including tokenizer assets. Allow at least 4 GB free disk for both
original and reconstructed copies, plus filesystem overhead. Network requests
have finite transfer time and byte budgets, with no automatic retries or paid
infrastructure. Elapsed-time budgets are checked between and after socket
operations; an in-progress socket operation can delay timeout failure until
control returns. These are not process-level cancellation deadlines. A slow
connection may require an explicitly larger timeout;
that changes the transfer budget, not the model or precision. All subsequent
commands are offline. Only NumPy is required; no inference framework or GPU is
needed for reconstruction.

Outputs must be new paths. To repeat a run, choose a new output directory and
keep the previous audit. A verified acquisition can be reused for multiple
reconstructions. Do not edit the original files or point caches inside the
snapshot. Large files remain under ignored `artifacts/` or another local
artifact location, outside tracked source.

## What is acquired and recorded

The source revision and base model revision are immutable commit identifiers.
Acquisition confirms each resolved revision, reads only the two selected patch
paths from the upstream source repository, and selects the model's actual
Safetensors layout from its index. It obtains configuration, tokenizer and
processor assets, chat templates, and available license/notice files from the
same model revision. It excludes alternative weight formats, training files,
upstream conversations, and repository history. No upstream code is executed.

Every downloaded file is checked against the resolved source's size and Git
blob identity or LFS SHA256. The acquisition bundle contains:

- `model/`: the original files, made read-only after verification.
- `patch/manifest.json` and `patch/edit.npz`: the selected published patch.
- `reference.json`: the exact input reference, including expectations.
- `acquisition.json`: resolved revisions, source-relative paths and identities,
  every file's SHA256 and size, and the complete model inventory.

The inventory records each tensor's name, shape, stored dtype, element count,
shard, data offsets, and SHA256; the shard headers and index; parsed model
configuration; tokenizer assets; and exact chat-template contents and hashes.
The element total covers all stored tensors, including buffers and separately
stored aliases. It is not a claim about unique trainable parameters. A single
shard with an index is handled through that index as well.

The full original checkpoint is preserved, including vision components and
processor files in the pinned Qwen release. This does not construct a new
text-only architecture or claim equivalence of inference settings to the
upstream evaluation harness.

## Reconstruction and verification

Reconstruction copies every file into a hidden, explicitly incomplete staging
directory, then uses the existing exact sparse BF16 replay implementation.
Only the target tensor byte spans are written, through the actual shard
mapping. There is no floating-point conversion, model resave, quantization, or
change to the tokenizer or configuration. No hard links alias the original.

The verifier opens saved files afresh and does not call the edit function. It
checks the complete file and tensor sets, exact header bytes, index, assets,
and all unrelated tensor hashes. For each target, it checks published base and
edited hashes, saved substitution values, absence of changes outside the
sparse coordinates, and the observed stored-bit difference count. The
serialized audit is rechecked before the new directory is atomically renamed
to its final path. A completed reconstruction contains `model/` and
`audit.json` together. Handled failures remove staging; an abrupt process
termination may leave a hidden incomplete directory, never a finalized bundle.
Do not run concurrent writers against the same output path.

Offline verification checks recorded remote identities against local bytes.
It cannot authenticate an attacker replacing both a bundle and its audit.
Retain trusted audit hashes separately and supply the independently maintained
reference. Ordinary read-only permissions protect against accidental writes,
not against an owner deliberately changing permissions. Keep inputs stable
while commands run.

The legacy `replay` command remains a single-file mechanics utility; use the
directory commands above for complete experiment artifacts.

## Three distinct coordinate counts

The p50c NPZ stores only the substitutions that survived BF16 rounding. Its
index arrays therefore do not enumerate every coordinate targeted by the
original full-column operation. The verifier records:

- **Sparse substitutions:** the actual lengths of the validated index arrays.
- **Targeted column footprint:** matrix row count times the number of unique
  columns observed in those arrays, checked against the manifest's column
  count. The exact column IDs are retained. This derives the footprint of the
  published full-column operation; omitted unchanged coordinates are not
  claimed to be present in the sparse archive.
- **Changed stored entries:** the independently measured count of unequal
  uint16 BF16 bit patterns between original and saved tensors, including any
  signed-zero changes.

The reference expects 130,188 changed stored entries and a targeted footprint
of 131,072. These are acceptance checks, not measurements. A successful audit
must derive them from downloaded artifacts and saved weights. The
[pinned upstream release description](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
documents the column operation and the rounding distinction.

This replays the intervention released by Takehi, Teoh, Zhang, and Khattab,
whose experimental editor was Astra (`gpt-6-astra`). It is not a fresh-editor
discovery. Keep reconstruction evidence separate from model-loading checks,
development smoke runs, pilots, and benchmark experiments. An inference
runtime, decoding settings, datasets, and graders still need their own pinned
and validated execution configuration before those later stages.
