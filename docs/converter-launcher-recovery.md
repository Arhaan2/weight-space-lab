# Converter launcher recovery

Arhaan2. The original candidate conversion launcher dereferenced the configured
virtual-environment Python symlink before starting its package check. Python
therefore selected the system environment, where the pinned converter packages
were unavailable. The failed check did not start weight conversion or inference.

`weight_space_lab.browser_candidate_recovery` provides the explicitly authorized
continuation. Its `run_converter_python` helper makes the configured executable
absolute relative to the repository, preserves the virtual-environment path in
`argv[0]`, and checks that it is an executable file. Both the package check and
the conversion subprocess use that helper. The package check retains its
30-second timeout, and conversion retains its 1,800-second timeout and existing
environment, source, artifact, and parameter-contract checks.

The original `browser_candidate.py` remains unchanged because the completed
baseline manifest pins its bytes. The recovery launcher has a separate source
hash and authorization record. Its conversion receipt binds that record, the
original launcher hash, the repaired launcher hash, and the preserved failure.
Existing packaging and candidate binding incorporate the receipt into the
candidate artifact and execution manifest. Baseline provenance is not rewritten.

The recovery entry point accepts the existing `convert` arguments plus
`--recovery-record`. It does not import a proposal. The continuation then uses
the existing `package-candidate`, candidate authorization binding, browser run,
and paired comparison commands. Output destinations and receipts must be new;
an exclusive continuation intent records the single manually authorized attempt.
Failed attempts and spent ledgers are preserved. There are no automatic retries.

Offline verification uses the production helper to inspect `sys.executable`,
`sys.prefix`, `sys.base_prefix`, converter import locations, and the frozen
dependency versions. Import checks use the existing conversion environment,
including `MLC_LLM_HOME`, so they do not fall back to an unrelated host cache.
The regression test creates a real temporary virtual environment with a module
available only inside it, changes the subprocess working directory, and imports
that module through the helper. These checks perform no model inference.

Private recovery records, authorizations, paths, and execution logs remain under
`artifacts/`. Public reports contain sanitized results, source hashes, and the
preserved failure history. The original baseline and stopped-stage reports are
retained alongside the recovered comparison.
