"""Explicit, provenance-bound continuation of the converter launcher precheck.

The original frozen launcher and failed-attempt evidence stay unchanged. This
module retains its conversion checks and limits while preserving the configured
virtual-environment executable in argv[0]. It never imports a proposal or retries
a conversion automatically. Arhaan2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from . import browser_candidate as original_launcher
from .browser_candidate import _files, _save, source_identity, verify_baseline
from .browser_conversion import tensor_index
from .inventory import file_sha256, read_json
from .protocol import canonical_hash


def converter_python_command(converter_python, arguments, *, repository_root=None):
    """Anchor argv before changing cwd without dereferencing the venv executable."""
    root = (Path(repository_root) if repository_root is not None
            else Path(__file__).absolute().parent.parent).absolute()
    executable = Path(converter_python)
    if not executable.is_absolute():
        executable = root / executable
    executable = executable.absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('Configured converter Python must be an executable file')
    return [str(executable), *map(str, arguments)]


def run_converter_python(converter_python, arguments, *, repository_root=None, **options):
    """Use the same configured interpreter for probes and the actual conversion."""
    command = converter_python_command(converter_python, arguments,
                                       repository_root=repository_root)
    return subprocess.run(command, **options)


def verify_recovery_record(path):
    """Bind this explicit continuation to its repair, authorization and failure."""
    path = Path(path)
    record = read_json(path)
    if (record.get('schema_version') != 1
            or record.get('original_source_sha256') != file_sha256(original_launcher.__file__)
            or record.get('repaired_launcher_sha256') != file_sha256(__file__)
            or record.get('authorization_file_sha256') != file_sha256(record['authorization_path'])
            or record.get('failure_file_sha256') != file_sha256(record['failure_path'])):
        raise ValueError('Launcher recovery provenance or explicit authorization changed')
    failure = read_json(record['failure_path'])
    if (failure.get('status') != 'stopped'
            or failure.get('stage') != 'candidate_conversion_package_precheck'
            or failure.get('conversion_subprocess_started') is not False
            or failure.get('candidate_generation_dispatches') != 0):
        raise ValueError('Recovery requires the preserved pre-conversion package-check failure')
    return {'record_sha256': file_sha256(path),
            **{key: record[key] for key in ('original_source_sha256', 'repaired_launcher_sha256',
                'authorization_file_sha256', 'failure_file_sha256')}}


def convert(original, candidate, baseline_manifest, output, converter_python, *,
            recovery_record, policy_path='configs/candidate_policy.json'):
    """Continue once from the accepted checkpoint with the frozen conversion limits."""
    started = time.monotonic()
    recovery = verify_recovery_record(recovery_record)
    original, candidate, output = Path(original), Path(candidate), Path(output)
    manifest = verify_baseline(baseline_manifest)
    identity, _, _ = source_identity(original, candidate, manifest, policy_path=policy_path)
    script = Path('scripts/browser_convert.py').resolve()
    if file_sha256(script) != manifest['converter_adapter_sha256']:
        raise ValueError('Converter differs from baseline conversion adapter')
    if output.exists() or output.with_name(output.name+'-source-binding.json').exists():
        raise FileExistsError('Conversion destination and receipt must be new')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(baseline_manifest).parent
    expected_packages = manifest['converter_packages']
    installed = run_converter_python(converter_python, ['-c',
        'import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in '
        +repr(list(expected_packages))+'}))'], check=True, capture_output=True, text=True, timeout=30)
    if json.loads(installed.stdout) != expected_packages:
        raise ValueError('Converter environment differs from baseline pinned packages')
    command = converter_python_command(converter_python, [str(script),
        '--source', str((candidate/'model').resolve()), '--output', str(output.resolve()),
        '--wasm-metadata', str((root/'wasm-metadata.json').resolve())])
    env = dict(os.environ, TVM_BACKTRACE='1', MLC_LLM_HOME=str((root/'cache').resolve()),
        PYTHONPYCACHEPREFIX='/tmp/wsl-browser-convert', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TVM_NUM_THREADS='4')
    start_record = {'schema_version': 1, 'kind': 'future_candidate_conversion_preparation',
        **identity, 'baseline_manifest_sha256': file_sha256(baseline_manifest),
        'command': command, 'converter_packages': expected_packages,
        'target_generations': 0, 'new_proposals': 0, 'launcher_recovery': recovery}
    _save(output.with_name(output.name+'-conversion-start.json'), start_record)
    with output.with_name(output.name+'-convert.stdout.txt').open('x') as stdout, \
            output.with_name(output.name+'-convert.stderr.txt').open('x') as stderr:
        result = run_converter_python(command[0], command[1:], env=env,
                                      stdout=stdout, stderr=stderr, timeout=1800)
    if result.returncode:
        _save(output.with_name(output.name+'-conversion-failure.json'),
              {'returncode': result.returncode, 'preparation_seconds': time.monotonic()-started,
               'target_generations': 0, 'launcher_recovery': recovery})
        raise RuntimeError('Conversion failed; preserved logs are preparation evidence only')
    after, _, _ = source_identity(original, candidate, manifest, policy_path=policy_path)
    if after != identity:
        raise ValueError('Source candidate changed during conversion')
    if verify_recovery_record(recovery_record) != recovery:
        raise ValueError('Launcher recovery provenance changed during conversion')
    files = _files(output)
    tensor_index(output)
    receipt = {**start_record, 'converted_files': files,
        'converted_files_sha256': canonical_hash(files),
        'preparation_seconds': time.monotonic()-started, 'status': 'converted_source_bound'}
    _save(output.with_name(output.name+'-source-binding.json'), receipt)
    return {'converted': str(output.resolve()),
        'source_binding_receipt': str(output.with_name(output.name+'-source-binding.json').resolve()),
        'source_checkpoint_sha256': identity['source_checkpoint_sha256'],
        'target_generations': 0, 'new_proposals': 0, 'launcher_recovery': recovery}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['convert'])
    parser.add_argument('--source-original', required=True)
    parser.add_argument('--source-candidate', required=True)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--converter-python', required=True)
    parser.add_argument('--policy', default='configs/candidate_policy.json')
    parser.add_argument('--recovery-record', required=True)
    args = parser.parse_args()
    result = convert(args.source_original, args.source_candidate, args.baseline_manifest,
        args.output, args.converter_python, recovery_record=args.recovery_record, policy_path=args.policy)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
