#!/usr/bin/env python3
"""Later, separately bound maintenance for completed dedicated browser profiles.

No frozen controller or scientific protocol is changed. Importing this module
performs no model work. The CLI is intended for the study's pinned interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = ('Default/IndexedDB/http_127.0.0.1_4198.indexeddb.blob',)
SOURCE_PATHS = ('scripts/maintain_fresh_profiles.py', 'docs/fresh-profile-maintenance-addendum.md')
REQUIRED_CLEANUP = ('unload_confirmed', 'worker_terminated', 'chrome_exit_confirmed',
                    'local_harness_exit_confirmed', 'controller_exit_verified',
                    'chrome_os_exit_verified', 'native_capture_exit_confirmed',
                    'owned_browser_and_worker_resources_released')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular file: ' + path.name)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def plain_path(path, root):
    """Refuse symlinks at every component, including a leaf slated for removal."""
    path, root = Path(path), Path(root)
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Path escapes dedicated root')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink forbidden in maintenance path')
        if part == root:
            break
    if path.resolve() != path:
        raise ValueError('Noncanonical maintenance path')


def tree_files(directory):
    directory = Path(directory)
    result = []
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            p = Path(parent) / name
            if p.is_symlink():
                raise ValueError('Symlink forbidden in maintenance tree')
            mode = p.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError('Special file forbidden in maintenance tree')
            if stat.S_ISREG(mode):
                result.append(p)
    return sorted(result)


def live_profile_processes(profile):
    process = subprocess.run(['/bin/ps', '-axo', 'pid=,command='],
                             check=True, capture_output=True, text=True, timeout=10)
    flag = '--user-data-dir=' + str(profile)
    return [int(line.strip().split(None, 1)[0]) for line in process.stdout.splitlines()
            if flag in line]


def frozen_path_references(value, leaf):
    found = []
    def visit(v):
        if isinstance(v, dict):
            for k, x in v.items():
                visit(k)
                visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, str) and str(leaf) in v:
            found.append(v)
    visit(value)
    return found


def benchmark_link_digest(session, benchmark):
    link = session / 'benchmark'
    if not link.is_symlink() or link.resolve() != Path(benchmark).resolve():
        raise ValueError('Dedicated benchmark link does not resolve to frozen benchmark')
    return hashlib.sha256(os.readlink(link).encode()).hexdigest()


def nonprofile_files(session, profile, benchmark):
    benchmark_link_digest(session, benchmark)
    result = []
    for parent, dirs, files in os.walk(session, followlinks=False):
        for name in list(dirs):
            p = Path(parent) / name
            if p == profile or p == session / 'benchmark':
                dirs.remove(name)
                continue
            if p.is_symlink():
                raise ValueError('Unexpected non-profile directory symlink')
        for name in files:
            p = Path(parent) / name
            if p == session / 'benchmark':
                continue
            if p.is_symlink() or not p.is_file():
                raise ValueError('Unexpected non-profile evidence type')
            result.append(p)
    return sorted(result)


def evidence_snapshot(session, profile, benchmark):
    """Keep non-profile files; leave unrelated Chrome singleton links untouched."""
    protected = nonprofile_files(session, profile, benchmark)
    for name in ('Last Version', 'Local State'):
        p = profile / name
        plain_path(p, session)
        if not p.is_file():
            raise ValueError('Required browser profile metadata missing: ' + name)
        protected.append(p)
    indexed = profile / 'Default/IndexedDB'
    plain_path(indexed, session)
    for p in indexed.iterdir():
        if p.name.endswith('.indexeddb.leveldb'):
            plain_path(p, session)
            protected.extend(tree_files(p))
    return {str(p.relative_to(session)): digest(p) for p in sorted(set(protected))}


def plan_cleanup(root, study, state, row, verify_session, processes=live_profile_processes):
    if row['status'] != 'completed' or state['active'] is not None or state.get('reporting_active'):
        raise ValueError('Only completed idle sessions may be maintained')
    session = root / 'sessions' / row['key']
    arm = 'baseline-1' if row['artifact_key'] == 'pristine' else 'candidate-1'
    target = session / 'execution' / arm
    profile = target / 'chrome-profile'
    plain_path(profile, root)
    manifest, _ = verify_session(root, study, state, row)
    supervisor = read(target / 'supervisor-result.json')
    if (supervisor['status'] != 'completed' or supervisor['study_manifest_sha256'] != state['study_sha256']
            or not all(supervisor.get('cleanup', {}).get(k) is True for k in REQUIRED_CLEANUP)
            or supervisor['cleanup'].get('retained_owned_pids_alive') != []
            or supervisor['cleanup'].get('forced_termination') is not False):
        raise ValueError('Completed owned-process cleanup is unverified')
    observed = row.get('observed') or {}
    if not observed or any(supervisor.get(k) != v for k, v in observed.items()):
        raise ValueError('Saved supervisor and state observation differ')
    browser = read(target / 'browser-session.json')
    if browser['profilePath'] != str(profile):
        raise ValueError('Dedicated profile ownership differs')
    if processes(profile):
        raise ValueError('A live process still owns this profile')
    maps = [study, manifest]
    if (root / 'candidate-binding.json').exists():
        maps.append(read(root / 'candidate-binding.json'))
    payloads = []
    for relative in ALLOWLIST:
        leaf = profile / relative
        plain_path(leaf, root)
        if not leaf.is_dir():
            raise ValueError('Expected original cache leaf missing; no implicit repeat')
        if any(frozen_path_references(value, leaf) for value in maps):
            raise ValueError('Cache leaf is referenced by frozen evidence')
        for p in tree_files(leaf):
            if p.stat().st_nlink != 1:
                raise ValueError('Shared hard-link cache payload forbidden')
            payloads.append({'path': str(p.relative_to(profile)), 'bytes': p.stat().st_size})
    # Search actual saved session metadata too, without publishing any content.
    for p in nonprofile_files(session, profile, study['benchmark_source']):
        if p.suffix == '.json':
            value = read(p)
            if any(frozen_path_references(value, profile / name) for name in ALLOWLIST):
                raise ValueError('Cache leaf referenced by saved session evidence')
    return {'session_key': row['key'], 'profile': str(profile), 'allowlist': list(ALLOWLIST),
            'cache_payloads': payloads, 'planned_file_count': len(payloads),
            'planned_logical_bytes': sum(x['bytes'] for x in payloads),
            'preserved_evidence_sha256': evidence_snapshot(session, profile, study['benchmark_source']),
            'preserved_benchmark_link_sha256': benchmark_link_digest(session, study['benchmark_source']),
            'benchmark_source': study['benchmark_source'],
            'recorded_cleanup': supervisor['cleanup'], 'live_profile_processes': [],
            'manifest_sha256': digest(session / 'manifest.json'),
            'supervisor_sha256': digest(target / 'supervisor-result.json')}


def remove_planned(plan, session, processes=live_profile_processes, removed=None):
    profile = Path(plan['profile'])
    plain_path(profile, session)
    if processes(profile):
        raise ValueError('Profile acquired a live owner after planning')
    if (benchmark_link_digest(session, plan['benchmark_source']) != plan['preserved_benchmark_link_sha256']
            or evidence_snapshot(session, profile, plan['benchmark_source']) != plan['preserved_evidence_sha256']):
        raise ValueError('Evidence changed before cache removal')
    actual = []
    for name in ALLOWLIST:
        leaf = profile / name
        plain_path(leaf, session)
        for p in tree_files(leaf):
            if p.stat().st_nlink != 1:
                raise ValueError('Shared cache payload changed')
            actual.append({'path': str(p.relative_to(profile)), 'bytes': p.stat().st_size})
    if actual != plan['cache_payloads']:
        raise ValueError('Cache inventory changed before removal')
    removed = [] if removed is None else removed
    for entry in actual:
        p = profile / entry['path']
        plain_path(p, profile)
        p.unlink()
        removed.append(entry)
    for name in ALLOWLIST:
        leaf = profile / name
        for parent, dirs, files in os.walk(leaf, topdown=False):
            if files:
                raise ValueError('Unexpected cache file appeared')
            plain_path(Path(parent), profile)
            if not Path(parent).is_relative_to(leaf):
                raise ValueError('Directory removal escaped planned leaf')
            Path(parent).rmdir()
    if (benchmark_link_digest(session, plan['benchmark_source']) != plan['preserved_benchmark_link_sha256']
            or evidence_snapshot(session, profile, plan['benchmark_source']) != plan['preserved_evidence_sha256']):
        raise ValueError('Protected evidence changed during cache removal')
    return removed


def identity(module, root, study, state):
    if any(row['status'] != 'completed' for row in state['sessions'][:5]):
        raise ValueError('Maintenance was introduced only after the first five completed quality sessions')
    if study['python_identity'] != {'version': sys.version, 'executable_sha256': digest(Path(sys.executable).resolve())}:
        raise ValueError('Pinned interpreter differs')
    for relative, sha in study['source_files'].items():
        if digest(REPO / relative) != sha:
            raise ValueError('Frozen source changed')
    return {'schema_version': 1, 'study_sha256': state['study_sha256'],
            'introduced_after_quality_sessions': 5,
            'source_files': {name: digest(REPO / name) for name in SOURCE_PATHS},
            'allowlist': list(ALLOWLIST), 'limits': module.LIMITS,
            'deadline_epoch_seconds': study['deadline_epoch_seconds'],
            'limits_unchanged': True}


def publish(module, root, out):
    ledger = root / 'profile-maintenance'
    ident = read(ledger / 'identity.json')
    operations = []
    for path in sorted(ledger.glob('quality-*/result.json')) + sorted(ledger.glob('resource-*/result.json')):
        result = read(path)
        operations.append({k: result[k] for k in ('session_key', 'status', 'removed_file_count',
            'removed_logical_bytes', 'charged_seconds', 'preserved_evidence_sha256', 'preserved_benchmark_link_sha256', 'failure_type')})
    value = {k: ident[k] for k in ('schema_version', 'study_sha256', 'introduced_after_quality_sessions', 'source_files', 'limits_unchanged')}
    state = module.load(root)[2]
    admissions = [v for k, v in state.get('action_attempts', {}).items() if k.startswith('profile_maintenance_admission:')]
    publications = [v for k, v in state.get('action_attempts', {}).items() if k.startswith('profile_maintenance_publication:')]
    value.update(kind='fresh_profile_cache_maintenance', operations=operations,
        admission_charged_seconds=sum(v.get('charged_seconds', 0) for v in admissions),
        admission_failures=sum(v.get('status') == 'failed' for v in admissions),
        publication_charged_seconds=sum(v.get('charged_seconds', 0) for v in publications),
        scope='Later operational cleanup of one local-origin IndexedDB blob cache in each completed dedicated browser profile.',
        notes=['Introduced after the first five quality sessions; this was not part of the original frozen protocol.',
               'Only completed, independently owned profiles with recorded cleanup and no live process owner are eligible.',
               'All non-profile scientific evidence, browser Last Version, Local State and IndexedDB leveldb metadata are preserved and hash checked.',
               'Every maintenance action is separately reserved and charged under the original local and calendar limits; no generation is retried.',
               'This changes retained disk cache only after a session ends. It does not establish constant OS disk-cache or host conditions.'])
    return value


def publish_charged(module, root, out):
    began = time.monotonic()
    root, study, state = module.load(root)
    n = sum(k.startswith('profile_maintenance_publication:') for k in state['action_attempts'])
    kind = f'profile_maintenance_publication:{n + 1}'
    seconds = module.begin_action(root, study, state, kind, 60)
    error = None; value = None
    try:
        with module.bounded_python(max(.001, seconds - 2)):
            value = publish(module, root, out)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    # Charge a conservative two-second allowance for final receipt/public output I/O.
    module.finish_action(root, state, kind, time.monotonic() - began + 2, error)
    value['publication_charged_seconds'] += state['action_attempts'][kind]['charged_seconds']
    try:
        with module.bounded_python(2):
            module.atomic(out, value)
    except BaseException as exc:
        state['halted'] = True
        state['action_attempts'][kind].update(status='failed', error=type(exc).__name__ + ': publication finalization failed')
        state['stages'][-1].update(status='failed', error=type(exc).__name__ + ': publication finalization failed')
        module.atomic(root / 'state.json', state)
        raise


def bind(module, root, out):
    began = time.monotonic()
    root, study, state = module.load(root)
    ledger = root / 'profile-maintenance'
    count = sum(k.startswith('profile_maintenance_admission:') for k in state['action_attempts'])
    kind = f'profile_maintenance_admission:{count + 1}'
    seconds = module.begin_action(root, study, state, kind, 300)
    error = None
    try:
        with module.bounded_python(seconds):
            ident = identity(module, root, study, state)
            if (ledger / 'identity.json').exists():
                if read(ledger / 'identity.json') != ident:
                    raise ValueError('Maintenance source or immutable identity differs')
                for receipt in ledger.glob('*/result.json'):
                    result = read(receipt)
                    if result['status'] != 'completed':
                        raise ValueError('Previous maintenance failed; no retry')
                    row = next(r for r in state['sessions'] if r['key'] == result['session_key'])
                    session = root / 'sessions' / row['key']
                    profile = session / 'execution' / ('baseline-1' if row['artifact_key']=='pristine' else 'candidate-1') / 'chrome-profile'
                    if row['status'] != 'completed' or evidence_snapshot(session, profile, study['benchmark_source']) != result['preserved_evidence_sha256'] or benchmark_link_digest(session, study['benchmark_source']) != result['preserved_benchmark_link_sha256']:
                        raise ValueError('Previously maintained evidence changed')
            else:
                write_once(ledger / 'identity.json', ident)
                write_once(ledger / 'introduction.json', {'state_at_admission_sha256': digest(root / 'state.json'),
                    'zero_target_calls': True, 'explicit_post_freeze_maintenance': True,
                    'recorded_epoch_seconds': time.time()})
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    module.finish_action(root, state, kind, time.monotonic() - began, error)
    publish_charged(module, root, out)


def prune_one(module, root, key, out):
    began = time.monotonic()
    root, study, state = module.load(root)
    ledger = root / 'profile-maintenance'
    if read(ledger / 'identity.json')['source_files']['scripts/maintain_fresh_profiles.py'] != digest(Path(__file__)):
        raise ValueError('Maintenance helper changed')
    operation = ledger / key
    if (operation / 'result.json').exists():
        result = read(operation / 'result.json')
        if result['status'] != 'completed':
            raise ValueError('Failed maintenance cannot be repeated')
        return result
    kind = 'profile_cache_prune:' + key
    seconds = module.begin_action(root, study, state, kind, 300)
    row = next(r for r in state['sessions'] if r['key'] == key)
    error = None; plan = None; removed = []
    try:
        with module.bounded_python(seconds):
            # begin_action reserves ownership in state; the saved pre-action session
            # must be idle. This copy is only for the cleanup admission predicate.
            idle = dict(state, active=None)
            plan = plan_cleanup(root, study, idle, row, module.verify_session)
            write_once(operation / 'intent.json', {'study_sha256': state['study_sha256'],
                'helper_sha256': digest(Path(__file__)), 'started_epoch_seconds': time.time(), **plan})
            remove_planned(plan, root / 'sessions' / key, removed=removed)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    failure = None
    try:
        module.finish_action(root, state, kind, time.monotonic() - began + 2, error)
    except BaseException as exc:
        failure = exc
    charged = state['action_attempts'][kind]
    result = {'session_key': key, 'status': charged['status'],
              'removed_file_count': len(removed), 'removed_logical_bytes': sum(x['bytes'] for x in removed),
              'charged_seconds': charged['charged_seconds'], 'preserved_evidence_sha256': (plan or {}).get('preserved_evidence_sha256', {}),
              'failure_type': charged['error'].split(':', 1)[0] if charged.get('error') else None,
              'error': charged.get('error'),
              'preserved_benchmark_link_sha256': (plan or {}).get('preserved_benchmark_link_sha256')}
    try:
        with module.bounded_python(2):
            write_once(operation / 'result.json', result)
            if failure:
                module.atomic(out, publish(module, root, out))
    except BaseException:
        state['halted'] = True
        module.atomic(root / 'state.json', state)
        raise
    if failure:
        raise failure
    publish_charged(module, root, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prune', 'run-rest'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=REPO / 'reports/fresh-baseline-r4-v1/result.json')
    parser.add_argument('--maintenance-out', type=Path, default=REPO / 'reports/fresh-baseline-r4-v1/maintenance.json')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    from weight_space_lab import fresh_baseline_study as module
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['WSL_NATIVE_PYTHON'] = sys.executable
    root = args.root.resolve()
    with module._lock(root):
        bind(module, root, args.maintenance_out)
        root, study, state = module.load(root)
        module.available(root, study, state)
        for row in state['sessions']:
            if row['status'] == 'completed':
                print(json.dumps(prune_one(module, root, row['key'], args.maintenance_out)), flush=True)
        if args.command == 'run-rest':
            root, study, state = module.load(root)
            if not state['candidate_binding_sha256']:
                raise ValueError('Run-rest requires the existing once-bound R4 candidate')
            for row in state['sessions'][5:]:
                if row['status'] == 'completed':
                    continue
                result = module.run_one(root, row['key'])
                print(json.dumps(result), flush=True)
                module.export(root, args.out)
                if result['status'] != 'completed':
                    return
                print(json.dumps(prune_one(module, root, row['key'], args.maintenance_out)), flush=True)
        publish_charged(module, root, args.maintenance_out)
        module.export(root, args.out)


if __name__ == '__main__':
    main()
