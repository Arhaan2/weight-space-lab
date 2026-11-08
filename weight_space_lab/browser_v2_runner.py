"""Arhaan2: local v2 supervisor. Acknowledges only durable controller gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .browser_check import _collector, _stop_process
from .browser_integration_v2 import Controller
from .inventory import read_json
from .native_runner import _lock


def durable_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)  # Atomic appearance and no replacement of an earlier ACK.
    finally:
        os.unlink(temp)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def verify_generation_evidence(target, expected, code_sha):
    proof = read_json(target/'fetch-preflight.json')
    if (proof.get('status') != 'passed' or proof.get('engine_loads') != 0
            or proof.get('generation_dispatches') != 0 or proof.get('code_version_sha256') != code_sha
            or proof.get('requested_artifact_sha256') != expected['convertedArtifactSha256']):
        raise ValueError('Actual browser fetch-only proof differs')
    events = [json.loads(line) for line in (target/'events.jsonl').read_text().splitlines()]
    loaded = [e['data'] for e in events if e['type'] == 'task-artifact-loaded']
    if len(loaded) != 1:
        raise ValueError('Exactly one verified loaded artifact required')
    actual = loaded[0]
    for key, value in {'modelId': expected['model']['id'], 'modelUrl': expected['model']['modelUrl'],
                       'modelLib': expected['model']['modelLib'], 'sourceCheckpointSha256': expected['sourceCheckpointSha256'],
                       'convertedArtifactSha256': expected['convertedArtifactSha256'], 'allAssetsVerified': True,
                       'runtime': 'WebLLM 0.2.86', 'cacheBackend': 'indexeddb', 'executionBackend': 'webgpu',
                       'serverInferenceFallback': False}.items():
        if actual.get(key) != value:
            raise ValueError('Loaded artifact/runtime differs: '+key)
    if actual.get('deviceCount', 0) < 1:
        raise ValueError('Local WebGPU device not observed')
    assets = {e['data']['url']:e['data']['sha256'] for e in events if e['type']=='asset-verified'}
    if assets != {a['url']:a['sha256'] for a in expected['assets']}:
        raise ValueError('Actually loaded asset hashes differ')
    effective = [e['data'] for e in events if e['type']=='task-effective-settings']
    if len(effective) != 1 or effective[0].get('contextWindow') != 4096 or effective[0].get('maxHistorySize') != 1:
        raise ValueError('Effective context settings differ')
    generation = effective[0]['generation']
    wanted = {'temperature':0, 'top_p':1, 'seed':42, 'max_tokens':256,
              'presence_penalty':0, 'frequency_penalty':0, 'repetition_penalty':1,
              'n':1, 'ignore_eos':False, 'extra_body':{'enable_thinking':False}}
    if generation != wanted:
        raise ValueError('Effective generation settings differ')


def sample_resources(collector, process, retained):
    values, unavailable = collector.host_signals()
    record = {'epoch_ms': time.time()*1000, 'host_scope': 'host-wide, includes unrelated workloads',
              'host': values, 'unavailable': unavailable, 'owned_processes': []}
    if process:
        table = collector.process_table()
        retained.update(collector.descendant_pids(table, process.pid, retained))
        for pid in sorted(retained & table.keys()):
            try:
                record['owned_processes'].append({'pid': pid, 'role': collector.process_role(collector.process_args(pid)),
                    'scope': 'owned process footprint/RSS; shared memory may be counted more than once',
                    **collector.usage(pid)})
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
    return record


def deadline_watchdog(process, deadline_mono, stopped, triggered, target):
    """Independent cleanup timer; slow telemetry cannot extend owned execution."""
    if stopped.wait(max(0, deadline_mono-time.monotonic()-12)):
        return
    triggered.set()
    actions = []
    for sig, delay in ((signal.SIGTERM, 3), (signal.SIGKILL, 0)):
        try:
            os.killpg(process.pid, sig)
            actions.append({'signal': int(sig), 'epoch_ms': time.time()*1000})
        except ProcessLookupError:
            pass
        if delay:
            stopped.wait(delay)
    durable_json(target/'deadline-watchdog.json', {'deadline_monotonic':deadline_mono, 'actions':actions})


def execute(root, arm, attempt_id, code_sha):
    root = Path(root).resolve()
    benchmark = root/'benchmark'
    target = root/'execution'/attempt_id
    target.mkdir(parents=True, mode=0o700)  # Never reuse an attempt directory.
    config = root/(arm+'-config.json')
    collector = _collector(benchmark)
    process, error, terminal, cleanup = None, None, None, {}
    retained, handled = set(), set()
    watchdog, watchdog_stopped, watchdog_triggered = None, threading.Event(), threading.Event()
    began = time.monotonic()
    with _lock(root), Controller(root) as controller:
        if controller.summary()['stopped']:
            raise ValueError('V2 is terminal; no further execution')
        controller._version(code_sha)
        try:
            with (target/'stdout.txt').open('x') as out, (target/'stderr.txt').open('x') as err, (target/'resources.jsonl').open('x') as resources:
                next_sample = 0
                while True:
                    if time.monotonic() >= next_sample:
                        record = sample_resources(collector, process, retained)
                        resources.write(json.dumps(record)+'\n')
                        resources.flush()
                        os.fsync(resources.fileno())
                        next_sample = time.monotonic()+5
                        values = record['host']
                        if values.get('hostMemoryPressureLevel') is None or values.get('hostThermalState') is None:
                            error = 'resource_state_unknown'
                            break
                        if values['hostMemoryPressureLevel'] >= 4 or values['hostThermalState'] >= 2:
                            error = 'host_resource_stop'
                            break
                    if process is None:
                        command = ['node', 'scripts/run-task-evaluation.mjs', str(config), str(target),
                                   '--attempt-id', attempt_id, '--code-version-sha', code_sha]
                        process = subprocess.Popen(command, cwd=benchmark, stdout=out, stderr=err, start_new_session=True)
                        if controller.deadline()['deadline_monotonic'] is not None:
                            watchdog = threading.Thread(target=deadline_watchdog, args=(process, controller.deadline()['deadline_monotonic'], watchdog_stopped, watchdog_triggered, target), daemon=True)
                            watchdog.start()
                    if watchdog_triggered.is_set():
                        error = 'global_deadline_cleanup_reserve'
                        break
                    if controller.deadline()['started_epoch_ms'] is not None and controller.remaining_seconds() <= 12:
                        error = 'global_deadline_cleanup_reserve'
                        break
                    if controller.deadline()['started_epoch_ms'] is None and time.monotonic()-began > 300:
                        error = 'pre_engine_preparation_timeout'
                        break
                    for kind in ('load', 'generation'):
                        intent_path = target/(kind+'-intent.json')
                        if kind not in handled and intent_path.exists():
                            intent = read_json(intent_path)
                            if kind == 'generation':
                                verify_generation_evidence(target, read_json(config), code_sha)
                            # Controller commits SQLite FULL-synchronous intent/reservation before ACK.
                            ack = controller.handle_intent(intent, arm=arm, attempt_id=attempt_id, code_sha256=code_sha)
                            if kind == 'load' and watchdog is None:
                                watchdog = threading.Thread(target=deadline_watchdog, args=(process, ack['deadline_monotonic'], watchdog_stopped, watchdog_triggered, target), daemon=True)
                                watchdog.start()
                            durable_json(target/(kind+'-ack.json'), ack)
                            handled.add(kind)
                            print(json.dumps({'event': kind+'_durably_granted', 'arm': arm,
                                  'attempt': attempt_id, 'remaining_seconds': controller.remaining_seconds()}), flush=True)
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
        except BaseException as exc:
            error = type(exc).__name__+': '+str(exc)
        finally:
            if process:
                cleanup.update(_stop_process(process, min(10, max(1, controller.remaining_seconds()))))
            else:
                cleanup['controller_exit_verified'] = True
            watchdog_stopped.set()
            if watchdog:
                watchdog.join(timeout=4)
        if (target/'result.json').exists():
            terminal = read_json(target/'result.json')
            cleanup.update(terminal.get('cleanup', {}))
        pid = (terminal or {}).get('driver', {}).get('chrome_pid')
        cleanup['chrome_os_exit_verified'] = not alive(pid) if pid else None
        cleanup['retained_owned_pids_alive'] = [p for p in sorted(retained) if alive(p)]
        if cleanup['retained_owned_pids_alive'] or cleanup['chrome_os_exit_verified'] is False:
            error = error or 'owned_process_cleanup_unconfirmed'
        if cleanup.get('chrome_exit_confirmed') and cleanup.get('chrome_os_exit_verified'):
            # Dedicated browser exit also releases a worker whose completion message was lost.
            cleanup['worker_terminated'] = True
            cleanup['worker_release_scope'] = 'dedicated browser OS exit plus page worker termination when observed'
        rows = {a['attempt_id']: a for a in controller.attempts()}
        dispatched = attempt_id in rows and rows[attempt_id]['generation_state'] != 'not_dispatched'
        intent_absent = not (target/'generation-intent.json').exists()
        ack_absent = not (target/'generation-ack.json').exists()
        if not intent_absent and not dispatched:
            error = error or 'generation_intent_without_durable_resolution'
        expected = read_json(config)
        if terminal:
            for field, key in (('configuration_sha256','configurationSha256'), ('source_checkpoint_sha256','sourceCheckpointSha256'), ('converted_artifact_sha256','convertedArtifactSha256')):
                if terminal.get(field) != expected[key]:
                    error = error or 'terminal_identity_mismatch'
        text = json.dumps(terminal or {}).lower()
        oom = any(x in text for x in ('out of memory', 'out-of-memory')) or any(
            e.get('type') == 'event' and e.get('data', {}).get('type') == 'device-lost'
            for e in (terminal or {}).get('events', []))
        tokens = (terminal or {}).get('generatedTokens')
        success = bool(not error and not oom and terminal and terminal.get('status') == 'completed'
                       and dispatched and len(terminal.get('outputs', [])) == 1)
        if success:
            sample = terminal['outputs'][0]
            success = sample.get('local_webgpu_inference_observed') is True and sample.get('webgpu_submitted_batches', 0) > 0
            if not success:
                error = 'local_webgpu_execution_unconfirmed'
        failure_class = None if success else ('oom_or_device_loss' if oom else error or 'unclassified_runtime_failure')
        # No heuristic retry classification. A human-readable diagnosed URL/cache failure
        # must be explicitly recorded before any retry; unknown failures terminate here.
        generation_seconds = None
        if terminal and terminal.get('outputs'):
            latency = terminal['outputs'][0].get('response_latency_ms')
            if isinstance(latency, (int, float)):
                generation_seconds = latency/1000
        evidence = {'terminal_result_path': str(target/'result.json'), 'driver_exit_code': process.returncode if process else None,
                    'confirmed_pre_generation': not dispatched and intent_absent and ack_absent,
                    'generation_intent_absent': intent_absent, 'generation_ack_absent': ack_absent,
                    'unknown_execution_state': not bool(terminal) or (not intent_absent and not dispatched),
                    'oom_or_device_loss': oom, 'resource_stop': error in ('host_resource_stop','resource_state_unknown'),
                    'supervisor_error': error, 'elapsed_seconds': time.monotonic()-began}
        outcome = 'completed' if success else 'failed'
        reason = str((terminal or {}).get('done', {}).get('reason', (terminal or {}).get('execution_error', '')))
        if (not success and not error and not oom and not dispatched and intent_absent and ack_absent and terminal
                and terminal.get('done', {}).get('failure_class') == 'confirmed-pre-generation-asset-routing-or-fetch'):
            failures = [e['data'] for e in terminal.get('events', []) if e.get('type') == 'failure-classification']
            if len(failures) == 1 and failures[0].get('application_generation_attempted') is False:
                fault = failures[0].get('asset_failure') or {}
                if fault.get('url') and fault.get('message') and fault.get('message') in reason:
                    outcome, failure_class = 'load_failed', 'fetch'
                    evidence['diagnosis'] = 'Hash-guarded runtime asset failure before generation: '+reason
                    evidence['asset_failure'] = fault
        if attempt_id in rows:
            result = controller.finish_attempt(attempt_id, outcome=outcome,
                failure_class=failure_class, evidence=evidence, cleanup=cleanup,
                generated_tokens=tokens, generation_seconds=generation_seconds)
        else:
            result = {'outcome':'pre_load_failure', 'failure_class': failure_class, 'evidence':evidence, 'cleanup':cleanup}
        durable_json(target/'supervisor-result.json', result)
        durable_json(target/'accounting-after.json', controller.summary())
        print(json.dumps({'arm':arm, 'attempt':attempt_id, 'outcome': result['outcome'],
                         'failure_class':result.get('failure_class'), 'generated_tokens': tokens,
                         'remaining_seconds':controller.remaining_seconds()}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=['baseline','candidate'], required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--code-version-sha', required=True)
    args = parser.parse_args()
    execute(args.root,args.arm,args.attempt_id,args.code_version_sha)


if __name__ == '__main__':
    main()
