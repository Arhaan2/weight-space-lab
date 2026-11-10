#!/usr/bin/env python3
"""Publish a new sanitized recovery comparison from preserved evidence only.

No grader, model, checkpoint editor or budget controller is imported or run.
The baseline and earlier stopped-stage reports remain unchanged. Arhaan2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from report_ifeval_candidate import METRICS, fields, file_sha, privacy_check, read, sha


def resource_summary(directory, events):
    rows = [json.loads(line) for line in (directory / 'resources.jsonl').read_text().splitlines()]
    host, roles = {}, {}
    for row in rows:
        for key, value in row.get('host', {}).items():
            if type(value) in (int, float):
                prior = host.setdefault(key, {'minimum': value, 'maximum': value, 'samples': 0})
                prior['minimum'] = min(prior['minimum'], value)
                prior['maximum'] = max(prior['maximum'], value)
                prior['samples'] += 1
        for process in row.get('owned_processes', []):
            role = roles.setdefault(process['role'], {})
            for key in ('footprintBytes', 'rssBytes', 'wiredBytes'):
                value = process.get(key)
                if type(value) in (int, float):
                    role[key] = max(role.get(key, 0), value)
    points = [event['data'] for event in events if event['type'] == 'point']
    peaks = {}
    for key in ('gpuBufferBytes', 'wasmBytes', 'heapBytes', 'eventLoopDelayMs'):
        values = [point[key] for point in points if type(point.get(key)) in (int, float)]
        peaks[key] = max(values) if values else None
    return {
        'host_scope': 'host-wide samples include unrelated workloads; not model-attributed physical GPU memory',
        'host_sampled_numeric_ranges': host, 'host_sample_count': len(rows),
        'owned_process_scope': 'maximum single observed process per role; RSS/footprint can include shared memory',
        'owned_role_sampled_maxima': roles,
        'worker_scope': 'GPUBuffer descriptor accounting and WASM address space are not physical VRAM',
        'worker_sampled_maxima': peaks, 'worker_sample_count': len(points),
        'worker_signals': [fields(event['data'], ('id', 'label', 'scope', 'source', 'unit', 'status', 'reason'))
                           for event in events if event['type'] == 'signal'],
        'initialization': [fields(event['data'], ('applicationGenerationCalls', 'applicationWarmupCalls',
            'observedWebGPUDevices', 'queueSubmittedBatchesDuringInitialization', 'scope',
            'runtimeAutomaticModelWarmup')) for event in events if event['type'] == 'initialization-computation'],
        'worker_series_scope': 'one complete worker run including load, warmup, scored requests and cleanup; the client repeats this shared series on each terminal output',
        'sampling_limit': 'Observed sample maxima do not establish absolute peaks; unavailable metrics remain unavailable.',
    }


def candidate_summary(root):
    manifest = read(root / 'manifest.json')
    config = read(root / 'baseline-config.json')
    graded = read(root / 'candidate/graded.json')
    directory = root / 'execution/candidate-1'
    execution = read(directory / 'supervisor-result.json')
    terminal = read(directory / 'result.json')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    assert graded['manifest_sha256'] == sha(manifest)
    assert read(root / 'ready.json') == {'manifest_sha256': sha(manifest)}
    assert manifest['checkpoint']['sha256'] == config['sourceCheckpointSha256'] == terminal['source_checkpoint_sha256']
    assert manifest['artifact_sha256'] == config['convertedArtifactSha256'] == terminal['converted_artifact_sha256']
    actual = {row['item_id']: row for row in terminal['outputs']}
    expected = {row['item_id']: row for row in execution['outputs']}
    assert len(actual) == len(terminal['outputs']) and actual.keys() == expected.keys()
    for item, row in actual.items():
        for key in ('raw_output', 'generated_tokens', 'input_tokens', 'finish_reason', 'response_latency_ms'):
            assert row.get(key) == expected[item].get(key), 'terminal/supervisor observation differs'
    loads = [event['data'] for event in events if event['type'] == 'loaded']
    cleanup = fields(execution['cleanup'], ('process_group_stop_requested', 'controller_exit_verified',
        'controller_returncode', 'unload_confirmed', 'worker_terminated', 'forced_termination',
        'chrome_exit_confirmed', 'local_harness_exit_confirmed', 'owned_browser_and_worker_resources_released',
        'release_scope', 'chrome_os_exit_verified'))
    cleanup['retained_owned_processes_alive'] = len(execution['cleanup']['retained_owned_pids_alive'])
    book = execution['generation_budget']
    result = {
        'status': execution['status'], 'runtime_failure_recorded': execution.get('error') is not None,
        'checkpoint': manifest['checkpoint'], 'artifact_sha256': manifest['artifact_sha256'],
        'manifest_sha256': sha(manifest), 'configuration_sha256': config['configurationSha256'],
        'graded_evidence_sha256': file_sha(root / 'candidate/graded.json'),
        'terminal_result_sha256': file_sha(directory / 'result.json'),
        'events_sha256': file_sha(directory / 'events.jsonl'),
        'summary': graded['summary'], 'load_ms': loads[0] if len(loads) == 1 else None,
        'budget': {'limits': manifest['limits'],
            **fields(execution, ('engine_load_attempts', 'warmup_dispatches', 'scored_dispatches',
                'generation_dispatches', 'first_load_through_cleanup_seconds', 'supervisor_seconds',
                'heldout_generations', 'frontier_requests')),
            'reserved': book['reserved'], 'observed': book['observed'], 'charged': book['charged'],
            'terminal': book['cancelled']},
        'warmup': [fields(row, ('generated_tokens', 'input_tokens', 'response_latency_ms', 'finish_reason',
            'assistant_prefix_tokens', 'request_max_tokens')) for row in execution['warmup_outputs']],
        'resources': resource_summary(directory, events), 'cleanup': cleanup,
    }
    return result, manifest, config, graded


def recovery_summary(root):
    """Whitelist repair evidence; public diagnostics never expose machine paths."""
    repair = read(root / 'recovery/repair-record.json')
    environment = read(root / 'recovery/environment-verification.json')
    intent = read(root / 'recovery/intent.json')
    outcome = read(root / 'recovery/result.json')
    assert repair['verification']['environment_verification_sha256'] == file_sha(root / 'recovery/environment-verification.json')
    assert repair['failure_file_sha256'] == file_sha(root / 'candidate-stage-failure.json')
    assert outcome['intent_sha256'] == file_sha(root / 'recovery/intent.json')
    assert intent['repair_record_sha256'] == file_sha(root / 'recovery/repair-record.json')
    assert environment['passed'] and environment['expected_environment_active']
    assert environment['all_frozen_dependency_pins_match'] and environment['returncode'] == 0
    assert environment['helper_source_sha256'] == repair['repaired_launcher_sha256']
    assert environment['changed_subprocess_cwd_verified']
    observed = environment['observed']
    prefix = Path(observed['sys_prefix'])
    assert observed['sys_prefix'] != observed['sys_base_prefix']
    assert observed['sys_executable'] == environment['invoked_interpreter']
    executable_relative = str(Path(environment['invoked_interpreter']).relative_to(prefix))
    locations = {name: str(Path(location).relative_to(prefix)) for name, location in observed['module_locations'].items()}
    assert intent['manual_conversion_continuations'] == outcome['manual_conversion_continuations'] == 1
    assert intent['automatic_retries'] == outcome['automatic_retries'] == 0
    assert intent['imports'] == outcome['imports'] == 0
    assert intent['import_ledger_before_sha256'] == outcome['import_ledger_after_sha256']
    assert intent['baseline_ledger_before_sha256'] == outcome['baseline_ledger_after_sha256']
    return {
        'bug': 'The original converter launcher dereferenced the virtual-environment Python symlink and executed system Python for its package check.',
        'fix': 'A separately versioned launch helper makes the configured interpreter absolute before subprocess directory changes without dereferencing it; package checks and conversion use that same executable.',
        'baseline_frozen_sources_preserved': repair['source_revision']['historical_source_unchanged'],
        'authorization_scope': 'One explicitly authorized manual recovery after the documented precheck failure; no re-import and no automatic retries.',
        'source_revision': repair['source_revision'],
        'original_launcher_source_sha256': repair['original_source_sha256'],
        'repaired_launcher_source_sha256': repair['repaired_launcher_sha256'],
        'repair_record_sha256': file_sha(root / 'recovery/repair-record.json'),
        'environment_verification_sha256': file_sha(root / 'recovery/environment-verification.json'),
        'continuation_intent_sha256': file_sha(root / 'recovery/intent.json'),
        'continuation_result_sha256': file_sha(root / 'recovery/result.json'),
        'focused_tests': fields(repair['verification'], ('focused_tests_passed', 'focused_tests_log_sha256')),
        'regression_test': 'A real temporary symlink-based virtual environment is launched through the production helper after a working-directory change; a module installed only in that environment must remain importable.',
        'environment': {
            **fields(environment, ('passed', 'through_production_helper', 'expected_environment_active',
                'all_frozen_dependency_pins_match', 'dependency_pin_count', 'expected_converter_packages',
                'lockfile_sha256', 'helper_source_sha256', 'changed_subprocess_cwd_verified',
                'target_generations', 'engine_loads', 'installations', 'dependency_upgrades')),
            'invoked_interpreter_relative_to_environment': executable_relative,
            'observed_executable_matches_invoked': True,
            'sys_prefix_is_expected_environment': True,
            'sys_prefix_differs_from_sys_base_prefix': True,
            'module_locations_relative_to_environment': locations,
            'all_observed_dependency_versions': observed['packages'],
            'production_cache_and_environment_reused': True,
            'initial_offline_probe_correction': environment['initial_offline_probe_correction'],
            'path_privacy': 'Exact interpreter, sys.prefix, sys.base_prefix and import paths are retained privately; public paths are relative to the verified virtual environment.',
        },
        'continuation': fields(outcome, ('status', 'returncode', 'continuation_seconds',
            'manual_conversion_continuations', 'automatic_retries', 'imports', 'converted', 'packaged',
            'bound', 'browser_result_present', 'comparison_present')),
        'prior_import_and_baseline_ledgers_unchanged': True,
    }


def report(root, public):
    root, public = Path(root).resolve(), Path(public)
    if any((public / name).exists() for name in ('README.md', 'result.json')):
        raise FileExistsError('recovery report files must be new')
    verified = read(public / 'verification.json')
    assert verified['passed'] is True and all(verified['checks'].values())
    prior = read(root.parent.parent / 'reports/ifeval-browser-v1-candidate/result.json')
    baseline_public = read(root.parent.parent / 'reports/ifeval-browser-v1/result.json')
    baseline = fields(baseline_public, ('checkpoint', 'artifact_sha256', 'configuration_sha256', 'manifest_sha256',
        'summary', 'resources', 'cleanup', 'budget', 'warmup', 'load_ms'))
    bg = read(root / 'baseline/graded.json')
    bm = read(root / 'manifest.json')
    bc = read(root / 'baseline-config.json')
    candidate, cm, cc, cg = candidate_summary(root / 'candidate-comparison')
    for key in ('settings', 'items', 'warmupPrompt', 'totalLimitMs'):
        assert bc[key] == cc[key], 'matched frozen profile differs'
    assert bm['policy'] == cm['policy'] and bm['grader'] == cm['grader']
    assert bg['summary'] == baseline['summary']
    paired = read(root / 'candidate-comparison/comparison.json')
    ready = read(root / 'candidate-comparison/comparison-ready.json')
    assert ready['baseline_graded_file_sha256'] == file_sha(root / 'baseline/graded.json')
    assert ready['paired_manifest_sha256'] == sha(read(root / 'candidate-comparison/paired-analysis-manifest.json'))
    imported = read(root / 'handoff/import-result.json')
    source = read(root / 'candidate/deployed/source-audit.json')
    artifact = read(root / 'candidate/deployed/candidate-artifact.json')
    effective = read(root / 'candidate/deployed/deployment-difference.json')
    conversion = read(root / 'candidate/converted-source-binding.json')
    assert imported['accepted'] is True
    assert source['base_sha256'] == imported['audit']['original_sha256'] == baseline['checkpoint']['sha256']
    assert source['result_sha256'] == imported['audit']['result_sha256'] == candidate['checkpoint']['sha256']
    assert artifact['artifact_sha256'] == candidate['artifact_sha256']
    assert artifact['effective_deployment_audit_sha256'] == sha(effective)
    assert artifact['conversion_binding_sha256'] == sha(conversion)
    assert ready['candidate_artifact_file_sha256'] == file_sha(root / 'candidate/deployed/candidate-artifact.json')
    source_public = fields(source, ('base_sha256', 'result_sha256', 'total_stored_coordinates', 'targeted_coordinates',
        'changed_stored_entries', 'changed_numerical_entries', 'bit_only_changed_entries', 'changed_fraction',
        'l2_norm', 'max_abs_change', 'unchanged_tensor_count', 'budget_basis'))
    source_public['per_tensor'] = {name: fields(row, ('dtype', 'shape', 'base_sha256', 'result_sha256',
        'changed_stored_entries', 'changed_numerical_entries', 'targeted_coordinates', 'l2_norm', 'max_abs_change'))
        for name, row in source['per_tensor'].items()}
    effective_public = fields(effective, ('quantization', 'group_size', 'parameter_contract_verified', 'parameter_count',
        'changed_tensor_names', 'unchanged_tensor_count', 'conversion_erases_edit', 'per_source_tensor'))
    effective_public['aggregate_counts'] = {key: sum(row[key] for row in effective['per_source_tensor'].values()) for key in ('quantized_weight_words_changed', 'quantized_nibbles_changed', 'quantization_scales_changed', 'decoded_values_changed', 'decoded_changed_outside_source_target_columns', 'decoded_changed_outside_source_target_groups')}
    order = [row['item_id'] for row in bc['items']]
    indexes = [{row['item_id']: row for row in data['grades']} for data in (bg, cg)]
    records = [{row['item_id']: row for row in data['records']} for data in (bg, cg)]
    per_item, counts = [], {mode: Counter() for mode in ('strict', 'loose')}
    for position, item in enumerate(order, 1):
        row = {'schedule_position': position}
        for mode in ('strict', 'loose'):
            values = [bool(index[item]['metrics'][f'prompt_level_{mode}_accuracy'])
                      if index[item]['status'] == 'graded' else None for index in indexes]
            change = 'unresolved' if None in values else 'improved' if values[1] > values[0] else 'regressed' if values[1] < values[0] else 'tied'
            counts[mode][change] += 1
            row[mode] = {'baseline': values[0], 'candidate': values[1], 'change': change}
        for name, index in zip(('baseline', 'candidate'), records):
            record = index[item]
            row[name] = {'status': record['status'], 'generated_tokens': record['token_counts']['output'],
                         'generation_seconds': record['duration_seconds']}
        per_item.append(row)
    for arm, data in (('baseline', baseline), ('candidate', candidate)):
        for name, _ in METRICS:
            metric = data['summary']['metrics'][name]
            projected = paired[arm + '_projection']['official_metrics'][name]
            assert (metric['passed'], metric['scheduled'], metric['value']) == (projected['numerator'], projected['denominator'], projected['value'])
    if not counts['strict']['unresolved']:
        assert all(counts['strict'][key] == paired['comparison']['paired_flips'][key] for key in ('improved', 'regressed', 'tied'))
    result = {
        'schema_version': 1, 'kind': 'ifeval_development_recovery_comparison_report', 'author': 'Arhaan2',
        'scope': 'exploratory on-device workflow validation; editor isolation unverified', 'scientific_result': False,
        'recovery': recovery_summary(root),
        'prior_failure_history': fields(prior, ('status', 'failure', 'consumption')),
        'proposal_provenance': prior['proposal_provenance'],
        'import': {**prior['import'], 'repeated_during_recovery': False},
        'frozen_profile': {'settings': bc['settings'], 'grading_policy': bm['policy'], 'grader': bm['grader'],
            'scheduled_prompts': len(order), 'same_order_tokenizer_template_runtime_warmup_and_extraction_verified': True},
        'baseline': baseline, 'candidate': candidate,
        'conversion': {'status': conversion['status'], 'preparation_seconds': conversion['preparation_seconds'],
            'converter_packages': conversion['converter_packages'], 'converter_subprocess_limit_seconds': 1800,
            'package_check_limit_seconds': 30, 'packaging_seconds': artifact['preparation_seconds'],
            'target_generations': conversion['target_generations'], 'new_proposals': conversion['new_proposals']},
        'saved_checkpoint_audit': source_public, 'deployed_artifact_audit': effective_public,
        'paired': {**fields(paired, ('declared_metric', 'paired_estimand', 'semantic_grading_complete', 'interpretation')),
            **fields(paired['comparison'], ('delta', 'confidence_interval', 'uncertainty_scope', 'weighting')),
            'strict_flips': {key: counts['strict'][key] for key in ('improved', 'regressed', 'tied', 'unresolved')},
            'loose_flips': {key: counts['loose'][key] for key in ('improved', 'regressed', 'tied', 'unresolved')},
            'per_item': per_item},
        'selection': paired['selection'], 'heldout_examples_left_unexecuted': 128,
        'independent_verification': {'passed': verified['passed'], 'checks_passed': sum(verified['checks'].values()),
            'checks_total': len(verified['checks']), 'verification_file_sha256': file_sha(public / 'verification.json')},
        'token_accounting': 'Four injected empty-thinking prefix IDs count toward max_tokens but not reported completion tokens. A 1024 cap permits at most 1020 sampled tokens; the 16-token warmup cap permits at most 12. Reservations retain 1024/16 bounds.',
        'attribution': prior['attribution'],
    }
    assert conversion['command'][0] == read(root / 'recovery/environment-verification.json')['invoked_interpreter']
    assert conversion['converter_packages'] == result['recovery']['environment']['expected_converter_packages']
    assert conversion['launcher_recovery']['repaired_launcher_sha256'] == result['recovery']['repaired_launcher_source_sha256']
    result['recovery']['actual_conversion_used_verified_interpreter'] = True
    request = read(root / 'handoff/request.json')
    private_ids, prompts = [request['request_id'], *order], [row['prompt'] for row in bc['items']]
    privacy_check(result, private_ids, prompts)
    markdown = render(result)
    privacy_check(markdown, private_ids, prompts)
    public.mkdir(parents=True, exist_ok=True)
    with (public / 'result.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (public / 'README.md').open('x') as stream:
        stream.write(markdown)
    return {'written': True, 'selected_arm': result['selection']['selected_arm'], 'strict_flips': result['paired']['strict_flips']}


def render(result):
    base, candidate, paired = result['baseline'], result['candidate'], result['paired']
    selection = result['selection']
    conclusion = 'The candidate is selected' if selection['selected_arm'] == 'candidate' else 'The pristine baseline is retained' if selection['selected_arm'] == 'baseline' else 'No arm is selected because technical execution or grading is incomplete'
    lines = ['# IFEval browser candidate recovery comparison', '',
        f"Arhaan2. {conclusion} under the frozen strict prompt-compliance rule. This is exploratory on-device workflow validation; editor isolation remains unverified. All 128 held-out examples remain unexecuted.", '',
        'The [original baseline](../ifeval-browser-v1/README.md) and [earlier stopped-stage report](../ifeval-browser-v1-candidate/README.md) are preserved. The first conversion-stage package check failed because the launcher dereferenced the virtual-environment interpreter symlink to system Python. A separately versioned helper now makes the configured interpreter absolute without dereferencing it and uses that same executable for environment verification and conversion. The user explicitly authorized one manual continuation; this was not an automatic retry. The imported checkpoint was reused without another import.', '',
        f"Offline verification through the production helper confirmed the configured virtual environment remained active after changing subprocess directory: the invoked executable and observed sys.executable matched, sys.prefix was the expected environment, and sys.prefix differed from sys.base_prefix. All {result['recovery']['environment']['dependency_pin_count']} frozen dependency pins matched; {result['recovery']['focused_tests']['focused_tests_passed']} focused tests passed, including a real temporary virtual-environment regression test with an environment-only module. The actual conversion command used that verified executable. No installations or dependency upgrades were performed.",
        f"Frozen converter packages: " + ', '.join(f"`{name}=={version}`" for name, version in result['conversion']['converter_packages'].items()) + '.',
        f"Original launcher SHA-256: `{result['recovery']['original_launcher_source_sha256']}`. Recovery launcher SHA-256: `{result['recovery']['repaired_launcher_source_sha256']}`. Base Git revision: `{result['recovery']['source_revision']['base_git_commit']}`. The repair is separately hash-bound; baseline frozen provenance was not rewritten.",
        'An initial offline import probe omitted the existing production cache setting and encountered a cache-directory permission error. Its diagnostic was preserved; final environment verification reused the unchanged production cache/environment. This offline probe was not conversion or model inference.', '',
        '| Compliance metric | Baseline | Candidate | Change |', '| --- | ---: | ---: | ---: |']
    for key, label in METRICS:
        b, c = base['summary']['metrics'][key], candidate['summary']['metrics'][key]
        lines.append(f"| {label} | {b['passed']}/{b['scheduled']} ({b['value']:.2%}) | {c['passed']}/{c['scheduled']} ({c['value']:.2%}) | {(c['value']-b['value'])*100:+.3f} pp |")
    strict, loose = paired['strict_flips'], paired['loose_flips']
    ci = paired['confidence_interval']
    lines += ['', f"Strict outcomes: {strict['improved']} improved, {strict['regressed']} regressed, {strict['tied']} tied, {strict['unresolved']} unresolved. Loose outcomes: {loose['improved']} improved, {loose['regressed']} regressed, {loose['tied']} tied, {loose['unresolved']} unresolved.",
        f"The descriptive paired bootstrap 95% interval for the strict difference is {ci['low']*100:+.3f} to {ci['high']*100:+.3f} percentage points (10,000 item resamples, seed 0). It covers this preselected development comparison; it does not establish a causal explanation or held-out generalization.", '',
        '| Execution metric | Baseline | Candidate |', '| --- | ---: | ---: |']
    for label, getter in (
        ('Natural completions', lambda x: x['summary']['execution_status_counts'].get('ok', 0)),
        ('Length truncations', lambda x: x['summary']['execution_status_counts'].get('truncated', 0)),
        ('Unresolved grader errors', lambda x: x['summary']['unresolved_grader_errors']),
        ('Scored output tokens', lambda x: f"{x['summary']['known_generated_tokens']:,}"),
        ('Scored input tokens', lambda x: f"{x['summary']['known_input_tokens']:,}"),
        ('Scored generation seconds', lambda x: f"{x['summary']['generation_seconds']:.3f}"),
        ('Load seconds', lambda x: 'unavailable' if x['load_ms'] is None else f"{x['load_ms']/1000:.3f}"),
        ('Load through cleanup seconds', lambda x: 'unavailable' if x['budget']['first_load_through_cleanup_seconds'] is None else f"{x['budget']['first_load_through_cleanup_seconds']:.3f}"),
    ):
        lines.append(f'| {label} | {getter(base)} | {getter(candidate)} |')
    audit, deployment = result['saved_checkpoint_audit'], result['deployed_artifact_audit']
    lines += ['', f"The saved source audit found {audit['targeted_coordinates']:,} targeted coordinates, {audit['changed_stored_entries']:,} stored-bit changes and {audit['changed_numerical_entries']:,} numerical changes; {audit['unchanged_tensor_count']} other tensors remained unchanged. These source changes are reported separately from converted quantization differences.", '',
        f"Candidate checkpoint SHA-256: `{candidate['checkpoint']['sha256']}`.",
        f"Candidate deployed artifact SHA-256: `{candidate['artifact_sha256']}`.", '',
        '| Source tensor | Targeted | Stored changes | Numerical changes | Decoded q4 changes | Outside target columns | Outside target groups |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, row in audit['per_tensor'].items():
        q = deployment['per_source_tensor'][name]
        lines.append(f"| `{name}` | {row['targeted_coordinates']:,} | {row['changed_stored_entries']:,} | {row['changed_numerical_entries']:,} | {q['decoded_values_changed']:,} | {q['decoded_changed_outside_source_target_columns']:,} | {q['decoded_changed_outside_source_target_groups']:,} |")
    lines += ['', '| Quantized tensor | Weight words changed | Nibbles changed | Scales changed | Decoded max absolute change | Decoded L2 change |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, q in deployment['per_source_tensor'].items():
        lines.append(f"| `{name}` | {q['quantized_weight_words_changed']:,} | {q['quantized_nibbles_changed']:,} | {q['quantization_scales_changed']:,} | {q['decoded_max_abs_change']:.9g} | {q['decoded_l2_change']:.9g} |")
    lines += ['', f"The deployed parameter contract verified all {deployment['parameter_count']} entries; {deployment['unchanged_tensor_count']} stayed byte-identical. Across both affected source tensors, {deployment['aggregate_counts']['decoded_values_changed']:,} decoded values changed, including {deployment['aggregate_counts']['decoded_changed_outside_source_target_columns']:,} outside exact source columns and {deployment['aggregate_counts']['decoded_changed_outside_source_target_groups']:,} outside their quantization groups.", '', 'The source and deployed-artifact audits were verified before generation. Quantization can change values outside the exact source columns within their 32-coordinate groups; group spillover is disclosed above. Unchanged tokenizer, compiled model library and runtime configuration were retained. Full target columns and decoded tensor identities appear in [result.json](result.json).', '',
        '| Resource observation (sampled maximum) | Baseline | Candidate |', '| --- | ---: | ---: |']
    for key, label in (('gpuBufferBytes', 'Worker GPUBuffer descriptor bytes'), ('wasmBytes', 'Worker WASM address-space bytes'), ('heapBytes', 'Worker JS heap bytes'), ('eventLoopDelayMs', 'Worker timer delay ms')):
        values = [x['resources']['worker_sampled_maxima'][key] for x in (base, candidate)]
        display = ['unavailable' if v is None else f'{v:,.3f}' if isinstance(v, float) else f'{v:,}' for v in values]
        lines.append(f'| {label} | {display[0]} | {display[1]} |')
    budget, clean = candidate['budget'], candidate['cleanup']
    lines += ['', 'GPUBuffer totals measure descriptor accounting; WASM bytes measure address space. Neither is physical VRAM. Owned-process RSS/footprint maxima and host-wide ranges remain separately available in the JSON. Process measurements may include shared memory, host measurements include unrelated workloads, and sampled maxima do not establish absolute peaks. One shared worker series is counted once per run, not repeated per output. Timing and resource differences come from separate local runs with host activity; they do not isolate the weight edit as their cause.', '',
        f"Candidate capacity consumed: {budget['engine_load_attempts']}/1 engine-load attempt, {budget['warmup_dispatches']}/1 warmup, {budget['scored_dispatches']}/32 scored dispatches, {budget['generation_dispatches']}/33 total dispatches, and {budget['observed']['generated_tokens']:,}/32,784 reported generated tokens including warmup. Automatic generation retries: {budget['charged']['retries']}; pending requests: {budget['charged']['pending_requests']}; limit violations: {budget['charged']['limit_violations']}. The original successful import remains the sole consumed submission. The earlier failed precheck and this separately authorized manual conversion continuation are both preserved.",
        result['token_accounting'],
        f"The converter retained its existing 1,800-second subprocess limit and 30-second package check. Conversion preparation took {result['conversion']['preparation_seconds']:.3f} seconds and packaging {result['conversion']['packaging_seconds']:.3f} seconds. Browser limits remained 120 seconds per request and 5,400 seconds from engine load through cleanup.",
        f"Candidate execution status: `{candidate['status']}`; runtime failure recorded: `{candidate['runtime_failure_recorded']}`. Cleanup: unload `{clean.get('unload_confirmed')}`, worker termination `{clean.get('worker_terminated')}`, browser OS exit `{clean.get('chrome_os_exit_verified')}`, local harness exit `{clean.get('local_harness_exit_confirmed')}`, remaining owned processes {clean['retained_owned_processes_alive']}. Ordinary failed instructions and truncations remained results under the unchanged grading policy.", '',
        'The frozen comparison retained all 32 development examples in the same order, the same pristine origin, tokenizer, chat template, quantization, browser runtime, warmup, answer extraction and official grading. Thinking was disabled; temperature 0, top-p 1, per-request seed 42, context 4,096 and output cap 1,024 remained unchanged. There were no baseline reruns, held-out generations, new proposals, or experimental frontier requests.', '',
        f"Frozen selection policy: {selection['rule']} Technical execution and grading complete: `{selection['technical_and_grading_complete']}`.", '',
        'The supplied exploratory proposal used file-reading and JSON-parsing tools to ingest the development packet, with no external research, additional weight inspections, or target forward passes in that proposal session. Its selected columns are not a verified causal explanation for repetition. The original request and provenance remain separate, and editor isolation remains unverified.', '',
        f"The [independent recovery verification](verification.json) passed {result['independent_verification']['checks_passed']}/{result['independent_verification']['checks_total']} checks. The original baseline and stopped-stage records remain historical evidence; this report is a new result.", '',
        'Per-item results use 1-based frozen schedule positions. Pass/fail refer to official compliance, not technical execution success.', '',
        '| Position | Strict baseline → candidate | Strict change | Loose baseline → candidate | Loose change | Completion baseline → candidate |', '| ---: | --- | --- | --- | --- | --- |']
    label = lambda value: 'unresolved' if value is None else 'pass' if value else 'fail'
    for row in paired['per_item']:
        s, l = row['strict'], row['loose']
        lines.append(f"| {row['schedule_position']} | {label(s['baseline'])} → {label(s['candidate'])} | {s['change']} | {label(l['baseline'])} → {label(l['candidate'])} | {l['change']} | {row['baseline']['status']} → {row['candidate']['status']} |")
    lines += ['', "Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Existing BrowserLLM work, Gufran's contribution, upstream licenses and scientific attribution remain preserved. Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google Research IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Private prompts, packets, response, authorization and machine paths are excluded.", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/ifeval-browser-v1')
    parser.add_argument('--public', default='reports/ifeval-browser-v1-candidate-recovery')
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.public), indent=2))
