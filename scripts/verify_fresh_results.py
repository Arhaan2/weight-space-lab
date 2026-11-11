"""Independent arithmetic and publication checks for the fresh comparison. No inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

KEYS = ['pristine', 'v1', 'v2-r1', 'v2-r2', 'v2-r3', 'astra-r4']
RESOURCE_ORDER = KEYS + list(reversed(KEYS))
LIMITS = {'engine_loads': 18, 'warmups': 18, 'quality_requests': 192,
          'resource_requests': 720, 'dispatches': 930, 'reserved_target_tokens': 242976,
          'browser_seconds': 10800, 'local_seconds': 14400, 'calendar_seconds': 86400,
          'retained_bytes': 34359738368, 'minimum_free_bytes': 8589934592,
          'editor_proposals': 1, 'imports': 1, 'builds': 1, 'conversions': 1,
          'packages': 1, 'automatic_retries': 0}
METRICS = ('scored_output_tokens', 'response_ms_sum', 'load_ms', 'session_ms',
           'mean_response_ms', 'full_response_tokens_per_second', 'response_ms_per_output_token',
           'gpu_buffer_descriptor_bytes_sampled_max', 'wasm_address_space_bytes_sampled_max',
           'native_peak_footprint_gb', 'native_idle_footprint_gb', 'native_recovery_footprint_gb',
           'cpu_capacity_percent', 'gpu_time_ms_per_second', 'host_power_watts',
           'cpu_coverage', 'gpu_coverage', 'power_coverage')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_fresh_comparison(root: Path, data: dict, equal) -> None:
    path = root / 'reports/fresh-baseline-r4-v1/result.json'
    if not path.exists():
        equal(data.get('fresh_comparison'), None, 'no unpublished fresh measurements')
        return
    report = json.loads(path.read_text())
    equal(data.get('fresh_comparison'), report, 'fresh canonical report')
    equal(json.loads((root / 'site/fresh-comparison.json').read_text()), report,
          'fresh standalone download')
    equal(report['schema_version'], 1, 'fresh schema')
    equal(report['kind'], 'fresh_baseline_r4_result', 'fresh report kind')
    equal(report['id'], 'fresh-baseline-r4-v1', 'fresh study identity')
    verify_maintenance(root, data, report, equal)
    equal(report['limits'], LIMITS, 'fresh finite limits')
    equal(bool(re.fullmatch('[0-9a-f]{64}', report['study_sha256'])), True, 'fresh study digest')
    if report['candidate_binding_sha256'] is not None:
        equal(bool(re.fullmatch('[0-9a-f]{64}', report['candidate_binding_sha256'])), True, 'fresh candidate binding digest')

    def number(actual, expected, label):
        if expected is None:
            equal(actual, None, label)
        else:
            equal(finite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), True, label)

    source = report['source_identity']
    equal(bool(re.fullmatch('[0-9a-f]{40}', source['base_commit'])), True, 'fresh base commit')
    equal(bool(source['files']), True, 'fresh frozen source files present')
    for name, digest in source['files'].items():
        relative = Path(name)
        equal(not relative.is_absolute() and '..' not in relative.parts, True, f'fresh safe source {name}')
        equal(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest, f'fresh source {name}')

    artifacts = report['artifacts']
    artifact_by_key = {row['key']: row for row in artifacts}
    expected_keys = KEYS if report['candidate_binding_sha256'] else KEYS[:-1]
    equal([row['key'] for row in artifacts], expected_keys, 'fresh artifact order')
    prior = {row['key']: row for row in data['resource_study']['artifacts']}
    for key in KEYS[:-1]:
        equal(artifact_by_key[key]['artifact_sha256'], prior[key]['artifact_sha256'], f'{key} prior identity preserved')
        equal(artifact_by_key[key]['source_sha256'], prior[key]['source_sha256'], f'{key} prior source preserved')
    libraries = {row['model_library_sha256'] for row in artifacts}
    equal(len(libraries), 1, 'fresh shared compiled library')
    equal(libraries, {data['resource_study']['artifacts'][0]['model_library_sha256']}, 'fresh historical compiled library')
    for row in artifacts:
        key = row['key']
        operations = row['weight_operations']
        targeted = 0 if key == 'pristine' else 122880
        equal(row['targeted_source_coordinates'], targeted, f'{key} targeted source extent')
        equal(row['observed_changed_source_entries'], row['changed_source_entries'], f'{key} observed source count')
        equal(row['source_edit']['changed_stored_entries'], row['changed_source_entries'], f'{key} source audit count')
        equal(row['source_edit']['targeted_coordinates'], targeted, f'{key} source audit targeted extent')
        if key == 'astra-r4':
            equal(type(row['changed_source_entries']) is int and 0 < row['changed_source_entries'] <= targeted, True, 'R4 observed stored changes within target')
            equal(row['source_edit']['complete'], True, 'R4 complete saved source audit')
            equal(row['source_edit']['artifact_verification'], 'passed', 'R4 saved artifact verification')
            equal(report['editor_provenance']['source_edit'], row['source_edit'], 'R4 source provenance agrees')
            equal(row['source_edit']['result_sha256'], row['source_sha256'], 'R4 source identity')
            equal(row['source_tensor_contract'], {f'model.language_model.layers.{layer}.mlp.down_proj.weight': {'shape': [1024, 3584], 'dtype': 'BF16'} for layer in (15, 19)}, 'R4 saved tensor contract')
        else:
            equal(row['changed_source_entries'], targeted, f'{key} historical source edit extent')
        if key == 'pristine':
            equal(operations, [], 'fresh pristine has no intervention')
            continue
        equal(len(operations), 2, f'{key} operation count')
        equal({op['tensor'] for op in operations},
              {f'model.language_model.layers.{layer}.mlp.down_proj.weight' for layer in (15, 19)},
              f'{key} tensor family')
        for op in operations:
            columns = op['columns']
            equal(op['op'], 'scale_columns', f'{key} operation type')
            equal(len(columns), 60, f'{key} edited width')
            equal(columns, list(range(columns[0], columns[0] + 60)), f'{key} contiguous columns')
            equal(0 <= columns[0] <= 3524, True, f'{key} column bounds')
            equal(op['factor'] in (.5, .75, 1.25, 1.5), True, f'{key} multiplier family')
        if key != 'astra-r4':
            equal(operations, prior[key]['weight_operations'], f'{key} exact old transformation')
    if 'astra-r4' in artifact_by_key:
        equal(len({row['artifact_sha256'] for row in artifacts}), 6, 'six distinct deployed artifacts')
        editor = report['editor_provenance']
        for name, expected in [('requested_model', 'gpt-6-astra'), ('actual_collaboration_agent', True),
                               ('proposals', 1), ('target_generations', 0), ('new_proposal_from_pristine', True)]:
            equal(editor[name], expected, 'R4 recorded editor ' + name)
        equal(editor['weight_operations'], artifact_by_key['astra-r4']['weight_operations'], 'R4 recorded intervention')
        equal(0 < len(editor['pre_evaluation_rationale'].split()) <= 150, True, 'bounded pre-evaluation prediction')
        for name in ('request_sha256', 'packet_sha256', 'proposal_sha256', 'source_audit_sha256', 'pre_evaluation_rationale_sha256'):
            equal(bool(re.fullmatch('[0-9a-f]{64}', editor[name])), True, 'R4 provenance digest ' + name)

    quality = report['quality']
    resource = report['resource']
    equal(quality['order'], KEYS, 'fresh quality order')
    equal(resource['session_order'], RESOURCE_ORDER, 'fresh mirrored resource order')
    equal(quality['untouched_final_prompts'], 0, 'fresh exposure boundary')
    expected_quality = {'temperature': 0, 'topP': 1, 'seed': 42, 'thinking': False,
                        'contextWindow': 4096, 'maxTokens': 1024, 'requestLimitMs': 120000,
                        'warmups': 1, 'warmupTokens': 16}
    equal(quality['settings'], expected_quality, 'fresh frozen quality generation settings')
    equal(resource['settings'], {**expected_quality, 'maxTokens': 64, 'requestLimitMs': 60000},
          'fresh frozen resource generation settings')
    items = json.loads((root / 'reports/ifeval-browser-v1-final/item-ids.json').read_text())['development']
    expected_ids = [item['item_id'] for item in items['items']]
    rows = quality['artifacts']
    equal([row['artifact_key'] for row in rows], KEYS, 'all fresh quality slots represented')
    quality_tokens = 0
    for row in rows:
        key = row['artifact_key']
        metrics = row.get('metrics')
        if metrics is None:
            equal(row['status'] != 'completed', True, f'{key} missing quality is not complete')
            continue
        outcomes = row['item_outcomes']
        equal([item['item_id'] for item in outcomes], expected_ids, f'{key} exact ordered development set')
        equal(metrics['denominator'], 32, f'{key} quality denominator')
        equal(metrics['instruction_denominator'], items['instruction_count'], f'{key} instruction denominator')
        for mode in ('strict', 'loose'):
            equal(metrics[mode + '_passes'], sum(item[mode + '_pass'] == 1 for item in outcomes), f'{key} {mode} count')
        tokens = sum(item['output_tokens'] or 0 for item in outcomes)
        equal(metrics['scored_output_tokens'], tokens, f'{key} quality tokens')
        quality_tokens += tokens
        number(metrics['generation_seconds'], sum(item['duration_seconds'] or 0 for item in outcomes), f'{key} quality duration')
        equal(metrics['natural_completions'], sum(item['status'] == 'ok' for item in outcomes), f'{key} natural completions')
        equal(metrics['truncations'], sum(item['status'] == 'truncated' for item in outcomes), f'{key} truncations retained')
        for item in outcomes:
            equal(item['output_tokens'] is None or (type(item['output_tokens']) is int and 0 <= item['output_tokens'] <= 1020), True, f'{key} bounded quality output')
        if row['status'] == 'completed':
            equal(metrics['attempted'], 32, f'{key} complete quality attempts')
            equal(metrics['unexecuted'], 0, f'{key} no unexecuted quality items')
            equal(metrics['unresolved_grader_errors'], 0, f'{key} resolved quality grading')
            equal(metrics['technically_valid'], True, f'{key} valid quality execution')
            equal(metrics['semantic_grading_complete'], True, f'{key} complete semantic grading')
            equal(metrics['deployment_limits_satisfied'], True, f'{key} deployment limits')
            equal(metrics['memory_limits_satisfied'], True, f'{key} memory limits')

    sessions = resource['sessions']
    equal([row['artifact_key'] for row in sessions], RESOURCE_ORDER, 'all resource positions represented')
    equal([row['position'] for row in sessions], list(range(1, 13)), 'fresh resource positions')
    resource_tokens = resource_warmup_tokens = 0
    for row in sessions:
        summary = row.get('summary')
        if summary is None:
            equal(row['status'] != 'completed', True, 'missing resource summary is not complete')
            continue
        requests = row['requests']
        measured = [request for request in requests if not request['warmup']]
        warmups = [request for request in requests if request['warmup']]
        label = f"fresh resource {row['position']}"
        equal([request['request_index'] for request in measured], list(range(1, len(measured) + 1)), label + ' request order')
        equal(summary['measured_requests'], len(measured), label + ' request count')
        equal(summary['warmups'], len(warmups), label + ' warm-up count')
        tokens = sum(request['output_tokens'] for request in measured)
        warmup_tokens = sum(request['output_tokens'] for request in warmups)
        resource_tokens += tokens
        resource_warmup_tokens += warmup_tokens
        equal(summary['scored_output_tokens'], tokens, label + ' output tokens')
        equal(summary['warmup_output_tokens'], warmup_tokens, label + ' warm-up tokens')
        duration = sum(request['response_ms'] for request in measured)
        number(summary['response_ms_sum'], duration, label + ' response duration')
        number(summary['mean_response_ms'], duration / len(measured) if measured else None, label + ' mean response')
        number(summary['full_response_tokens_per_second'], tokens / (duration / 1000) if duration else None, label + ' throughput')
        number(summary['response_ms_per_output_token'], duration / tokens if tokens else None, label + ' ms/token')
        equal(summary['natural_completions'], sum(request['finish_reason'] == 'stop' for request in measured), label + ' natural completions')
        equal(summary['truncations'], sum(request['finish_reason'] == 'length' for request in measured), label + ' truncations')
        if row['status'] == 'completed':
            equal(summary['status'], 'complete', label + ' complete summary')
            equal(len(measured), 60, label + ' complete requests')
            equal(len(warmups), 1, label + ' one warm-up')
            equal(all(request['valid'] is True for request in requests), True, label + ' valid observations')
        for request in requests:
            equal(type(request['output_tokens']) is int and 0 <= request['output_tokens'] <= (12 if request['warmup'] else 60), True, label + ' bounded request tokens')
            equal(finite(request['response_ms']) and request['response_ms'] >= 0, True, label + ' finite response')

    def environment_matches(row):
        expected = report['environment']
        observed = row.get('observed_environment') or {}
        return bool(row['status'] == 'completed' and row['expected_environment'] == expected
                    and observed.get('platform') and observed.get('userAgent')
                    and 'Chrome/' + expected['browser_version'].split('.')[0] + '.' in observed['userAgent']
                    and str(observed.get('hardwareConcurrency')) == str(expected['hardware']['hw.logicalcpu'])
                    and ('chrome_version' not in observed or observed['chrome_version'] == expected['browser_version']))

    quality_match = all(environment_matches(row) for row in rows) and all(row['observed_environment'] == rows[0]['observed_environment'] for row in rows)
    equal(quality['matched_environment'], quality_match, 'fresh quality observed environment matching')
    resource_match = all(environment_matches(row) for row in sessions) and all(row['observed_environment'] == sessions[0]['observed_environment'] for row in sessions)
    resource_match = resource_match and all((row.get('summary') or {}).get('runtime') == report['environment']['runtime']
                                          and (row.get('summary') or {}).get('quantization') == report['environment']['quantization']
                                          and (row.get('summary') or {}).get('model_library_sha256') in libraries for row in sessions)
    equal(resource['matched_environment'], resource_match, 'fresh resource observed runtime matching')

    for key in KEYS:
        selected = [row for row in sessions if row['artifact_key'] == key]
        aggregate = resource['aggregates'][key]
        equal(aggregate['n_scheduled'], 2, f'{key} two resource replicates')
        equal(aggregate['n_completed'], sum(row['status'] == 'completed' for row in selected), f'{key} completed replicates')
        equal(set(aggregate['metrics']), set(METRICS), f'{key} complete metric set')
        for metric in METRICS:
            values = [(row.get('summary') or {}).get(metric) if row['status'] == 'completed' and (row.get('summary') or {}).get('status') == 'complete' else None for row in selected]
            available = len(values) == 2 and all(finite(value) for value in values)
            target = aggregate['metrics'][metric]
            equal(target['values'], values, f'{key} {metric} session values')
            for name, expected in [('mean', sum(values) / 2 if available else None), ('min', min(values) if available else None), ('max', max(values) if available else None)]:
                number(target[name], expected, f'{key} {metric} {name}')
            if key == 'pristine':
                continue
            comparison = resource['comparisons'][key]['metrics'][metric]
            base = resource['aggregates']['pristine']['metrics'][metric]['mean']
            candidate = target['mean']
            comparable = resource['matched_environment'] is True and finite(base) and finite(candidate)
            number(comparison['baseline_mean'], base, f'{key} {metric} baseline')
            number(comparison['candidate_mean'], candidate, f'{key} {metric} candidate')
            number(comparison['absolute_delta'], candidate - base if comparable else None, f'{key} {metric} difference')
            number(comparison['relative_percent'], 100 * (candidate - base) / base if comparable and base != 0 else None, f'{key} {metric} relative difference')

    if quality.get('objectives') is not None:
        equal(quality['matched_environment'], True, 'fresh objectives require matched quality')
        base = rows[0]['metrics']
        new = rows[-1]['metrics']
        best = max(row['metrics']['strict_passes'] for row in rows[:-1])
        expected = {'fresh_baseline_strict_passes': base['strict_passes'], 'best_prior_fresh_strict_passes': best,
                    'candidate_strict_passes': new['strict_passes'], 'quality_gain_over_baseline': new['strict_passes'] > base['strict_passes'],
                    'quality_gain_over_best_prior': new['strict_passes'] > best,
                    'output_tokens_no_greater_than_baseline': new['scored_output_tokens'] <= base['scored_output_tokens'],
                    'strict_and_token_target_met': new['strict_passes'] > best and new['scored_output_tokens'] <= base['scored_output_tokens']}
        for key, value in expected.items():
            equal(quality['objectives'][key], value, 'fresh objective ' + key)

    usage = report['actual_usage']
    observed_rows = []
    for row in rows + sessions:
        if not row.get('observed'):
            equal(row['status'] != 'completed', True, 'completed fresh session has observations')
            continue
        if 'generation_budget' not in row or 'warmup' not in row:
            equal(row['status'] != 'completed', True, 'unresolved observation is not complete')
            equal(all(value is None for value in row['observed'].values()), True,
                  'unresolved observation counters stay unavailable')
            continue
        observed_rows.append(row)
    equal(usage['unresolved_attempted_sessions'], usage['attempted_sessions'] - len(observed_rows),
          'fresh unresolved attempts retain their reservations')
    equal(usage['observation_scope'],
          'Known observed supervisor counters; unresolved attempts remain reserved and are not asserted to have zero dispatches.',
          'fresh known-observation accounting scope')
    known_tokens = 0
    for row in observed_rows:
        label = f"fresh {row['artifact_key']} position {row['position']}"
        observed = row['observed']
        warmup = row['warmup']
        equal(warmup['observed_count'], len(warmup['finish_reasons']), label + ' warm-up observations')
        equal(type(warmup['output_tokens']) is int and 0 <= warmup['output_tokens'] <= 12 * warmup['observed_count'], True, label + ' warm-up token ceiling')
        budget = row['generation_budget']
        total = budget['observed']['generated_tokens']
        equal(total is None or (type(total) is int and total >= 0), True, label + ' sampled-token observation type')
        if type(total) is int:
            known_tokens += total
        if row.get('summary') is not None:
            equal(warmup['output_tokens'], row['summary']['warmup_output_tokens'], label + ' resource warm-up tokens')
            equal(warmup['observed_count'], row['summary']['warmups'], label + ' resource warm-up count')
        if row['status'] == 'completed':
            measured_count = 32 if 'metrics' in row else 60
            for key, expected in [('engine_load_attempts', 1), ('warmup_dispatches', 1), ('scored_dispatches', measured_count), ('generation_dispatches', measured_count + 1)]:
                equal(observed[key], expected, label + ' ' + key)
            equal(warmup['observed_count'], 1, label + ' one recorded warm-up')
            scored_tokens = row['metrics']['scored_output_tokens'] if 'metrics' in row else row['summary']['scored_output_tokens']
            equal(total, scored_tokens + warmup['output_tokens'], label + ' complete sampled tokens')
            for key, expected in [('requested_sequences', measured_count + 1), ('completed_sequences', measured_count + 1), ('failed_sequences', 0), ('retries', 0), ('pending_requests', 0), ('limit_violations', 0)]:
                equal(budget['charged'][key], expected, label + ' generation accounting ' + key)
    for key in ('engine_load_attempts', 'warmup_dispatches', 'scored_dispatches', 'generation_dispatches'):
        equal(usage[key], sum(row['observed'][key] or 0 for row in observed_rows), 'fresh reconciled ' + key)
    all_token_totals_known = all(type(row['generation_budget']['observed']['generated_tokens']) is int for row in observed_rows)
    equal(usage['sampled_tokens_including_warmups'], known_tokens if all_token_totals_known else None, 'fresh reconciled sampled tokens')
    for key, selected in [('completed_quality_sessions', rows), ('completed_resource_sessions', sessions), ('completed_sessions', rows + sessions)]:
        equal(usage[key], sum(row['status'] == 'completed' for row in selected), 'fresh reconciled ' + key)
    for key, charged_key in [('generation_retries', 'retries'), ('failed_target_sequences', 'failed_sequences')]:
        equal(usage[key], sum(row['generation_budget']['charged'][charged_key] for row in observed_rows), 'fresh reconciled ' + key)
    for key, limit in [('engine_load_attempts', 'engine_loads'), ('warmup_dispatches', 'warmups'), ('generation_dispatches', 'dispatches'), ('reserved_target_tokens', 'reserved_target_tokens'), ('charged_browser_seconds', 'browser_seconds'), ('charged_local_seconds', 'local_seconds')]:
        equal(finite(usage[key]) and usage[key] >= 0, True, 'fresh finite accounting ' + key)
        if finite(usage[key]) and usage[key] > LIMITS[limit]:
            equal(report['status'], 'stopped', 'fresh exceeded budget remains stopped: ' + key)
    if report['status'] == 'complete':
        for key, expected in [('engine_load_attempts', 18), ('warmup_dispatches', 18), ('scored_dispatches', 912), ('generation_dispatches', 930), ('reserved_target_tokens', 242976), ('completed_sessions', 18), ('completed_quality_sessions', 6), ('completed_resource_sessions', 12), ('attempted_sessions', 18), ('unresolved_attempted_sessions', 0), ('generation_retries', 0), ('failed_target_sequences', 0), ('editor_proposals', 1), ('imports', 1), ('builds', 1), ('conversions', 1), ('packages', 1)]:
            equal(usage[key], expected, 'fresh complete usage ' + key)
        equal(all(row['status'] == 'completed' for row in rows + sessions), True, 'fresh complete schedule')
        equal(usage['sampled_tokens_including_warmups'], quality_tokens + resource_tokens + resource_warmup_tokens + sum(row['warmup']['output_tokens'] for row in rows), 'fresh exact tokens including all warm-ups')
        for row in rows + sessions:
            cleanup = row['cleanup']
            equal(cleanup.get('retained_owned_pids_alive'), [], 'fresh no retained owned processes')
            for key in ('unload_confirmed', 'worker_terminated', 'chrome_exit_confirmed', 'local_harness_exit_confirmed', 'controller_exit_verified', 'chrome_os_exit_verified'):
                equal(cleanup.get(key), True, 'fresh cleanup ' + key)


def verify_maintenance(root: Path, data: dict, report: dict, equal) -> None:
    path = root / 'reports/fresh-baseline-r4-v1/maintenance.json'
    if not path.exists():
        equal(data.get('fresh_maintenance'), None, 'no unpublished maintenance observations')
        return
    maintenance = json.loads(path.read_text())
    equal(data.get('fresh_maintenance'), maintenance, 'fresh maintenance canonical sidecar')
    equal(json.loads((root / 'site/fresh-maintenance.json').read_text()), maintenance, 'fresh maintenance download')
    for key, expected in [('schema_version', 1), ('kind', 'fresh_profile_cache_maintenance'),
                          ('study_sha256', report['study_sha256']),
                          ('introduced_after_quality_sessions', 5), ('limits_unchanged', True)]:
        equal(maintenance[key], expected, 'fresh maintenance ' + key)
    sources = maintenance['source_files']
    equal(set(sources), {'scripts/maintain_fresh_profiles.py', 'docs/fresh-profile-maintenance-addendum.md'},
          'separately bound maintenance source')
    for name, digest in sources.items():
        equal(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, 'maintenance source ' + name)
    expected_sessions = {}
    for kind, records in [('quality', report['quality']['artifacts']), ('resource', report['resource']['sessions'])]:
        for row in records:
            expected_sessions[f"{kind}-{row['position']:02d}-{row['artifact_key']}"] = row
    operations = maintenance['operations']
    equal(len({operation['session_key'] for operation in operations}), len(operations), 'one maintenance receipt per session')
    charged = maintenance['admission_charged_seconds']
    equal(finite(charged) and charged >= 0, True, 'maintenance admission charged time')
    publication_seconds = maintenance['publication_charged_seconds']
    equal(finite(publication_seconds) and publication_seconds >= 0, True, 'maintenance publication charged time')
    charged += publication_seconds
    for operation in operations:
        key = operation['session_key']
        equal(key in expected_sessions, True, 'maintenance scheduled session')
        row = expected_sessions[key]
        equal(row['status'], 'completed', 'maintenance only after completed measurement')
        equal(operation['status'] in ('completed', 'failed'), True, 'maintenance outcome retained')
        for name in ('removed_file_count', 'removed_logical_bytes'):
            equal(type(operation[name]) is int and operation[name] >= 0, True, 'maintenance nonnegative ' + name)
        seconds = operation['charged_seconds']
        equal(finite(seconds) and seconds >= 0, True, 'maintenance charged duration')
        charged += seconds
        for name, digest in operation['preserved_evidence_sha256'].items():
            relative = Path(name)
            equal(not relative.is_absolute() and '..' not in relative.parts, True, 'safe preserved evidence path')
            equal(bool(re.fullmatch('[0-9a-f]{64}', digest)), True, 'preserved evidence digest')
        if operation['status'] == 'completed':
            equal(operation['failure_type'], None, 'completed maintenance has no failure')
            equal(bool(re.fullmatch('[0-9a-f]{64}', operation['preserved_benchmark_link_sha256'])), True,
                  'maintenance preserves frozen benchmark link identity')
            arm = 'baseline' if row['artifact_key'] == 'pristine' else 'candidate'
            evidence_path = f'{arm}/graded.json' if key.startswith('quality-') else f'execution/{arm}-1/supervisor-result.json'
            public_digest = row['graded_evidence_sha256'] if key.startswith('quality-') else row['evidence_sha256']
            equal(operation['preserved_evidence_sha256'][evidence_path], public_digest, 'maintenance preserves published raw evidence identity')
    equal(charged <= report['actual_usage']['charged_local_seconds'] + 1e-6, True, 'maintenance charges included in total local allowance')
