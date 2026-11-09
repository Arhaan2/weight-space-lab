#!/usr/bin/env python3
"""Report the separately authorized IFEval suffix without changing old evidence.

Arhaan2. This command performs only the frozen read-only analysis and writes a
new private merge artifact plus a sanitized public report. It never executes a
model, grades an output, opens a ledger mutably, or changes a frozen source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.report_ifeval_heldout import resources, public_interpretation, DEVELOPMENT
from weight_space_lab.ifeval_continuation_merge import merge_evidence
from weight_space_lab.ifeval_heldout_analysis import ANALYSIS_SPEC
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash


AUTHOR = '[Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2)'
PLAN_SHA256 = 'f366b08d4d9a25d54307af6638b1c5fa66f8ee9b87174e087de18d1ad8f9dad8'
SESSION_NAMES = ('baseline', 'candidate_interrupted', 'candidate_continuation')
SESSION_LABELS = ('Baseline', 'Candidate interrupted', 'Candidate continuation')
PRIVATE_MARKERS = ('/Users/', '/private/', '/opt/homebrew/', '/home/',
                   'authorization-source.txt', 'raw_output', 'task_text')
CLEANUP_KEYS = ('unload_confirmed', 'worker_terminated', 'chrome_exit_confirmed',
    'local_harness_exit_confirmed', 'controller_exit_verified', 'chrome_os_exit_verified',
    'forced_termination', 'owned_browser_and_worker_resources_released')


def _new_json(path, value, *, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    if private:
        path.chmod(0o600)


def _accounting(records, grades, observations):
    attempted = [r for r in records if r['attempts']]
    ambiguous = {o['item_id'] for o in observations
                 if o.get('dispatch_outcome') == 'acknowledged_or_ambiguous'}
    status, grading = Counter(r['status'] for r in attempted), Counter(g['status'] for g in grades)
    unexecuted = sum(not r['attempts'] for r in records)
    no_output = sum(r['raw_output'] is None for r in attempted)
    return {
        'scheduled_prompts': len(records), 'attempted_scored_requests': len(attempted),
        'natural_completions': status['ok'], 'truncations': status['truncated'],
        'unexecuted_slots': unexecuted, 'dispatched_without_observed_output': no_output,
        'missing_outputs_including_unexecuted': unexecuted + no_output,
        'ambiguous_dispatch_outcomes': len(ambiguous),
        'attempted_model_failures': sum(r['status'] in {'error', 'timeout'}
            and r['item_id'] not in ambiguous for r in attempted),
        'execution_status_counts': dict(status), 'graded_outputs': grading['graded'],
        'grader_errors': grading['grader_error'], 'grader_timeouts': grading['grader_timeout'],
        'scored_generated_tokens': sum(r['token_counts']['output'] or 0 for r in attempted),
        'scored_input_tokens': sum(r['token_counts']['input'] or 0 for r in attempted),
        'scored_generation_seconds': sum(r['duration_seconds'] or 0 for r in attempted),
        'scored_requests_with_unknown_tokens': sum(r['token_counts']['output'] is None for r in attempted),
        'scored_requests_with_unknown_duration': sum(r['duration_seconds'] is None for r in attempted),
    }


def _failure(error):
    if not error:
        return None
    categories = {
        'requests must execute sequentially after warmup': 'original_supervisor_sequence_guard_race',
        'deadline': 'deadline_failure', 'host_resource_stop': 'host_resource_stop',
        'resource_state_unknown': 'resource_observation_unavailable',
        'cleanup_unconfirmed': 'cleanup_unconfirmed',
        'runtime': 'browser_runtime_failure', 'driver_did_not_complete': 'driver_incomplete',
        'identity': 'identity_integrity_failure', 'reconciliation': 'evidence_reconciliation_failure',
    }
    return {'category': next((v for k, v in categories.items() if k in error),
                              'execution_or_integrity_failure'),
            'preserved_error_sha256': canonical_hash(error),
            'detail_policy': 'Exact original error retained privately; no automatic retry.'}


def _session(directory, arm, evidence, *, positions):
    target = directory / 'execution' / (arm + '-1')
    run = read_json(target / 'supervisor-result.json')
    manifest = read_json(directory / 'manifest.json')
    grant_path = directory / 'load-grant.json'
    grant = read_json(grant_path) if grant_path.exists() else None
    book = run['generation_budget']
    warmups = [{key: row.get(key) for key in
        ('generated_tokens', 'input_tokens', 'response_latency_ms', 'finish_reason')}
        for row in run.get('warmup_outputs', [])]
    observations = run.get('outputs', [])
    accounting = _accounting(evidence['records'], evidence['grades'], observations)
    known_tokens = book['observed']['generated_tokens'] or 0
    unknown_tokens = run['generation_dispatches'] - book['observed']['token_measurements']
    duration = run.get('first_load_through_cleanup_seconds')
    start = grant['started_epoch_ms'] / 1000 if grant else None
    cleanup = run.get('cleanup', {})
    return {
        'run_id': evidence['records'][0]['run_id'],
        'original_schedule_positions': positions, 'execution_status': run['status'],
        'failure': _failure(run.get('error')),
        'source_checkpoint_sha256': manifest['checkpoint']['sha256'],
        'artifact_sha256': manifest['artifact_sha256'],
        'configuration_sha256': manifest['configuration_sha256'],
        'manifest_sha256': canonical_hash(manifest),
        'supervisor_result_sha256': file_sha256(target / 'supervisor-result.json'),
        'ledger_file_sha256': file_sha256(directory / 'generation-budget.sqlite3'),
        'accounting': accounting,
        'usage': {
            **{key: run[key] for key in ('engine_load_attempts', 'warmup_dispatches',
                                      'scored_dispatches', 'generation_dispatches')},
            'known_generated_tokens_including_warmup': known_tokens,
            'generated_tokens_including_warmup': known_tokens if unknown_tokens == 0 else None,
            'generation_requests_with_unknown_tokens': unknown_tokens,
            'generation_seconds_including_warmup': book['observed']['elapsed_seconds'],
            'load_through_cleanup_seconds': duration,
            'supervisor_seconds': run.get('supervisor_seconds'),
            'reserved_generated_tokens': book['reserved']['generated_tokens'],
            'charged_generated_tokens': book['charged']['generated_tokens'],
            'automatic_retries': book['charged']['retries'],
            'pending_requests': book['charged']['pending_requests'],
            'limit_violations': book['charged']['limit_violations'],
        },
        'ledger': {key: book[key] for key in ('limits', 'reserved', 'observed', 'charged', 'cancelled')},
        'warmups': warmups,
        'clock': {'start_epoch_seconds': start,
                  'cleanup_epoch_seconds_derived': start + duration if start is not None and duration is not None else None,
                  'deadline_epoch_seconds': grant['deadline_epoch_ms'] / 1000 if grant else None},
        'cleanup': {key: cleanup.get(key) for key in CLEANUP_KEYS},
        'remaining_owned_processes': len(cleanup.get('retained_owned_pids_alive', [])),
        'resources': resources(target),
    }


def _aggregate(sessions):
    values = list(sessions.values())
    additive = ('engine_load_attempts', 'warmup_dispatches', 'scored_dispatches',
        'generation_dispatches', 'known_generated_tokens_including_warmup',
        'generation_requests_with_unknown_tokens', 'reserved_generated_tokens',
        'charged_generated_tokens', 'automatic_retries', 'pending_requests', 'limit_violations')
    result = {key: sum(s['usage'][key] for s in values) for key in additive}
    result['generated_tokens_including_warmup'] = (
        result['known_generated_tokens_including_warmup']
        if result['generation_requests_with_unknown_tokens'] == 0 else None)
    for public_key, session_key in (('active_load_through_cleanup_seconds', 'load_through_cleanup_seconds'),
                                   ('generation_seconds_including_warmup', 'generation_seconds_including_warmup'),
                                   ('supervisor_seconds', 'supervisor_seconds')):
        measured = [s['usage'][session_key] for s in values]
        result[public_key] = sum(measured) if all(v is not None for v in measured) else None
    result['execution_failures'] = sum(s['failure'] is not None for s in values)
    result['scored_generated_tokens'] = sum(s['accounting']['scored_generated_tokens'] for s in values)
    result['scored_generation_seconds'] = sum(s['accounting']['scored_generation_seconds'] for s in values)
    result['warmup_generated_tokens'] = sum(w['generated_tokens'] or 0 for s in values for w in s['warmups'])
    result['new_weight_proposals'] = result['development_reruns'] = result['experimental_frontier_requests'] = 0
    return result


def _arm_session_observations(selected):
    """Pool sample counts/maxima only; do not sum process resource counters."""
    measurements = [session['resources'] for session in selected.values()]
    peaks = {}
    for key in ('gpuBufferBytes', 'wasmBytes', 'heapBytes', 'eventLoopDelayMs'):
        values = [m.get('worker_sampled_maxima', {}).get(key) for m in measurements]
        peaks[key] = max((v for v in values if v is not None), default=None)
    return {
        'source_sessions': list(selected),
        'resources': {
            'host_sample_count': sum(m.get('host_sample_count', 0) for m in measurements),
            'worker_sample_count': sum(m.get('worker_sample_count', 0) for m in measurements),
            'worker_sampled_maxima_across_sessions': peaks,
            'meaning': 'Sample counts summed and sampled worker maxima maximized across source sessions; process counters remain separate by session.',
        },
        'cleanup': {
            'all_owned_process_closures_verified': all(session['remaining_owned_processes'] == 0
                and all(session['cleanup'][key] is True for key in
                    ('worker_terminated', 'chrome_exit_confirmed', 'local_harness_exit_confirmed',
                     'controller_exit_verified', 'chrome_os_exit_verified')) for session in selected.values()),
            'all_graceful_unloads_confirmed': all(session['cleanup']['unload_confirmed'] is True
                for session in selected.values()),
            'forced_termination_sessions': sum(session['cleanup']['forced_termination'] is True
                for session in selected.values()),
        },
    }


def _paired_instructions(items, baseline, candidate, policy):
    by_record = [{r['item_id']: r for r in evidence['records']} for evidence in (baseline, candidate)]
    by_grade = [{g['item_id']: g for g in evidence['grades']} for evidence in (baseline, candidate)]
    result = {}
    for mode in ('strict', 'loose'):
        counts = Counter()
        for item in items:
            identifier = item['item_id']
            n = len(item['target']['instruction_id_list'])
            records = [source[identifier] for source in by_record]
            grades = [source[identifier] for source in by_grade]
            if any(r['status'] not in {'ok', 'truncated'} or g['status'] != 'graded'
                   for r, g in zip(records, grades)):
                counts['unresolved'] += n
                continue
            flags = [([False] * n if r['status'] == 'truncated' and policy['truncation_policy'] == 'fail'
                      else g['metrics']['instruction_' + mode]) for r, g in zip(records, grades)]
            if any(len(flag) != n for flag in flags):
                raise ValueError('Instruction outcomes do not match frozen instruction identities')
            for left, right in zip(*flags):
                counts['improved' if right > left else 'regressed' if right < left else 'tied'] += 1
                if left == right:
                    counts['both_passed' if left else 'both_failed'] += 1
        result[mode] = {key: counts[key] for key in
            ('improved', 'regressed', 'tied', 'unresolved', 'both_passed', 'both_failed')}
        if sum(result[mode][k] for k in ('improved', 'regressed', 'tied', 'unresolved')) != 173:
            raise ValueError('Instruction paired outcomes do not cover all 173 instructions')
    return result


def _metrics(analysis):
    result = {}
    for mode in ('strict', 'loose'):
        pair = analysis['prompt_comparisons'][mode]
        comparison = pair['comparison']
        result['prompt_' + mode] = {
            **pair['official_metrics']['prompt_level_' + mode + '_accuracy'],
            'difference_percentage_points': comparison['delta'] * 100,
            'ci95_percentage_points': [comparison['confidence_interval'][k] * 100 for k in ('low', 'high')],
        }
        instruction = analysis['instruction_comparisons'][mode]
        result['instruction_' + mode] = {k: instruction[k] for k in
            ('baseline', 'candidate', 'difference_percentage_points')}
        result['instruction_' + mode]['ci95_percentage_points'] = [
            instruction['confidence_interval'][k] for k in ('low_percentage_points', 'high_percentage_points')]
    return result


def _conclusion(complete, metric):
    if not complete:
        return 'The continued comparison remains incomplete; no completed held-out compliance effect is established.'
    delta, (low, high) = metric['difference_percentage_points'], metric['ci95_percentage_points']
    if delta == 0:
        return 'Strict prompt compliance is unchanged; this exploratory comparison does not show a held-out gain.'
    direction = 'higher' if delta > 0 else 'lower'
    uncertainty = ('The paired 95% interval includes zero, so the direction remains uncertain.'
                   if low <= 0 <= high else 'The paired 95% interval excludes zero for this preselected comparison.')
    return f'The candidate has {direction} observed strict prompt compliance on this subset. {uncertainty}'


def _number(value, decimals=3):
    return 'unavailable' if value is None else f'{value:,.{decimals}f}'


def _markdown(value):
    metrics, sessions = value['metrics'], value['sessions']
    lines = ['# 128-example IFEval on-device subset — continued comparison', '',
        f'Authorship: {AUTHOR}.', '', value['conclusion'], '',
        f"Status: **{value['status']}**. The frozen candidate comparison joins 128 preserved baseline grades, "
        '94 preserved candidate-prefix grades, and only the newly graded 34-position candidate suffix. '
        'The candidate spans two sessions with an extra warm-up; this does not constitute successful completion '
        'of the original uninterrupted protocol.', '',
        '| Compliance | Baseline | Candidate | Difference (pp) | Paired 95% interval (pp) |',
        '| --- | ---: | ---: | ---: | --- |']
    for key, label in (('prompt_strict', 'Strict prompt (primary)'), ('prompt_loose', 'Loose prompt'),
                       ('instruction_strict', 'Strict instruction'), ('instruction_loose', 'Loose instruction')):
        m = metrics[key]
        scores = [f"{m[a]['numerator']}/{m[a]['denominator']} ({100*m[a]['value']:.2f}%)" for a in ('baseline', 'candidate')]
        low, high = m['ci95_percentage_points']
        lines.append(f"| {label} | {' | '.join(scores)} | {m['difference_percentage_points']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    lines += ['', 'Counts are aggregated across the disjoint prefix and suffix; their percentages are never averaged. '
        'All 128 prompts and 173 instructions per arm remain in the denominators. Intervals use the frozen '
        '10,000 paired percentile bootstrap resamples, seed 0, with whole prompts as clusters. Instruction ratios '
        'are recomputed from compliant and scheduled instruction counts in each resample. No exact paired '
        'significance test was added, and significance is not a completion condition.', '',
        '| Paired outcomes | Improvements | Regressions | Ties | Unresolved |',
        '| --- | ---: | ---: | ---: | ---: |']
    for level, key in (('prompt', 'paired_prompt_outcomes'), ('instruction', 'paired_instruction_outcomes')):
        for mode, counts in value[key].items():
            lines.append(f"| {mode.title()} {level} | " + ' | '.join(str(counts[k]) for k in
                ('improved', 'regressed', 'tied', 'unresolved')) + ' |')
    lines += ['', 'Instruction pairs refer to corresponding constraints within each frozen prompt. Their uncertainty '
        'continues to use prompt clusters; instructions are not treated as independent bootstrap units.']
    if not value['complete_paired_evaluation']:
        lines += ['', 'Missing or unresolved slots receive the frozen zero credit in aggregate projections. '
            'These are incomplete-schedule bounds, not measured model failures or an identified completed semantic effect. '
            f"Remaining incomplete suffix positions: {', '.join(map(str, value['incomplete_suffix_schedule_positions'])) or 'none'}."]
    if not value['semantic_grading_complete']:
        lines += ['', 'Unresolved evaluator errors are not demonstrated model failures; the difference between arm bounds is not an identified semantic effect.']
    lines += ['', '| Arm observation | Baseline (128 prompts) | Candidate combined (128 prompts) |',
        '| --- | ---: | ---: |']
    observations = (('natural_completions', 'Natural completions'), ('truncations', 'Truncations'),
        ('unexecuted_slots', 'Unexecuted slots'), ('dispatched_without_observed_output', 'Dispatched without output'),
        ('ambiguous_dispatch_outcomes', 'Ambiguous dispatch outcomes'), ('attempted_model_failures', 'Attempted model failures'),
        ('grader_errors', 'Grader errors'), ('grader_timeouts', 'Grader timeouts'),
        ('scored_generated_tokens', 'Scored sampled tokens'), ('scored_generation_seconds', 'Scored generation seconds'))
    for key, label in observations:
        vals = [value['arms'][a]['accounting'][key] for a in ('baseline', 'candidate')]
        lines.append('| ' + label + ' | ' + ' | '.join(_number(v, 3 if key.endswith('seconds') else 0) for v in vals) + ' |')
    lines += ['', '| Session usage | Baseline | Candidate interrupted | Candidate continuation |',
        '| --- | ---: | ---: | ---: |']
    for key, label in (('engine_load_attempts', 'Loads'), ('warmup_dispatches', 'Warm-ups'),
        ('scored_dispatches', 'Scored dispatches'), ('generation_dispatches', 'All generation dispatches'),
        ('known_generated_tokens_including_warmup', 'Known tokens including warm-up'),
        ('generation_seconds_including_warmup', 'Generation seconds including warm-up'),
        ('load_through_cleanup_seconds', 'Load through cleanup seconds')):
        lines.append('| ' + label + ' | ' + ' | '.join(_number(sessions[name]['usage'][key],
            3 if 'seconds' in key else 0) for name in SESSION_NAMES) + ' |')
    for key, label in (('natural_completions', 'Natural scored completions'), ('truncations', 'Scored truncations'),
        ('scored_generated_tokens', 'Scored sampled tokens'), ('scored_generation_seconds', 'Scored generation seconds')):
        lines.append('| ' + label + ' | ' + ' | '.join(_number(sessions[name]['accounting'][key],
            3 if 'seconds' in key else 0) for name in SESSION_NAMES) + ' |')
    aggregate, timing = value['cumulative_usage'], value['timing']
    lines += ['', f"Cumulative consumption: **{aggregate['engine_load_attempts']} loads, {aggregate['warmup_dispatches']} warm-ups, "
        f"{aggregate['scored_dispatches']} scored requests, {aggregate['generation_dispatches']} total dispatches**, and "
        f"{aggregate['known_generated_tokens_including_warmup']:,} known generated tokens including warm-ups. "
        f"Token counts are unknown for {aggregate['generation_requests_with_unknown_tokens']} requests. "
        f"Reservations: {aggregate['reserved_generated_tokens']:,} tokens; conservative charged usage: "
        f"{aggregate['charged_generated_tokens']:,}. Automatic retries: {aggregate['automatic_retries']}; "
        f"pending requests: {aggregate['pending_requests']}; limit violations: {aggregate['limit_violations']}.", '',
        f"Summed active load-through-cleanup time: {_number(aggregate['active_load_through_cleanup_seconds'])} seconds. "
        f"Candidate interruption gap: {_number(timing['candidate_interruption_gap_seconds'])} seconds. "
        f"First baseline load through final cleanup calendar span: {_number(timing['first_load_to_final_cleanup_calendar_seconds'])} seconds. "
        f"The original aggregate elapsed record remains {_number(timing['preserved_original_aggregate_span_seconds'])} seconds. "
        'The continuation has its own load-through-cleanup clock capped at 5,400 seconds; the closed original '
        '10,800-second clock and ledgers were not reopened or replenished.', '',
        'The continuation allowance was one load, one warm-up capped at 16 tokens, 34 scored requests capped at '
        '1,024 tokens each, at most 35 dispatches and 34,832 generated tokens, and 120 seconds per request. '
        'The expected completed-study total is three loads, three warm-ups and 259 dispatches. Warm-ups are '
        'excluded from compliance scores. Four injected empty-thinking prefix IDs count toward each request '
        'cap but not sampled completion usage, allowing at most 1,020 sampled scored tokens and 12 sampled warm-up tokens.', '',
        value['original_interruption'], '',
        '| Session cleanup | Status | Graceful unload | Worker / browser / harness closed | Remaining processes |',
        '| --- | --- | --- | --- | ---: |']
    for name, label in zip(SESSION_NAMES, SESSION_LABELS):
        s, c = sessions[name], sessions[name]['cleanup']
        process_closed = all(c[k] is True for k in ('worker_terminated', 'chrome_exit_confirmed',
            'local_harness_exit_confirmed', 'controller_exit_verified', 'chrome_os_exit_verified'))
        lines.append(f"| {label} | {s['execution_status']} | {c['unload_confirmed']} | {process_closed} | {s['remaining_owned_processes']} |")
    lines += ['', f"Recorded session-level execution failures: {aggregate['execution_failures']}. "
        'The original supervisor failure remains part of the result even when all new suffix outputs complete. '
        'Per-session failure categories, cleanup confirmations, token accounting and resource observations are preserved in the JSON.', '',
        '| Resource observation (sampled maximum) | Baseline | Candidate interrupted | Candidate continuation |',
        '| --- | ---: | ---: | ---: |']
    for key, label in (('gpuBufferBytes', 'Worker GPUBuffer descriptor bytes'), ('wasmBytes', 'Worker WASM address-space bytes'),
                       ('heapBytes', 'Worker JS heap bytes'), ('eventLoopDelayMs', 'Worker timer delay ms')):
        vals = [sessions[name]['resources'].get('worker_sampled_maxima', {}).get(key) for name in SESSION_NAMES]
        lines.append('| ' + label + ' | ' + ' | '.join(_number(v, 3 if key.endswith('Ms') else 0) for v in vals) + ' |')
    if value['environment_diagnostics']['developer_tool_diagnostic_observed']:
        lines += ['', 'A developer-tool diagnostic was preserved privately. Host telemetry remained available in '
            'the continuation session; the diagnostic is not counted as an execution failure. Execution status '
            'and failures follow the terminal supervisor evidence.']
    lines += ['', value['resource_interpretation'], '',
        'Frozen generation settings remain fresh conversations, thinking disabled, temperature 0, top-p 1, '
        'seed 42 reset per request, context 4,096, output cap 1,024, WebLLM 0.2.86 and q4f16_1. '
        'Ordinary incorrect and truncated responses are retained under the official frozen scoring policy.', '',
        DEVELOPMENT, '',
        'The [original report](../ifeval-browser-v1-heldout/README.md) retains the interrupted result and its '
        'qualitative inspection of changed prefix outcomes. Official compliance grades do not establish substantive answer correctness.']
    if value['suffix_qualitative_interpretation']:
        q = value['suffix_qualitative_interpretation']
        lines += ['', q['summary'], '']
        for row in q['changed_outcomes']:
            lines.append(f"- Schedule item {row['schedule_position']}: strict prompt {row['strict_change']}; "
                f"loose prompt {row['loose_change']}. {' '.join(row['observation'].splitlines())}")
    lines += ['', 'Frozen identities:', '', f"- Authorized continuation plan: `{value['continuation_plan_sha256']}`.",
        f"- Original evaluation freeze: `{value['original_evaluation_freeze_sha256']}`.",
        f"- Continuation manifest: `{value['continuation_manifest_sha256']}`."]
    for arm in ('baseline', 'candidate'):
        for key, label in (('source_checkpoint_sha256', 'source checkpoint'), ('artifact_sha256', 'deployed artifact')):
            lines.append(f"- {arm.title()} {label}: `{value['arms'][arm][key]}`.")
    for name, digest in value['runner_source_identities'].items():
        lines.append(f"- {name.replace('_', ' ').capitalize()}: `{digest}`.")
    lines += ['', 'The [machine-readable report](result.json) retains all four endpoints, frozen uncertainty settings, '
        'session and cumulative accounting, and sanitized identities. The [independent execution verification](verification.json) '
        'audits the continuation and preserved evidence. Private merge references remain with the private evidence. '
        'Public artifacts contain no private prompts, outputs, approval text, editor packets or machine paths.', '',
        value['limitations'], '', 'Proposal provenance (reported, not independently observed): ' + value['proposal_provenance'], '',
        'Upstream attribution remains with [Google Research IFEval (Zhou et al.)]'
        '(https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), '
        '[Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), Qwen, WebLLM/MLC, '
        'existing BrowserLLM work, Gufran’s contribution, and their preserved licenses.']
    return '\n'.join(lines) + '\n'


def report(original_root, continuation_root, public):
    old, tail, public = map(lambda p: Path(p).resolve(), (original_root, continuation_root, public))
    if any((public / name).exists() for name in ('README.md', 'result.json')):
        raise FileExistsError('Combined report must use a fresh destination; preserved reports are never overwritten')
    verified = read_json(public / 'verification.json')
    if verified.get('passed') is not True:
        raise ValueError('An independent successful evidence audit is required before reporting')
    plan, manifest = read_json(tail / 'continuation-plan.json'), read_json(tail / 'manifest.json')
    if canonical_hash(plan) != PLAN_SHA256:
        raise ValueError('The report must bind the exact authorized prepared continuation plan')
    analysis = merge_evidence(old, tail, tail / 'merge-plan.json')
    frozen = read_json(old / 'evaluation-freeze.json')
    items = read_json(old / 'evaluation-items.json')
    baseline = read_json(old / 'baseline/baseline/graded.json')
    prefix = read_json(old / 'candidate/candidate/graded.json')
    suffix = read_json(tail / 'candidate/graded.json')
    candidate = {key: prefix[key][:94] + suffix[key] for key in ('records', 'grades')}
    sessions = {
        'baseline': _session(old / 'baseline', 'baseline', baseline, positions=[1, 128]),
        'candidate_interrupted': _session(old / 'candidate', 'candidate', prefix, positions=[1, 128]),
        'candidate_continuation': _session(tail, 'candidate', suffix, positions=[95, 128]),
    }
    cumulative = _aggregate(sessions)
    candidate_usage = _aggregate({k: v for k, v in sessions.items() if k != 'baseline'})
    clocks = {k: s['clock'] for k, s in sessions.items()}
    first, last = clocks['baseline']['start_epoch_seconds'], clocks['candidate_continuation']['cleanup_epoch_seconds_derived']
    continuation_start, old_end = clocks['candidate_continuation']['start_epoch_seconds'], clocks['candidate_interrupted']['cleanup_epoch_seconds_derived']
    timing = {
        'preserved_original_aggregate_span_seconds': plan['historical_usage']['first_load_through_final_cleanup_seconds'],
        'preserved_original_active_session_seconds': sum(sessions[k]['usage']['load_through_cleanup_seconds']
            for k in ('baseline', 'candidate_interrupted')),
        'candidate_interruption_gap_seconds': continuation_start - old_end if continuation_start is not None and old_end is not None else None,
        'first_load_to_final_cleanup_calendar_seconds': last - first if last is not None and first is not None else None,
        'calendar_method': 'Epoch load timestamps plus preserved monotonic session durations; includes gaps and excludes post-cleanup grading/reporting.',
        'continuation_clock_limit_seconds': 5400, 'original_clock_limit_seconds_preserved': 10800,
        'original_clock_reopened': False,
    }
    observations = []
    for directory in (old / 'candidate', tail):
        observations += read_json(directory / 'execution/candidate-1/supervisor-result.json').get('outputs', [])
    arms = {
        'baseline': {key: sessions['baseline'][key] for key in ('source_checkpoint_sha256', 'artifact_sha256', 'accounting')},
        'candidate': {'source_checkpoint_sha256': manifest['checkpoint']['sha256'],
            'artifact_sha256': manifest['artifact_sha256'],
            'accounting': _accounting(candidate['records'], candidate['grades'], observations)},
    }
    arms['baseline']['usage'] = _aggregate({'baseline': sessions['baseline']})
    arms['candidate']['usage'] = candidate_usage
    arms['baseline'].update(_arm_session_observations({'baseline': sessions['baseline']}))
    arms['candidate'].update(_arm_session_observations({k: v for k, v in sessions.items() if k != 'baseline'}))
    metrics = _metrics(analysis)
    complete = analysis['status'] == 'complete_manually_continued'
    filenames = {'frozen_original_runner': 'weight_space_lab/ifeval_heldout_runner.py',
        'continuation_runner': 'weight_space_lab/ifeval_candidate_continuation.py',
        'continuation_supervisor_repair': 'weight_space_lab/ifeval_supervisor_gate.py',
        'frozen_continuation_merger': 'weight_space_lab/ifeval_continuation_merge.py'}
    source_pins = {}
    for label, filename in filenames.items():
        matches = [(p, digest) for p, digest in manifest['files'].items() if p.endswith('/' + filename)]
        if len(matches) != 1 or file_sha256(matches[0][0]) != matches[0][1]:
            raise ValueError('Frozen runner source identity is unavailable or changed')
        source_pins[label] = matches[0][1]
    interpretation_path = tail / 'qualitative-suffix-review.json'
    interpretation = public_interpretation(read_json(interpretation_path)) if interpretation_path.exists() else None
    if interpretation and any(not 95 <= row['schedule_position'] <= 128 for row in interpretation['changed_outcomes']):
        raise ValueError('New qualitative interpretation must concern only the suffix')
    incomplete = [position for position, record, grade in zip(range(95, 129), suffix['records'], suffix['grades'])
                  if record['status'] not in {'ok', 'truncated'} or grade['status'] != 'graded']
    # Publish no console text. The supervisor remains the execution-failure authority.
    console_path = tail / 'execution-console.txt'
    console = console_path.read_text(errors='replace') if console_path.exists() else ''
    diagnostic = ('xcrun:' in console or
                  ('Error loading required libraries.' in console and 'libxcodebuildLoader.dylib' in console))
    telemetry = sessions['candidate_continuation']['resources'].get('host_sample_count', 0) > 0
    value = {
        'schema_version': 1, 'kind': 'continued_ifeval_subset_combined_report', 'author': 'Arhaan2',
        'label': '128-example IFEval on-device subset', 'status': analysis['status'],
        'complete_paired_evaluation': complete, 'semantic_grading_complete': analysis['semantic_grading_complete'],
        'conclusion': _conclusion(complete, metrics['prompt_strict']),
        'scheduled_prompts_per_arm': 128, 'scheduled_instructions_per_arm': 173,
        'continuation_plan_sha256': canonical_hash(plan), 'continuation_manifest_sha256': canonical_hash(manifest),
        'original_evaluation_freeze_sha256': canonical_hash(frozen),
        'merge_plan_object_sha256': canonical_hash(read_json(tail / 'merge-plan.json')),
        'merge_plan_internal_sha256': analysis['merge_plan_sha256'],
        'private_combined_analysis_sha256': canonical_hash(analysis),
        'runner_source_identities': source_pins,
        'report_generator_sha256': file_sha256(Path(__file__).resolve()),
        'verification_sha256': file_sha256(public / 'verification.json'),
        'original_records_rewritten': analysis['original_records_rewritten'],
        'original_outputs_regraded': analysis['original_outputs_regraded'],
        'new_suffix_outputs_graded': sum(g['status'] == 'graded' for g in suffix['grades']),
        'metrics': metrics, 'paired_prompt_outcomes': analysis['paired_outcomes'],
        'paired_instruction_outcomes': _paired_instructions(items, baseline, candidate, frozen['policy']),
        'analysis_specification': ANALYSIS_SPEC,
        'exact_paired_significance_test': analysis['exact_paired_significance_test'],
        'arms': arms, 'sessions': sessions, 'cumulative_usage': cumulative, 'timing': timing,
        'environment_diagnostics': {'developer_tool_diagnostic_observed': diagnostic,
            'host_telemetry_available_in_continuation_session': telemetry,
            'diagnostic_counted_as_execution_failure': False,
            'failure_authority': 'terminal supervisor evidence'},
        'continuation_limits': plan['limits'], 'cumulative_maximum_after_success': plan['cumulative_maximum_after_success'],
        'incomplete_suffix_schedule_positions': incomplete,
        'unexecuted_suffix_schedule_positions': [p for p, r in zip(range(95, 129), suffix['records']) if not r['attempts']],
        'original_interruption': 'The original candidate session stopped after 94 scored completions when its supervisor '
            'used an event snapshot missing the durable predecessor completion and rejected the gate for position 95. '
            'The dispatch audit found no acknowledgement, reservation or model call for positions 95–128. '
            'The original session required forced termination; process closure was verified, but graceful unload was not confirmed. '
            'Its failure, grades, artifacts, closed ledger and timing records remain unchanged. '
            'The separately versioned continuation runner uses the prepared supervisor repair.',
        'suffix_qualitative_interpretation': interpretation,
        'resource_improvements_demonstrated': False,
        'resource_interpretation': 'A single sequential comparison with an interrupted and resumed candidate does not '
            'demonstrate a controlled speed or energy improvement. GPUBuffer descriptor allocations and WASM address space '
            'are not physical VRAM. CPU, disk and kernel-energy observations are cumulative counters since process creation '
            'and are reported as maximum single observed process values per role, not task-only deltas or sums. Kernel energy '
            'is not battery energy. Host activity, shared allocations and sample spacing limit these observations. '
            'All three sessions retain their own worker and host sample counts and numeric ranges in the JSON.',
        'settings': frozen['settings'], 'quantization': frozen['quantization'], 'policy': frozen['policy'], 'grader': frozen['grader'],
        **{key: frozen[key] for key in ('evaluation_items_sha256', 'original_data_manifest_sha256',
            'original_data_manifest_file_sha256', 'tokenizer_sha256', 'template_sha256', 'runtime_sha256', 'model_library_sha256')},
        'upstream_source_revision': frozen['original_source']['revision'],
        'upstream_source_file_sha256': frozen['original_source']['sha256'],
        'editor_isolation': 'unverified', 'proposal_provenance': frozen['proposal_session_provenance']['statement'],
        'proposal_provenance_independently_observed': False, 'subset_exposed_for_future_edits': True,
        'limitations': 'This remains an exploratory manual-editor study with editor isolation unverified. '
            'It evaluates 128 prompts, not the full 541-example IFEval benchmark, and does not establish representative '
            'full-benchmark performance or absence of pretraining exposure. The bootstrap describes one preselected comparison '
            'and does not include candidate-selection uncertainty. Instruction compliance is distinct from substantive '
            'answer correctness. The selected weight columns are not a verified causal explanation of response behavior. '
            'The subset is exposed and cannot be called untouched for future edits. No new proposal, checkpoint conversion, '
            'weight edit, baseline rerun, previously dispatched scored rerun, or development experiment was performed.',
    }
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    markdown = _markdown(value)
    for text in (encoded, markdown):
        if any(marker in text for marker in PRIVATE_MARKERS):
            raise ValueError('Public report contains forbidden private material')
    private_path = tail / 'combined-analysis.json'
    if private_path.exists():
        if read_json(private_path) != analysis:
            raise ValueError('Existing private combined analysis differs; refusing to rewrite it')
    else:
        _new_json(private_path, analysis, private=True)
    public.mkdir(parents=True, exist_ok=True)
    for name, text in (('result.json', encoded), ('README.md', markdown)):
        with (public / name).open('x') as stream:
            stream.write(text)
    print(json.dumps({'status': value['status'], 'primary': metrics['prompt_strict'],
        'cumulative_usage': cumulative, 'incomplete_suffix_schedule_positions': incomplete}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', default='artifacts/ifeval-browser-v1-heldout')
    parser.add_argument('--continuation-root', default='artifacts/ifeval-browser-v1-heldout-continuation')
    parser.add_argument('--public', default='reports/ifeval-browser-v1-heldout-continuation-combined')
    args = parser.parse_args()
    report(args.original_root, args.continuation_root, args.public)
