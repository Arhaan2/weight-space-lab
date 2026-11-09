#!/usr/bin/env python3
"""Export the stopped candidate stage without mutating historical evidence.

JSON reads only: no execution modules, graders, inference, or ledger writes.
New report files are exclusive; the independent verifier may share the folder.
Arhaan2; exploratory development workflow, editor isolation unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

METRICS = (
    ('prompt_level_strict_accuracy', 'Strict prompt (primary)'),
    ('instruction_level_strict_accuracy', 'Strict instruction'),
    ('prompt_level_loose_accuracy', 'Loose prompt'),
    ('instruction_level_loose_accuracy', 'Loose instruction'),
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fields(value, names):
    return {name: value[name] for name in names if name in value}


def privacy_check(value, private_ids, private_prompts):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if re.search(r'/Users/|/private/|/tmp/|file://|https?://(?:127\.|localhost)|ai2-train:', encoded):
        raise ValueError('private location or item identifier in public output')
    if any(identity in encoded for identity in private_ids):
        raise ValueError('private identity in public output')
    if any(prompt and json.dumps(prompt, ensure_ascii=False)[1:-1] in encoded for prompt in private_prompts):
        raise ValueError('private prompt in public output')
    forbidden = {'raw_output', 'browser_observation', 'request_id', 'request_sha256', 'operations',
                 'authorization', 'task_text', 'prompt', 'warmupPrompt', 'model_inventory', 'diagnostics'}
    def walk(node):
        if isinstance(node, dict):
            if forbidden.intersection(node):
                raise ValueError('private evidence field in public output')
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)


def report(root, public):
    root, public = Path(root).resolve(), Path(public)
    if any((public / name).exists() for name in ('README.md', 'result.json')):
        raise FileExistsError('candidate report files must be new')
    failure = read(root / 'candidate-stage-failure.json')
    cleanup = read(root / 'candidate-cleanup.json')
    imported = read(root / 'handoff/import-result.json')
    source = read(root / 'candidate/source/audit.json')
    proposal = read(root / 'candidate/source/proposal.json')
    baseline = read(root.parent.parent / 'reports/ifeval-browser-v1/result.json')
    graded = read(root / 'baseline/graded.json')
    config = read(root / 'baseline-config.json')
    plan = read(root / 'candidate-comparison-plan.json')
    request = read(root / 'handoff/request.json')
    verified = read(public / 'verification-final.json')
    assert verified['passed'] is True
    assert imported['accepted'] is True and failure['status'] == 'stopped'
    assert failure['stage'] == 'candidate_conversion_package_precheck'
    assert source['artifact_verification'] == 'passed' and source['complete'] is True
    assert source['base_sha256'] == imported['audit']['original_sha256'] == baseline['checkpoint']['sha256']
    assert source['result_sha256'] == imported['audit']['result_sha256']
    for key in ('targeted_coordinates', 'changed_stored_entries', 'changed_numerical_entries'):
        assert source[key] == imported['audit'][key]
    for key in ('candidate_load_attempts', 'candidate_generation_dispatches', 'candidate_generated_tokens',
                'candidate_scored_dispatches', 'candidate_warmup_dispatches'):
        assert failure[key] == 0
    assert not failure['conversion_subprocess_started'] and not failure['packaging_started']
    assert all(not (root / path).exists() for path in failure['absent_stage_artifacts'])
    assert graded['summary'] == baseline['summary']
    assert sha(read(root / 'manifest.json')) == graded['manifest_sha256']
    assert cleanup['no_matching_stage_processes'] is True
    for relative, digest in failure['evidence_file_sha256'].items():
        assert file_sha(root / relative) == digest, 'failure evidence changed'
    source_public = fields(source, ('artifact_verification', 'complete', 'base_sha256', 'result_sha256',
        'total_stored_coordinates', 'targeted_coordinates', 'changed_stored_entries', 'changed_numerical_entries',
        'bit_only_changed_entries', 'changed_fraction', 'l2_norm', 'max_abs_change', 'unchanged_tensor_count',
        'budget_basis', 'precision'))
    source_public['target_support'] = [{'tensor': op['tensor'], 'columns': op['columns'], 'scale_factor': op['factor']} for op in proposal['operations']]
    source_public['per_tensor'] = {name: fields(row, ('dtype', 'shape', 'base_sha256', 'result_sha256',
        'changed_stored_entries', 'changed_numerical_entries', 'targeted_coordinates', 'l2_norm', 'max_abs_change'))
        for name, row in source['per_tensor'].items()}
    order = [row['item_id'] for row in config['items']]
    by_grade = {row['item_id']: row for row in graded['grades']}
    by_record = {row['item_id']: row for row in graded['records']}
    per_item = []
    for position, item in enumerate(order, 1):
        metrics = by_grade[item]['metrics']
        per_item.append({'schedule_position': position,
            'baseline_strict_compliant': metrics['prompt_level_strict_accuracy'],
            'baseline_loose_compliant': metrics['prompt_level_loose_accuracy'],
            'baseline_completion_status': by_record[item]['status'],
            'candidate_strict_compliant': None, 'candidate_loose_compliant': None,
            'candidate_completion_status': 'unexecuted', 'paired_change': 'unmeasured'})
    result = {
        'schema_version': 1, 'kind': 'ifeval_development_candidate_stopped_report', 'author': 'Arhaan2',
        'scope': 'exploratory on-device workflow validation; editor isolation unverified',
        'status': 'stopped_before_conversion_subprocess', 'scientific_result': False,
        'failure': fields(failure, ('stage', 'failure', 'cause', 'command_exit_code', 'command_file_invocations',
            'conversion_stage_attempts', 'conversion_subprocess_started', 'packaging_started', 'expected_packages',
            'configured_environment_packages_match', 'existing_conversion_subprocess_limit_seconds',
            'package_precheck_limit_seconds', 'frozen_implementation_changed')),
        'proposal_provenance': {
            'basis': 'User-supplied original exploratory proposal based on the supplied development packet.',
            'declared_session_tools': 'File-reading and JSON-parsing tools were used to ingest the supplied packet.',
            'declared_session_exclusions': 'No external research, additional weight inspections, or target forward passes were used in the proposal session.',
            'observation_limit': 'This is the supplied provenance declaration; editor isolation remains unverified.',
            'causality': 'Selected columns are not a verified causal explanation for repetition.',
            'original_request_preserved': True,
        },
        'import': {'accepted': True, 'execution_limit_seconds': 900,
            'observed': imported['budget']['observed'], 'charged': imported['budget']['charged'],
            'submission_slots_remaining': imported['budget']['remaining']['editor_requests'],
            'frontier_requests': imported['frontier_requests'], 'target_generations': imported['target_generations']},
        'saved_checkpoint_audit': source_public,
        'candidate_checkpoint_sha256': source['result_sha256'], 'candidate_deployed_artifact_sha256': None,
        'deployment_audit': {'status': 'unavailable_conversion_not_started', 'quantized_differences': None,
            'decoded_quantized_differences': None, 'reason': 'No converted or packaged artifact was produced.'},
        'frozen_profile': {'settings': config['settings'], 'grading_policy': plan['grading_policy'],
            'grader': plan['grader'], 'scheduled_prompts': len(order),
            'same_order_tokenizer_template_runtime_warmup_and_extraction_required': True},
        'baseline': fields(baseline, ('checkpoint', 'artifact_sha256', 'configuration_sha256', 'manifest_sha256',
            'summary', 'resources', 'cleanup', 'budget', 'warmup', 'load_ms')),
        'candidate': {'metrics': {key: None for key, _ in METRICS}, 'scored_examples_executed': 0,
            'scored_examples_unexecuted': 32, 'natural_completions': 0, 'truncations': 0,
            'scored_output_tokens': 0, 'scored_generation_seconds': None, 'generation_time_status': 'not_started',
            'load_through_cleanup_seconds': None, 'browser_deadline_started': False,
            'resources': {'status': 'not_observed', 'reason': 'No candidate engine load or browser execution.'}},
        'paired': {'status': 'not_evaluable', 'improved': None, 'regressed': None, 'tied': None,
            'unmeasured_pairs': 32, 'strict_delta': None, 'loose_delta': None, 'confidence_interval': None,
            'per_item': per_item},
        'consumption': fields(failure, ('command_file_invocations', 'conversion_stage_attempts',
            'candidate_load_attempts', 'candidate_warmup_dispatches', 'candidate_scored_dispatches',
            'candidate_generation_dispatches', 'candidate_generated_tokens', 'browser_deadline_started',
            'automatic_retries', 'new_weight_proposals', 'heldout_generations', 'experimental_frontier_requests',
            'baseline_reruns')),
        'candidate_authorized_limits': plan['candidate_budget'],
        'candidate_cleanup': fields(cleanup, ('pipeline_exit_code', 'no_matching_stage_processes',
            'conversion_subprocess_started', 'browser_execution_started', 'browser_cleanup', 'import_worker_completed', 'scope')),
        'selection': {**failure['selection'], 'rule': plan['selection']},
        'heldout_examples_left_unexecuted': 128,
        'verification': {'passed': True, 'checks_passed': sum(verified['checks'].values()),
            'checks_total': len(verified['checks']), 'final_report_file_sha256': file_sha(public / 'verification-final.json'),
            'initial_checker_correction': 'Initial report preserved. The checker expected nested package error text in the outer pipeline log; corrected to verify its recorded diagnostic location. No experiment retry or evidence change.',
            'current_evidence_files_unchanged': verified['measurements']['candidate_preservation']['matched'],
            'available_historical_files_unchanged': verified['measurements']['historical_preservation']['matched'],
            'historical_files_previously_unavailable_and_excluded': verified['measurements']['historical_unavailable_files_not_verified']},
        'attribution': ['Arhaan Aggarwal (Arhaan2)', "Existing BrowserLLM work and Gufran's contribution preserved",
            'Google Research IFEval, Zhou et al., Apache-2.0',
            'Allen Institute for AI RLVR-IFeval; upstream Ai2 ImpACT and Llama 2 notices preserved'],
    }
    private_ids, prompts = [request['request_id'], *order], [row['prompt'] for row in config['items']]
    privacy_check(result, private_ids, prompts)
    markdown = render(result)
    privacy_check(markdown, private_ids, prompts)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    public.mkdir(parents=True, exist_ok=True)
    with (public / 'result.json').open('x') as stream:
        stream.write(encoded)
    with (public / 'README.md').open('x') as stream:
        stream.write(markdown)
    return {'written': True, 'status': result['status'], 'selected_arm': None, 'unmeasured_pairs': 32}


def render(result):
    baseline, audit = result['baseline'], result['saved_checkpoint_audit']
    lines = ['# IFEval browser candidate stage — stopped before conversion', '',
        'Arhaan2. The proposal import and saved-checkpoint audit succeeded. The conversion stage failed during its package metadata check, before the converter subprocess began. No candidate engine load or generation occurred, so the matched development comparison is unavailable and no arm was selected. All 128 held-out examples remain unexecuted.', '',
        'The conversion wrapper resolves the configured virtual-environment Python symlink to the base/system interpreter. That interpreter lacks `apache-tvm-ffi` package metadata. Read-only diagnosis confirmed that the configured virtual environment itself contains the exact frozen package versions. The command file exited with code 1. Execution stopped under the explicit no-retry authorization; no frozen implementation was changed.', '',
        f"The one import slot was consumed successfully in {result['import']['observed']['elapsed_seconds']:.3f} seconds of its 900-second limit. There were no import failures, pending requests, or retries. The source audit passed, with {audit['unchanged_tensor_count']} tensors unchanged.", '',
        f"Pristine checkpoint SHA-256: `{audit['base_sha256']}`.",
        f"Candidate checkpoint SHA-256: `{audit['result_sha256']}`.",
        'Candidate deployed-artifact hash: **unavailable**. No converted or packaged artifact was produced; quantized and decoded quantized differences were not measured.', '',
        '| Source tensor | Targeted coordinates | Stored-bit changes | Numerical changes |', '| --- | ---: | ---: | ---: |']
    for name, row in audit['per_tensor'].items():
        lines.append(f"| `{name}` | {row['targeted_coordinates']:,} | {row['changed_stored_entries']:,} | {row['changed_numerical_entries']:,} |")
    target_description = '; '.join(f"`{row['tensor']}` columns {min(row['columns'])}–{max(row['columns'])} scaled by {row['scale_factor']}" for row in audit['target_support'])
    lines += [f"| Total | {audit['targeted_coordinates']:,} | {audit['changed_stored_entries']:,} | {audit['changed_numerical_entries']:,} |", '',
        'Target support: ' + target_description + '.', '',
        f"Actual stored changes comprise {audit['changed_fraction']:.8%} of all {audit['total_stored_coordinates']:,} stored coordinates. Bit-only changes: {audit['bit_only_changed_entries']}. Maximum absolute numerical change: {audit['max_abs_change']}; L2 change: {audit['l2_norm']:.12f}. These are measured saved BF16 changes, separate from unavailable quantization results. Every targeted coordinate happened to change in this saved checkpoint; that does not establish what would survive conversion.", '',
        'The [original baseline report](../ifeval-browser-v1/README.md) remains preserved and was not rerun. Candidate scores are unavailable, not zero scores.', '',
        '| Compliance metric | Completed baseline | Candidate |', '| --- | ---: | --- |']
    for key, label in METRICS:
        metric = baseline['summary']['metrics'][key]
        lines.append(f"| {label} | {metric['passed']}/{metric['scheduled']} ({metric['value']:.2%}) | Unavailable |")
    summary = baseline['summary']
    peak = baseline['resources']['worker_sampled_maxima']
    lines += ['', 'Per-item improvements, regressions and ties are unmeasured for all 32 pairs. No paired effect, confidence interval, or compliance improvement can be reported.', '',
        f"The existing baseline had {summary['execution_status_counts']['ok']} natural completions and {summary['execution_status_counts']['truncated']} truncations, with {summary['unresolved_grader_errors']} unresolved grader errors. It used {summary['known_generated_tokens']:,} scored output tokens over {summary['generation_seconds']:.3f} seconds. Candidate natural completions, truncations and scored output tokens are all zero because no requests ran; candidate generation time and browser elapsed time were not measured.", '',
        'Actual candidate capacity consumed: one successful import, one conversion-stage attempt that failed in its package precheck, zero conversion subprocesses, zero packaging operations, zero engine-load attempts, zero warmups, zero scored requests, zero total generation dispatches, and zero generated tokens. There were zero automatic retries, new weight proposals, baseline reruns, held-out generations, or experimental frontier requests. The fresh browser generation ledger was never created or spent because authorization binding was not reached.', '',
        'The existing conversion subprocess limit remained 1,800 seconds, with a 30-second package precheck; it was not replaced by the browser limit. The authorized browser bounds remained one load, one warmup of at most 16 tokens, 32 scored requests of at most 1,024 tokens, 33 total dispatches, 32,784 total tokens, 120 seconds per request, and 5,400 seconds from load through cleanup. The browser deadline never started.', '',
        'The frozen profile still requires the same 32 development examples and order, pristine origin, tokenizer, chat template, q4f16_1 quantization, WebLLM runtime, warmup policy, extraction and official grader; thinking disabled; temperature 0; top-p 1; seed 42 reset per request; context 4,096; output cap 1,024. No prompt, decoding, output-cleanup, grading or task-selection adjustment was made.', '',
        f"Candidate resource observations are unavailable because no browser run began. Existing baseline worker GPUBuffer descriptor accounting peaked at {peak['gpuBufferBytes']:,} bytes and WASM address space at {peak['wasmBytes']:,} bytes; JS heap was unavailable. These are not physical VRAM measurements. The JSON retains baseline host-wide ranges and maximum single observed process per role separately: host readings include unrelated workloads, RSS/footprint can include shared memory, and sampled maxima do not establish absolute peaks.", '',
        'Cleanup verification found the command had exited and no process matched the known checkpoint, converter, or experiment-runner command signatures. The import worker completed. Browser unload and browser cleanup are not applicable because no candidate browser started. This check does not inventory unrelated user browsers.', '',
        f"Frozen selection policy: {result['selection']['rule']} Status: **not evaluable**; selected arm: **none**, because candidate deployment, execution and grading did not complete. The existing baseline remains available, with no measured comparison decision.", '',
        'The supplied proposal provenance records file-reading and JSON-parsing tools used to ingest the development packet, with no external research, additional weight inspections, or target forward passes in that proposal session. The session was not tool-free. The selected columns are not a verified causal explanation for repetition. The original request is preserved separately from this provenance statement. Editor isolation remains unverified.', '',
        'The [final independent verifier](verification-final.json) passed all 22 checks, including 578 current evidence pins, 1,387 available historical pins, and independent recounting of actual BF16 changes. The 300 previously unavailable historical files remain outside the preservation claim. The [initial verification](verification.json) is preserved: its sole checker failure expected nested package-error text in the outer pipeline log. The corrected checker verifies the recorded diagnostic location. The final verifier additionally removes redundant private request-envelope identifiers from its source and derives exact checks from preserved evidence. Both prior verification outputs remain preserved; no experiment was rerun.', '',
        'Per-item results use the 1-based frozen schedule position, keeping source identifiers and private prompts out of this public report.', '',
        '| Position | Baseline strict | Baseline loose | Baseline completion | Candidate | Paired change |', '| ---: | --- | --- | --- | --- | --- |']
    label = lambda value: 'pass' if value else 'fail'
    for row in result['paired']['per_item']:
        lines.append(f"| {row['schedule_position']} | {label(row['baseline_strict_compliant'])} | {label(row['baseline_loose_compliant'])} | {row['baseline_completion_status']} | Unexecuted | Unmeasured |")
    lines += ['', "See [detailed measurements and audit identities](result.json). Authorship: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2). Existing BrowserLLM work, Gufran's contribution, upstream licenses and scientific attribution remain preserved. Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google Research IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Private prompts, packets, response, authorization and machine paths are excluded from this report.", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/ifeval-browser-v1')
    parser.add_argument('--public', default='reports/ifeval-browser-v1-candidate')
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.public), indent=2))
