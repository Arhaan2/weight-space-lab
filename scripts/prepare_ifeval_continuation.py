#!/usr/bin/env python3
"""Prepare, but never authorize or execute, the exact held-out suffix. Arhaan2."""
from __future__ import annotations

import argparse
import copy
from contextlib import closing
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlparse

from weight_space_lab.ifeval_baseline import POLICY, SETTINGS, save
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash


_BROWSER_OFFLINE_CHECKS = (
    'node --import tsx --test tests/unit/ifeval-study.test.ts tests/unit/task-evaluation.test.ts tests/unit/task-preflight.test.ts',
    'node --check scripts/run-task-evaluation.mjs',
    'node scripts/build-worker.mjs',
    'npm run typecheck',
)
_BROWSER_UNCHANGED_FILES = (
    'src/benchmark/worker/benchmark.worker.ts', 'src/benchmark/task-evaluation/client.ts',
    'src/benchmark/task-evaluation/artifacts.ts', 'src/benchmark/task-evaluation/records.ts',
    'src/benchmark/protocol/settings.ts', 'src/benchmark/protocol/schedule.ts',
    'harness/controller/server.ts', 'scripts/build-worker.mjs', 'scripts/run-task-evaluation.mjs',
    'package-lock.json', 'package.json', 'public/task-evaluation.html',
    'node_modules/@mlc-ai/web-llm-next/lib/index.js',
)
_BROWSER_CHANGED_FILES = ('src/benchmark/task-evaluation/protocol.ts', 'tests/unit/ifeval-study.test.ts')
_BROWSER_BUILD_FILES = (
    'public/task-evaluation.html', 'public/task-evaluation.js',
    'public/benchmark-worker/benchmark.worker.js', 'public/benchmark-worker/chunk-476ZUILE.js',
    'public/benchmark-worker/lib-LBX6YJFZ.js', 'public/benchmark-worker/lib-SQFUEKDP.js',
)


def validate_browser_offline(old, root, repo):
    """Require successful offline checks bound to the exact source/build bytes.

    A recorded checksum is compared with the file it describes before any
    preparation file is written. Recomputing a new runtime pin must not bless
    code that changed after the focused offline checks.
    """
    old, root, repo = map(lambda p: Path(p).resolve(), (old, root, repo))
    record = read_json(root/'browser-continuation-offline.json')
    expected_configuration = {'study':'ifeval-browser-v1', 'evaluationStage':'heldout-continuation',
        'arm':'candidate', 'originalScheduleOffset':94, 'originalScheduleLength':128,
        'scored_items':34, 'warmups':1, 'generation_dispatches':35, 'max_generated_tokens':34832}
    if (record.get('schema_version') != 1
            or record.get('kind') != 'heldout_candidate_continuation_browser_offline_verification'
            or record.get('prepared_only') is not True or record.get('execution_authorized') is not False
            or record.get('required_configuration') != expected_configuration
            or record.get('original_frozen_browser_sources_and_build_verified') is not True
            or record.get('all_cloned_model_files_byte_identical') is not True
            or any(type(record.get(key)) is not int or record[key] != 0 for key in
                   ('engine_loads','generation_dispatches','target_forward_passes','model_conversion_calls'))):
        raise ValueError('Browser offline record does not bind zero-inference candidate-only continuation checks')
    checks = record.get('checks', [])
    by_command = {check.get('command'):check for check in checks if isinstance(check,dict)}
    if len(by_command) != len(checks) or set(by_command) != set(_BROWSER_OFFLINE_CHECKS):
        raise ValueError('Required focused browser offline checks are missing or duplicated')
    unit = by_command[_BROWSER_OFFLINE_CHECKS[0]]
    if (type(unit.get('passed')) is not int or unit['passed'] < 17
            or type(unit.get('failed')) is not int or unit['failed'] != 0
            or any(type(by_command[command].get('exit_code')) is not int
                   or by_command[command]['exit_code'] != 0 for command in _BROWSER_OFFLINE_CHECKS[1:])):
        raise ValueError('Browser offline checks did not all pass')
    unchanged = record.get('unchanged_generation_runtime_and_harness', {})
    changed = record.get('changed_source', {})
    built = record.get('served_build_files', {})
    if (not isinstance(unchanged,dict) or not isinstance(changed,dict) or not isinstance(built,dict)
            or set(unchanged) != set(_BROWSER_UNCHANGED_FILES)
            or set(changed) != set(_BROWSER_CHANGED_FILES) or set(built) != set(_BROWSER_BUILD_FILES)):
        raise ValueError('Browser offline source/build pin inventory is incomplete or unexpected')
    verified = 0
    def check_file(path, expected):
        nonlocal verified
        if not isinstance(expected,str) or file_sha256(path) != expected:
            raise ValueError('Browser offline source/build/patch hash changed: '+str(path))
        verified += 1
    check_file(old/'browser-schedule-offline.json', record.get('original_browser_offline_record_sha256'))
    check_file(repo/'integrations/ifeval-heldout-continuation.patch', record.get('source_patch_sha256'))
    for name, digest in unchanged.items():
        check_file(old/'benchmark'/name, digest)
        check_file(root/'benchmark'/name, digest)
    for name, pins in changed.items():
        if not isinstance(pins,dict) or set(pins) != {'original_sha256','continuation_sha256'}:
            raise ValueError('Browser changed-source pins must bind original and continuation bytes')
        check_file(old/'benchmark'/name, pins['original_sha256'])
        check_file(root/'benchmark'/name, pins['continuation_sha256'])
    for name, digest in built.items():
        check_file(root/'benchmark'/name, digest)
    return {'passed':True,'verified_recorded_file_hashes':verified,
        'focused_browser_tests_passed':unit['passed'],'new_engine_loads':0,'new_generation_dispatches':0}


def readonly_ledger(path):
    path=Path(path).resolve()
    wal=path.with_name(path.name+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Historical ledger has unapplied WAL; do not alter it')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        db.row_factory=sqlite3.Row
        assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        meta=dict(db.execute('SELECT * FROM metadata WHERE singleton=1').fetchone())
        rows=[dict(r) for r in db.execute('SELECT * FROM attempts')]
    return meta,rows


def stopped_prefix(old):
    """Prove exactly which model calls happened; a denied gate is not a call."""
    old=Path(old).resolve()
    config=read_json(old/'candidate/baseline-config.json')
    items=read_json(old/'evaluation-items.json')
    result=read_json(old/'candidate/execution/candidate-1/supervisor-result.json')
    graded=read_json(old/'candidate/candidate/graded.json')
    meta,rows=readonly_ledger(old/'candidate/generation-budget.sqlite3')
    if (len(items)!=128 or len(config['items'])!=128 or len(rows)!=95 or not meta['cancelled']
        or [i['item_id'] for i in config['items']]!=[i['item_id'] for i in items]
        or [r['item_id'] for r in graded['records']]!=[i['item_id'] for i in items]
        or [g['item_id'] for g in graded['grades']]!=[i['item_id'] for i in items]
        or result['status']!='stopped' or result['scored_dispatches']!=94
        or result['generation_dispatches']!=95
        or {r['request_id'] for r in rows}!={f'generation-{i}' for i in range(1,96)}
        or any(r['status']!='completed' or r['retry_of'] is not None for r in rows)
        or [o['item_id'] for o in result['outputs']]!=[i['item_id'] for i in items[:94]]
        or any(r['status'] not in {'ok','truncated'} or len(r['attempts'])!=1 for r in graded['records'][:94])
        or any(g['status']!='graded' for g in graded['grades'][:94])
        or any(r['status']!='missing' or r['attempts'] or r['raw_output'] is not None for r in graded['records'][94:])):
        raise ValueError('Completed prefix is not exactly the preserved 94 scored requests')
    execution=old/'candidate/execution/candidate-1'
    if not (execution/'generation-96-intent.json').exists() or (execution/'generation-96-ack.json').exists():
        raise ValueError('The interrupted next gate must remain unacknowledged')
    events=[json.loads(line) for line in (execution/'events.jsonl').read_text().splitlines()]
    calls=[e['data'] for e in events if e['type']=='generation-call-intent']
    if len(calls)!=95 or any(c['request_id']=='generation-96' for c in calls):
        raise ValueError('Unexpected generation after the completed prefix')
    if sum(len(i['target']['instruction_id_list']) for i in items[94:])!=45:
        raise ValueError('Remaining instruction denominator differs')
    return items,config,result,graded


def preserve(old,repo):
    """Snapshot historical scientific evidence and pinned sources read-only."""
    files={}
    for arm in ('baseline','candidate'):
        manifest=read_json(old/arm/'manifest.json')
        files.update(manifest['files'])
    skip={'benchmark','chrome-profile','__pycache__','pycache'}
    for current,dirs,names in os.walk(old,followlinks=False):
        dirs[:]=[d for d in dirs if d not in skip and not Path(current,d).is_symlink()]
        for name in names:
            p=Path(current,name)
            if p.is_symlink() or name.endswith(('-shm','-wal')):
                continue
            files[str(p)]=file_sha256(p)
    for p in (repo/'reports/ifeval-browser-v1-heldout').iterdir():
        if p.is_file():
            files[str(p)]=file_sha256(p)
    for name,digest in files.items():
        if file_sha256(name)!=digest:
            raise ValueError('Historical evidence changed: '+name)
    return {'kind':'continuation_historical_preservation','files':files,
        'excluded_ephemeral_browser_profiles_and_sqlite_sidecars':True,
        'original_model_bytes_verified_separately':True}


def prepare(old,root):
    from weight_space_lab.ifeval_candidate_continuation import LIMITS
    from weight_space_lab.ifeval_continuation_merge import prepare_merge_plan
    from weight_space_lab.ifeval_heldout_analysis import ANALYSIS_SPEC
    repo=Path.cwd().resolve()
    old,root=Path(old).resolve(),Path(root).resolve()
    if old==root or old in root.parents or any((root/name).exists() for name in ('manifest.json','continuation-plan.json',
        'generation-budget.sqlite3','execution-authorization.json','execution','load-grant.json')):
        raise FileExistsError('Continuation preparation must be new; never reset prior evidence or capacity')
    # Existing independent verifier checks original model bytes, source/data
    # identities, all completed records, exact stopped prefix, and cleanup.
    sys.path.insert(0,str(repo/'scripts'))
    from verify_ifeval_heldout import verify
    verification=verify(old)
    if not verification['passed'] or verification['complete_paired_evaluation']:
        raise ValueError('Verified incomplete original evaluation required')
    items,config,execution,graded=stopped_prefix(old)
    original=read_json(old/'evaluation-freeze.json')
    old_manifest=read_json(old/'candidate/manifest.json')
    assert original['analysis_specification']==ANALYSIS_SPEC
    assert original['settings']==SETTINGS and original['policy']==POLICY
    environment=read_json(repo/'artifacts/ifeval-browser-v1/evaluator-environment.json')
    assert list(sys.version_info[:3])==environment['python_version']
    assert {p:importlib.metadata.version(p) for p in environment['packages']}==environment['packages']
    historical=preserve(old,repo)
    benchmark=root/'benchmark'
    validate_browser_offline(old,root,repo)
    config=copy.deepcopy(config)
    config.update(runId='ifeval-browser-v1-heldout-continuation-candidate',
        evaluationStage='heldout-continuation',originalScheduleOffset=94,originalScheduleLength=128,
        items=config['items'][94:])
    assert len(config['items'])==34 and config['settings']==SETTINGS
    for asset in config['assets']:
        assert file_sha256(benchmark/'public'/urlparse(asset['url']).path.lstrip('/'))==asset['sha256']
    for asset in config['servedBuild']:
        asset['sha256']=file_sha256(benchmark/'public'/urlparse(asset['url']).path.lstrip('/'))
    config.pop('configurationSha256')
    config['configurationSha256']=canonical_hash(config)
    subprocess.run(['node','--import','tsx','--input-type=module','-e',
        "import {readFileSync} from 'node:fs'; import {taskEvaluationSchema} from './src/benchmark/task-evaluation/protocol.ts'; taskEvaluationSchema.parse(JSON.parse(readFileSync(0,'utf8')));"],
        input=json.dumps(config),text=True,capture_output=True,check=True,cwd=benchmark,timeout=30)
    root.mkdir(exist_ok=True,mode=0o700)
    root.chmod(0o700)
    save(root/'original-verification-before.json',verification)
    save(root/'preservation-before.json',historical)
    save(root/'remaining-items.json',items[94:])
    save(root/'baseline-config.json',config)
    save(root/'grading-policy.json',POLICY)
    mapping=[{'item_id':item['item_id'],'original_schedule_position':position,
              'original_generation_request_id':f'generation-{position+1}',
              'continuation_generation_request_id':f'generation-{position-93}',
              'item_sha256':canonical_hash(item),'instruction_count':len(item['target']['instruction_id_list'])}
             for position,item in enumerate(items[94:],95)]
    save(root/'schedule-mapping.json',mapping)
    prior_ledger_paths=[repo/'artifacts/ifeval-browser-v1/generation-budget.sqlite3',
        repo/'artifacts/ifeval-browser-v1/candidate-comparison/generation-budget.sqlite3',
        repo/'artifacts/ifeval-browser-v1/handoff/import-ledger.sqlite3',
        old/'baseline/generation-budget.sqlite3',old/'candidate/generation-budget.sqlite3']
    ledger_audit=[]
    for p in prior_ledger_paths:
        meta,rows=readonly_ledger(p)
        ledger_audit.append({'path':str(p),'sha256':file_sha256(p),'attempts':len(rows),
            'cancelled':bool(meta['cancelled']),'cancellation_reason':meta['cancellation_reason'],
            'retries':sum(r['retry_of'] is not None for r in rows)})
    save(root/'historical-ledger-audit.json',ledger_audit)
    # Bind all model and source bytes without re-importing, re-quantizing or
    # reconstructing a checkpoint. The browser uses the cloned existing files.
    model_pins={}
    for evidence in original['artifact_verification'].values():
        for path_key,map_key in [('source_path','source_files'),('deployed_path','deployed_files')]:
            for name,pin in evidence[map_key].items():
                model_pins[str(Path(evidence[path_key])/name)]=pin['sha256']
    extra=[root/name for name in ('remaining-items.json','baseline-config.json','grading-policy.json',
        'schedule-mapping.json','historical-ledger-audit.json','preservation-before.json',
        'original-verification-before.json','browser-continuation-offline.json','python-offline-checks.txt')]
    extra += [repo/name for name in ('scripts/prepare_ifeval_continuation.py',
        'weight_space_lab/ifeval_candidate_continuation.py','weight_space_lab/ifeval_supervisor_gate.py',
        'weight_space_lab/ifeval_continuation_merge.py','tests/test_ifeval_candidate_continuation.py',
        'tests/test_ifeval_supervisor_gate.py','tests/test_ifeval_continuation_merge.py',
        'tests/test_ifeval_continuation_preparation.py','integrations/ifeval-heldout-continuation.patch')]
    for folder in ('scripts','src','harness','tests/unit'):
        extra += [p for p in (benchmark/folder).rglob('*') if p.is_file() and not p.is_symlink()]
    extra += [benchmark/'package-lock.json',(benchmark/'node_modules/@mlc-ai/web-llm-next/lib/index.js').resolve()]
    extra += [benchmark/'public'/urlparse(a['url']).path.lstrip('/') for a in config['assets']+config['servedBuild']]
    pins={**historical['files'],**model_pins,**{str(p.resolve()):file_sha256(p) for p in extra}}
    manifest={'schema_version':1,'kind':'prepared_candidate_suffix_continuation','author':'Arhaan2',
        'experiment_id':'ifeval-browser-v1-heldout-continuation','arm':'candidate',
        'created_epoch_ms':time.time()*1000,'execution_authorized':False,'heldout_execution_authorized':False,
        'original_root':str(old),'original_evaluation_freeze_sha256':canonical_hash(original),
        'original_freeze_sha256':canonical_hash(original),
        'original_candidate_manifest_sha256':canonical_hash(old_manifest),
        'original_candidate_configuration_file_sha256':file_sha256(old/'candidate/baseline-config.json'),
        'original_candidate_config_sha256':file_sha256(old/'candidate/baseline-config.json'),
        'checkpoint':old_manifest['checkpoint'],'artifact_sha256':old_manifest['artifact_sha256'],
        'configuration_sha256':config['configurationSha256'],'items_path':str(root/'remaining-items.json'),
        'items_sha256':canonical_hash(items[94:]),'limits':LIMITS,'settings':SETTINGS,'policy':POLICY,
        'grader':old_manifest['grader'],'official_ifeval_root':old_manifest['official_ifeval_root'],
        'official_assets':old_manifest['official_assets'],'files':pins,'original_schedule_offset':94,
        'original_schedule_length':128,'scheduled_prompts':34,'scheduled_instructions':45,
        'scope':'manual operational continuation of exposed heldout subset; candidate unchanged',
        'editor_isolation':'unverified','historical_capacity_transferred':False,
        'proposal_session_provenance':original['proposal_session_provenance']}
    save(root/'manifest.json',manifest)
    merge=prepare_merge_plan(old,root/'manifest.json',root/'merge-plan.json')
    historical_report=read_json(repo/'reports/ifeval-browser-v1-heldout/result.json')
    usage=historical_report['aggregate_budget']
    plan={'schema_version':1,'kind':'frozen_continuation_preparation','author':'Arhaan2',
        'label':'128-example IFEval on-device subset — candidate suffix continuation',
        'created_epoch_ms':time.time()*1000,'execution_authorized':False,'authorization_status':'pending',
        'limits':LIMITS,'manifest_sha256':canonical_hash(manifest),'merge_plan_sha256':canonical_hash(merge),
        'original_evaluation_freeze_sha256':canonical_hash(original),
        'task_order':[i['item_id'] for i in items[94:]],'original_schedule_offset':94,'original_schedule_length':128,
        'scheduled_prompts':34,'scheduled_instructions':45,'schedule_mapping_sha256':canonical_hash(mapping),
        'selected_artifact_unchanged':True,'completed_results_reused_without_regrading':{'baseline':128,'candidate':94},
        'analysis_specification':ANALYSIS_SPEC,'historical_usage':usage,
        'cumulative_maximum_after_success':{'engine_load_attempts':3,'warmup_dispatches':3,
            'scored_dispatches':256,'generation_dispatches':259,
            'reported_generated_tokens_upper_bound':usage['reported_generated_tokens']+LIMITS['generated_tokens'],
            'reserved_generated_tokens':usage['reserved_generated_tokens']+LIMITS['generated_tokens']},
        'extra_authorization_reason':'Both original engine-load and warmup slots were consumed; the stopped ledger is closed. One fresh session adds one dispatch beyond the original aggregate ceiling and needs a new absolute session deadline. Old limits, clocks and ledgers are never reset or borrowed from.',
        'clock_policy':'New at-most-5400-second clock starts at continuation load grant and includes cleanup; original 10800-second absolute clock stays immutable and is not resumed. Report interruption/calendar gap and summed active session duration separately.',
        'stop_policy':'No automatic retries, extensions, new proposals, conversions, baseline reruns or completed-candidate reruns; preserve partial suffix and verify cleanup on any failure.',
        'completion_policy':'Merge only separately validated records and grades under their real session/configuration identities. Preserve original incomplete report; write a new continuation report, disclosing restarted candidate session and the exposed subset.',
        'new_engine_load_attempts':0,'new_generation_dispatches':0,'new_generated_tokens':0,'new_grader_calls':0,
        'production_ledger_created':False,'authorization_created':False,'original_records_modified':False}
    save(root/'continuation-plan.json',plan)
    save(root/'ready.json',{'continuation_plan_sha256':canonical_hash(plan)})
    from weight_space_lab.ifeval_candidate_continuation import load
    load(root,require_authorization=False)
    forbidden=['generation-budget.sqlite3','execution-authorization.json','authorization-source.txt','execution','load-grant.json']
    assert not any((root/name).exists() for name in forbidden)
    assert all(file_sha256(p)==h for p,h in historical['files'].items())
    save(root/'preparation-verification.json',{'passed':True,'continuation_plan_sha256':canonical_hash(plan),
        'historical_files_unchanged':len(historical['files']),'all_prepared_pins_verified':len(pins),
        'exact_remaining_prompts':34,'exact_remaining_instructions':45,'new_engine_load_attempts':0,
        'new_generation_dispatches':0,'new_grader_calls':0,'production_ledger_created':False,
        'authorization_created':False,'execution_authorized':False})
    return {'prepared':True,'execution_authorized':False,'continuation_plan_sha256':canonical_hash(plan),
        'positions':[95,128],'prompts':34,'instructions':45,'limits':LIMITS}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',default='artifacts/ifeval-browser-v1-heldout')
    parser.add_argument('--destination',default='artifacts/ifeval-browser-v1-heldout-continuation')
    args=parser.parse_args()
    print(json.dumps(prepare(args.original,args.destination),indent=2))
