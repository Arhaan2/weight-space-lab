#!/usr/bin/env python3
"""Freeze the single authorized 128-example held-out comparison. Arhaan2.

Preparation only: no engine, converter, proposal import or model generation.
All private identities and authorization remain in the ignored artifact tree.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlparse

from weight_space_lab.analysis_pipeline import build_analysis_manifest
from weight_space_lab.browser_candidate import _files
from weight_space_lab.ifeval_baseline import load as load_development, save, SETTINGS, POLICY
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash

EXPECTED = {
    'baseline': {'source': 'f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d',
                 'artifact': '5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0'},
    'candidate': {'source': 'f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf',
                  'artifact': '6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b'},
}


def prior_execution_audit(old, destination, ids):
    """Inspect dispatch metadata, not response contents, before first exposure."""
    matches, scanned = [], []
    for folder in [old, destination]:
        for p in sorted(folder.rglob('*-intent.json')):
            if 'node_modules' in p.parts or 'benchmark' in p.parts:
                continue
            value = read_json(p)
            scanned.append({'path': str(p), 'sha256': file_sha256(p)})
            if value.get('kind') == 'generation' and value.get('item_id') in ids:
                matches.append(str(p))
    historical_ledgers = []
    for p in sorted(old.rglob('*.sqlite3')):
        if 'node_modules' in p.parts or 'benchmark' in p.parts:
            continue
        with sqlite3.connect(p.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
            rows = db.execute('SELECT status,retry_of FROM attempts').fetchall()
        historical_ledgers.append({'path': str(p), 'sha256': file_sha256(p),
                                   'attempts': len(rows), 'states': rows})
    if matches or any((destination/a/'execution').exists() for a in EXPECTED):
        raise ValueError('Existing held-out execution: refuse duplicate capacity')
    return {'checked_epoch_seconds': time.time(), 'heldout_dispatches_found': 0,
            'scope': 'preserved ifeval-browser-v1 and this destination dispatch metadata; no response inspection',
            'dispatch_metadata': scanned, 'historical_ledgers': historical_ledgers}


def preserve(old, repo):
    files = {}
    # Hash historical evidence, excluding immutable dependencies and bulky model
    # copies whose complete file maps are independently verified below.
    skip = {'node_modules', '__pycache__', 'pycache', '.git', 'evaluator-env', 'git-object-recovery'}
    for current, dirs, names in os.walk(old, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not Path(current,d).is_symlink()
                   and not (d == 'models' and Path(current).name == 'public')
                   and not (d in {'converted', 'deployed', 'source'} and Path(current).name == 'candidate')]
        for name in names:
            p = Path(current,name)
            if p.is_symlink() or name.endswith(('-shm', '-wal')):
                continue
            files[str(p)] = file_sha256(p)
    for d in ('reports/ifeval-browser-v1', 'reports/ifeval-browser-v1-candidate',
              'reports/ifeval-browser-v1-candidate-recovery'):
        for p in (repo/d).rglob('*'):
            if p.is_file() and not p.is_symlink():
                files[str(p)] = file_sha256(p)
    return {'files': files, 'excluded_sqlite_ephemeral_sidecars': True,
            'model_artifacts_verified_separately': True}


def freeze(old, destination, authorization):
    from weight_space_lab.ifeval_heldout_runner import PER_ARM_LIMITS, AGGREGATE_LIMITS, ledger
    from weight_space_lab.ifeval_heldout_analysis import ANALYSIS_SPEC
    repo = Path.cwd().resolve()
    old, destination = Path(old).resolve(), Path(destination).resolve()
    destination.mkdir(exist_ok=True, mode=0o700)
    if (destination/'evaluation-freeze.json').exists() or (destination/'baseline').exists():
        raise FileExistsError('Freeze/capacity already exists; never reset it')
    baseline = load_development(old)[1]
    candidate = load_development(old/'candidate-comparison')[1]
    items = read_json(old/'data/evaluation-items.json')
    data = read_json(old/'data/manifest.json')
    assert len(items) == 128 and sum(len(i['target']['instruction_id_list']) for i in items) == 173
    assert all(i['split'] == 'test' and i['benchmark'] == 'ifeval' for i in items)
    assert canonical_hash(items) == data['evaluation']['items_sha256']
    assert [i['item_id'] for i in items] == data['evaluation']['item_ids']
    audit = prior_execution_audit(old,destination,{i['item_id'] for i in items})
    save(destination/'prior-execution-audit.json',audit)
    save(destination/'preservation-before.json',preserve(old,repo))
    shutil.copyfile(authorization,destination/'authorization-source.txt')
    (destination/'authorization-source.txt').chmod(0o600)
    shutil.copyfile(old/'data/evaluation-items.json',destination/'evaluation-items.json')
    (destination/'evaluation-items.json').chmod(0o600)
    # Rehash actual saved source and deployed bytes, using the existing identity
    # convention (canonical complete file map), without a conversion or import.
    host = read_json(old/'weight-information/host.json')
    original = read_json(repo/'artifacts/browser-integration-v1/conversion/artifacts.json')
    edited = read_json(old/'candidate/deployed/candidate-artifact.json')
    source_maps = {'baseline':host['source_files'],
                  'candidate':read_json(old/'candidate/deployed/source-audit.json')['model_inventory']['files']}
    source_paths = {'baseline':Path(host['model_path']), 'candidate':old/'candidate/source/model'}
    deployed = {'baseline':original['arms']['baseline'], 'candidate':edited}
    checked = {}
    for arm in EXPECTED:
        observed = _files(source_paths[arm])
        assert observed == source_maps[arm] and canonical_hash(observed) == EXPECTED[arm]['source']
        actual = _files(deployed[arm]['path'])
        assert actual == deployed[arm]['files'] and canonical_hash(actual) == EXPECTED[arm]['artifact']
        checked[arm] = {'source_checkpoint_sha256':canonical_hash(observed),
                        'artifact_sha256':canonical_hash(actual), 'source_files':observed,
                        'deployed_files':actual, 'source_path':str(source_paths[arm]),
                        'deployed_path':deployed[arm]['path']}
    for name in edited['shared_runtime_assets']:
        assert checked['baseline']['deployed_files'][name] == checked['candidate']['deployed_files'][name]
    comparison = read_json(old/'candidate-comparison/comparison.json')
    assert comparison['selection']['selected_arm'] == 'candidate'
    assert comparison['selection']['technical_and_grading_complete'] is True
    env = read_json(old/'evaluator-environment.json')
    assert list(sys.version_info[:3]) == env['python_version']
    assert {p:importlib.metadata.version(p) for p in env['packages']} == env['packages']
    benchmark = destination/'benchmark'
    token_audit = read_json(old/'data/tokenizer-count-audit.json')
    assert file_sha256((benchmark/'node_modules/@mlc-ai/web-llm-next/lib/index.js').resolve()) == token_audit['runtime_sha256']
    assert file_sha256(old/'data/tokenizer-count-audit.json') == data['bundled_tokenizer_audit']['sha256']
    counts = {i['item_id']:i['rendered_input_tokens'] for i in token_audit['counts']}
    configs, manifests = {}, {}
    for arm, prior_manifest, config_path in [
        ('baseline',baseline,old/'baseline-config.json'),
        ('candidate',candidate,old/'candidate-comparison/baseline-config.json')]:
        target = destination/arm
        target.mkdir(mode=0o700)
        (target/'benchmark').symlink_to(benchmark,target_is_directory=True)
        shutil.copyfile(destination/'authorization-source.txt',target/'authorization-source.txt')
        config = read_json(config_path)
        config.update(runId='ifeval-browser-v1-heldout-'+arm,evaluationStage='heldout')
        config['items'] = [{'item_id':i['item_id'],'prompt':i['task_text'],
                            'rendered_input_tokens':counts[i['item_id']]} for i in items]
        assert all(0 < i['rendered_input_tokens'] <= 3072 for i in config['items'])
        for entry in config['servedBuild']:
            entry['sha256'] = file_sha256(benchmark/'public'/urlparse(entry['url']).path.lstrip('/'))
        for entry in config['assets']:
            assert file_sha256(benchmark/'public'/urlparse(entry['url']).path.lstrip('/')) == entry['sha256']
        config.pop('configurationSha256')
        config['configurationSha256'] = canonical_hash(config)
        save(target/'baseline-config.json',config)
        subprocess.run(['node','--import','tsx','--input-type=module','-e',
            "import {readFileSync} from 'node:fs'; import {taskEvaluationSchema} from './src/benchmark/task-evaluation/protocol.ts'; taskEvaluationSchema.parse(JSON.parse(readFileSync(process.argv[1])));",
            str(target/'baseline-config.json')],cwd=benchmark,check=True,capture_output=True,text=True,timeout=30)
        save(target/'grading-policy.json',POLICY)
        manifest = copy.deepcopy(prior_manifest)
        manifest.update(kind='ifeval_browser_frozen_heldout_arm',experiment_id='ifeval-browser-v1-heldout',
            created_epoch_ms=time.time()*1000,limits=PER_ARM_LIMITS,settings=SETTINGS,
            configuration_sha256=config['configurationSha256'],items_path=str(destination/'evaluation-items.json'),
            items_sha256=canonical_hash(items),heldout_execution_authorized=True,
            denominator='all 128 scheduled prompts and all 173 scheduled instructions, including missing/error slots',
            authorization_source_sha256=file_sha256(destination/'authorization-source.txt'))
        extra = [destination/'evaluation-items.json',target/'baseline-config.json',target/'grading-policy.json',
                 destination/'authorization-source.txt',destination/'prior-execution-audit.json',
                 destination/'preservation-before.json',old/'proposal-provenance.json',
                 old/'candidate-comparison/comparison.json',old/'candidate/deployed/candidate-artifact.json',
                 old/'candidate/deployed/source-audit.json',old/'candidate/converted-source-binding.json',
                 repo/'scripts/freeze_ifeval_heldout.py',repo/'scripts/run_ifeval_heldout.py',
                 repo/'weight_space_lab/ifeval_heldout_runner.py',repo/'weight_space_lab/ifeval_heldout_analysis.py',
                 repo/'tests/test_ifeval_heldout_runner.py',repo/'tests/test_ifeval_heldout_analysis.py']
        # Bind actual new scheduling sources, served bundles, and unchanged model
        # runtime. Original development pins remain as historical integrity pins.
        for folder in ('scripts','src','harness','tests/unit'):
            extra += [p for p in (benchmark/folder).rglob('*') if p.is_file() and not p.is_symlink()]
        extra += [benchmark/'package-lock.json', (benchmark/'node_modules/@mlc-ai/web-llm-next/lib/index.js').resolve()]
        extra += [benchmark/'public'/urlparse(e['url']).path.lstrip('/') for e in config['servedBuild']]
        extra += [destination/'browser-schedule-offline.json',destination/'python-offline-checks.txt']
        manifest['files'].update({str(p.resolve()):file_sha256(p) for p in extra})
        save(target/'manifest.json',manifest)
        save(target/'ready.json',{'manifest_sha256':canonical_hash(manifest)})
        manifests[arm],configs[arm] = manifest,config
    split = {'schema_version':1,'kind':'benchmark_split','split':'test',
             'source_manifest_sha256':canonical_hash(data),
             'items':[{'benchmark':i['benchmark'],'item_id':i['item_id'],
                       'dataset_revision':i['dataset_revision'],'item_sha256':canonical_hash(i)} for i in items]}
    save(destination/'paired-split-manifest.json',split)
    paired = {}
    for mode in ('strict','loose'):
        value = build_analysis_manifest(split,items,benchmark='ifeval',seeds=[42],policy=POLICY,
            arms={a:{'run_id':configs[a]['runId'],'profile':'browser_evaluation',
                     'configuration_sha256':configs[a]['configurationSha256'],'checkpoint':manifests[a]['checkpoint']}
                  for a in EXPECTED},stage='full',metric='prompt_level_'+mode+'_accuracy',evidence_kind='real_model')
        save(destination/f'paired-analysis-{mode}.json',value)
        paired[mode] = value['manifest_sha256']
    evaluation = {'schema_version':1,'kind':'single_final_heldout_evaluation','author':'Arhaan2',
        'label':'128-example IFEval on-device subset','created_epoch_ms':time.time()*1000,
        'execution_order':['baseline','candidate'],'limits':AGGREGATE_LIMITS,
        'arms':{a:{'manifest_sha256':canonical_hash(manifests[a]),**EXPECTED[a]} for a in EXPECTED},
        'authorization_source_sha256':file_sha256(destination/'authorization-source.txt'),
        'artifact_verification':checked,'original_data_manifest_sha256':canonical_hash(data),
        'original_data_manifest_file_sha256':file_sha256(old/'data/manifest.json'),
        'evaluation_items_sha256':canonical_hash(items),'scheduled_prompts':128,'scheduled_instructions':173,
        'original_source':data['sources']['evaluation'],'task_order':[i['item_id'] for i in items],
        'analysis_specification':ANALYSIS_SPEC,'paired_analysis_manifests':paired,
        'policy':POLICY,'grader':baseline['grader'],'settings':SETTINGS,
        'warmup_prompt':configs['baseline']['warmupPrompt'], 'quantization':'q4f16_1',
        'tokenizer_sha256':data['tokenizer_sha256'],'template_sha256':data['template_sha256'],
        'runtime_sha256':token_audit['runtime_sha256'],'model_library_sha256':original['model_library_sha256'],
        'proposal_session_provenance':read_json(old/'proposal-provenance.json'),
        'editor_isolation':'unverified','scope':'exploratory on-device experiment',
        'selected_on_development':True,'selection_changed':False,'historical_capacity_transferred':False,
        'after_evaluation':'subset exposed; no feedback into another weight-editing iteration',
        'source_sha256':file_sha256(__file__)}
    save(destination/'evaluation-freeze.json',evaluation)
    save(destination/'evaluation-ready.json',{'evaluation_freeze_sha256':canonical_hash(evaluation)})
    for arm in EXPECTED:
        with ledger(destination/arm,manifests[arm],create=True) as book:
            save(destination/arm/'accounting-before.json',book.summary())
    save(destination/'authorization-accounting-before.json',{'limits':AGGREGATE_LIMITS,
         'engine_load_attempts':0,'generation_dispatches':0,'generated_tokens':0,
         'historical_capacity_transferred':False,'evaluation_freeze_sha256':canonical_hash(evaluation)})
    return {'frozen':True,'evaluation_freeze_sha256':canonical_hash(evaluation),
            'order':evaluation['execution_order'],'prompts':128,'instructions':173,'engine_loads':0}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',default='artifacts/ifeval-browser-v1')
    parser.add_argument('--destination',default='artifacts/ifeval-browser-v1-heldout')
    parser.add_argument('--authorization',required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.old,args.destination,args.authorization),indent=2))
