"""Separately frozen resource-only comparison of five existing deployed artifacts.

No editing, conversion, quality grading, frontier calls or changes to old studies.
Inference/telemetry/merging reuse the preserved original resource runtime.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

from . import browser_resource as resource
from . import ifeval_v2_runtime as runtime
from .browser_check import _collector
from .inventory import file_sha256, read_json
from .protocol import canonical_hash
from .native_runner import _lock
from .ifeval_baseline import save
from .ifeval_v2_study import atomic, retained_bytes, prepare_benchmark, CLEANUP, bounded_python

ORDER = ['pristine','v1','v2-r1','v2-r2','v2-r3','v2-r3','v2-r2','v2-r1','v1','pristine']
LIMITS = {'sessions':10,'engine_loads':10,'warmups':10,'measured_requests':600,'dispatches':610,
    'reserved_target_tokens':38560,'browser_seconds':5400,'local_seconds':7200,'calendar_seconds':86400,
    'retained_bytes':32*1024**3,'minimum_free_bytes':8*1024**3,'automatic_retries':0,'frontier_calls':0}
SUMMARY_KEYS = ('status','scheduled_requests','measured_requests','warmups','scored_output_tokens','warmup_output_tokens',
    'response_ms_sum','load_ms','session_ms','natural_completions','truncations','idle_gaps_ms',
    'gpu_buffer_descriptor_bytes_sampled_max','wasm_address_space_bytes_sampled_max',
    'native_peak_footprint_gb','native_idle_footprint_gb','native_recovery_footprint_gb',
    'cpu_capacity_percent','gpu_time_ms_per_second','host_power_watts','cpu_coverage','gpu_coverage','power_coverage',
    'native_coverage','energy','model_lib','runtime','quantization')
METRICS = ('scored_output_tokens','response_ms_sum','load_ms','session_ms','mean_response_ms',
    'full_response_tokens_per_second','response_ms_per_output_token','gpu_buffer_descriptor_bytes_sampled_max',
    'wasm_address_space_bytes_sampled_max','native_peak_footprint_gb','native_idle_footprint_gb',
    'native_recovery_footprint_gb','cpu_capacity_percent','gpu_time_ms_per_second','host_power_watts',
    'cpu_coverage','gpu_coverage','power_coverage')
COUNTERS = {
 'mean_response_ms':'Sum of measured full-response milliseconds divided by measured requests; warm-up excluded.',
 'full_response_tokens_per_second':'Measured sampled output tokens / total full-response seconds. Not decode-only throughput.',
 'response_ms_per_output_token':'Total measured full-response milliseconds / sampled output tokens. Null for no output tokens.',
 'native_peak_footprint_gb':'Sampled maximum owned Chrome process-tree physical footprint, decimal GB (1e9 bytes); shared memory may be counted across processes. Not dedicated model memory or physical VRAM.',
 'native_idle_footprint_gb':'Median owned process-tree footprint in the original final loaded-idle window, decimal GB.',
 'native_recovery_footprint_gb':'Median owned process-tree footprint in the original recovery window, decimal GB.',
 'cpu_capacity_percent':'Original interval-weighted CPU percentage of total logical-CPU capacity during profile active-phase windows; partial/boundary-crossing intervals excluded.',
 'gpu_time_ms_per_second':'Experimental AGX driver-accounted command-buffer milliseconds per second for the owned Chrome process tree, including rendering. Concurrent queues may overlap; not GPU utilization/capacity, shader occupancy or model-only GPU time.',
 'host_power_watts':'Host-wide observed power, only if captured; unavailable in unprivileged capture, never zero-filled.',
 'gpu_buffer_descriptor_bytes_sampled_max':'Maximum sampled GPUBuffer allocation descriptors; not physical VRAM.',
 'wasm_address_space_bytes_sampled_max':'Maximum sampled WASM address space; not resident physical memory.',
 'coverage':'Original collector coverage fractions/records retained. Missing signals and excluded intervals are not treated as zero.'}


def validate_config(config):
    if config.get('session_order') != ORDER or config.get('limits') != LIMITS or config.get('quality_reruns') != 0:
        raise ValueError('Resource-only schedule/limits changed')
    if set(config.get('expected_artifacts',{})) != set(ORDER) or len(set(config['expected_artifacts'].values())) != 5:
        raise ValueError('Five distinct frozen artifact identities required')


def ensure_file(path, expected=None, events=None):
    """Materialize a placeholder by one bounded normal read; never invent bytes."""
    path=Path(path)
    before=path.stat()
    if getattr(before,'st_flags',0)&0x40000000:
        script='import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); n=0\nfor b in iter(lambda:f.read(1048576),b""): h.update(b); n+=len(b)\nprint(n,h.hexdigest())'
        result=subprocess.run([sys.executable,'-c',script,str(path)],capture_output=True,text=True,timeout=25,check=True)
        size,digest=result.stdout.strip().split()
        if int(size)!=before.st_size or (expected and digest!=expected):
            raise ValueError('Materialized file does not match frozen bytes')
        if events is not None: events.append({'path':str(path),'size_bytes':int(size),'sha256':digest,'method':'bounded ordinary filesystem read'})
    value=file_sha256(path)
    if expected and value!=expected: raise ValueError('Frozen file bytes changed: '+path.name)
    return value


def reserve(state, position, limits, now, free_bytes, stored_bytes):
    """One durable intent consumes its session; only the fixed next position may run."""
    if state['halted'] or state['active'] is not None: raise RuntimeError('Stopped/interrupted study; no redispatch')
    if position!=sum(r['status']=='completed' for r in state['sessions'])+1:
        raise ValueError('Fixed order or completed predecessor missing')
    row=state['sessions'][position-1]
    if row['status']!='pending': raise ValueError('Session already attempted; no retries')
    if now>=state['deadline_epoch_seconds']: raise TimeoutError('Calendar exhausted')
    seconds=min(1800,limits['browser_seconds']-state['charged_browser_seconds'],
        limits['local_seconds']-state['charged_local_seconds'],state['deadline_epoch_seconds']-now)
    if seconds<300: raise TimeoutError('Insufficient remaining session/cleanup allowance')
    if free_bytes<limits['minimum_free_bytes'] or stored_bytes>limits['retained_bytes']:
        raise RuntimeError('Storage limit reached')
    row.update(status='running',started_epoch_seconds=now,reserved_seconds=seconds,
        capacity_reserved={'loads':1,'warmups':1,'measured_requests':60,'dispatches':61,'target_tokens':3856},load_granted=False)
    state['active']={'position':position,'pid':os.getpid(),'reserved_seconds':seconds,'started_epoch_seconds':now}
    return seconds


def completed_execution(execution, summary):
    return bool(execution and execution.get('status')=='completed' and summary and summary.get('status')=='complete'
        and execution.get('engine_load_attempts')==1 and execution.get('warmup_dispatches')==1
        and execution.get('scored_dispatches')==60 and execution.get('generation_dispatches')==61
        and summary.get('measured_requests')==60 and summary.get('warmups')==1
        and all(execution.get('cleanup',{}).get(k) is True for k in (*CLEANUP,'native_capture_exit_confirmed'))
        and execution['cleanup'].get('retained_owned_pids_alive')==[])


def finish(state, position, elapsed, execution, summary, error=None):
    row=state['sessions'][position-1]
    if state['active'] is None or state['active']['position']!=position: raise ValueError('No matching active reservation')
    good=completed_execution(execution,summary) and elapsed<=row['reserved_seconds'] and error is None
    row.update(status='completed' if good else 'stopped',charged_seconds=elapsed,error=error)
    if execution:
        row['observed']={k:execution.get(k) for k in ('status','engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches','generation_budget','cleanup')}
    state['charged_browser_seconds']+=elapsed
    state['charged_local_seconds']+=elapsed
    state['active']=None
    state['halted']=not good
    return good


def _source_files():
    root=Path(__file__).resolve().parents[1]
    names='__init__ acquisition analysis_pipeline benchmark_data browser_candidate browser_check browser_conversion browser_integration_v2 browser_resource browser_tasks browser_v2_runner checkpoint checkpoint_numeric editor_budget editor_contract editor_tools evaluation grading ifeval_baseline ifeval_candidate_continuation ifeval_handoff ifeval_heldout_runner ifeval_supervisor_gate ifeval_v2_runtime ifeval_v2_study inventory local_pilot native_assets native_backend native_runner patches phase8 protocol published_study reconstruction records replay resource_followup'.split()
    return [*[root/'weight_space_lab'/(name+'.py') for name in names],root/'configs/resource_all_edits.json',root/'docs/resource-all-edits-protocol.md',root/'integrations/browser-resource-comparison.patch']


def _artifact_inputs():
    original=Path('artifacts/ifeval-browser-v2-collaboration-continuation').resolve()
    state=read_json(original/'state.json')
    paths={'v1':Path('artifacts/ifeval-browser-v1/candidate/deployed/candidate-artifact.json').resolve()}
    paths.update({f'v2-r{i}':Path(state['slots'][f'r{i}-astra']['artifact_path']) for i in (1,2,3)})
    conversion=Path('artifacts/browser-integration-v1/conversion/artifacts.json').resolve()
    ensure_file(conversion)
    baseline=read_json(conversion)['arms']['baseline']
    return original,conversion,baseline,paths


def prepare(root, config_path):
    start=time.monotonic();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (root/'study.json').exists() or (root/'state.json').exists(): raise FileExistsError('Never replace an existing study')
    config=read_json(config_path);validate_config(config);hydration=[]
    original,conversion,base,paths=_artifact_inputs()
    prior=read_json(original/'study.json');source_manifest=read_json(original/'sessions/r3-astra/manifest.json')
    source=original/'benchmark'
    for path,digest in source_manifest['files'].items():
        if Path(path).is_relative_to(source):ensure_file(path,digest,hydration)
    artifacts={'pristine':{**base,'manifest_path':str(conversion),'manifest_sha256':ensure_file(conversion)}}
    for key,path in paths.items():
        ensure_file(path,events=hydration);artifacts[key]={**read_json(path),'manifest_path':str(path),'manifest_sha256':ensure_file(path)}
    for key,a in artifacts.items():
        if a['artifact_sha256']!=config['expected_artifacts'][key] or canonical_hash(a['files'])!=a['artifact_sha256']:
            raise ValueError('Wrong artifact supplied for '+key)
        for name,value in a['files'].items():
            ensure_file(Path(a['path'])/name,value['sha256'],hydration)
        if key!='pristine' and a['source_checkpoint_sha256']==base['source_checkpoint_sha256']:
            raise ValueError('Edited artifact source must differ from pristine')
    for name in ('model.wasm','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','mlc-chat-config.json'):
        if len({a['files'][name]['sha256'] for a in artifacts.values()})!=1:raise ValueError('Deployment control differs: '+name)
    quality=read_json('reports/ifeval-browser-v2-collaboration-run/result.json')
    v1_proposal=Path('artifacts/ifeval-browser-v1/candidate/source/proposal.json').resolve()
    ensure_file(v1_proposal,events=hydration)
    for key,a in artifacts.items():
        a['weight_operations']=[] if key=='pristine' else read_json(v1_proposal)['operations'] if key=='v1' else quality['rounds'][int(key[-1])-1]['weight_operations']
    # Reuse the already-built patched harness, never the integration smoke candidate.
    context={'benchmark_source':str(source),'baseline':prior['baseline']}
    benchmark=prepare_benchmark(root,context)
    for key,a in artifacts.items():
        if key=='pristine':continue
        target=benchmark/'public/models'/('v2-'+a['artifact_sha256'])/'resolve'/a['artifact_sha256']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(a['path'],target,copy_function=os.link)
    preset=resource.inspect_preset(benchmark)
    preset['changes']=['Five exact existing artifacts, two sessions each in fixed mirrored order; no quality gate.',
        'Original standard request/idle/reset settings unchanged; unprivileged capture; fresh dedicated profiles and durable gates.']
    env=resource.environment_identity()
    collector=_collector(benchmark);signals,unavailable=collector.host_signals()
    if signals.get('hostMemoryPressureLevel') is None or signals.get('hostThermalState') is None:
        raise RuntimeError('Native host resource signals unavailable before freeze')
    if signals['hostMemoryPressureLevel']>=4 or signals['hostThermalState']>=2:raise RuntimeError('Host resource stop before freeze')
    owned=collector.usage(os.getpid())
    preflight={'kind':'read_only_native_resource_preflight','environment':env,'host_signals':signals,'unavailable':unavailable,
        'own_process_usage':owned,'target_calls':0,'materializations':hydration}
    save(root/'native-preflight.json',preflight)
    protected=[original/'state.json',original/'study-result.json',original/'final-selection.json',
        Path('artifacts/ifeval-browser-v2-narrowed/state.json').resolve(),
        Path('reports/ifeval-browser-v1-final/result.json').resolve(),Path('reports/ifeval-browser-v2-collaboration-run/result.json').resolve()]
    files={str(p):ensure_file(p) for p in _source_files()}
    files[str(v1_proposal)]=ensure_file(v1_proposal)
    files.update({str(p):ensure_file(p) for p in protected})
    for p in (root/'preflight-artifacts.json',root/'native-preflight.json',root/'preflight-source-scope.json'):
        if p.exists():files[str(p)]=ensure_file(p)
    for a in artifacts.values():files[a['manifest_path']]=a['manifest_sha256']
    for p in (Path(prior['baseline']['baseline_root'])/'baseline-config.json',Path(prior['baseline']['baseline_root'])/'manifest.json'):
        files[str(p)]=ensure_file(p)
    benchmark_files={str(p):ensure_file(p) for p in benchmark.rglob('*') if p.is_file() and not p.is_symlink()
        and p.suffix not in ('.pyc','.pyo') and not any(x in p.relative_to(benchmark).parts for x in ('models','node_modules','artifacts','.next','out','__pycache__'))}
    files.update(benchmark_files)
    frozen=time.time()
    source_root=Path(__file__).resolve().parents[1]
    source_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=source_root,capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    study={'schema_version':1,'kind':'resource_only_existing_edits','config':config,'baseline':prior['baseline'],
        'resource_preset':preset,'artifacts':artifacts,'environment':env,'files':files,
        'source_code':{'base_commit':source_commit,'module_files':{p.name:files[str(p)] for p in _source_files() if p.suffix=='.py'},
            'route':'Clean published source checkout plus separately frozen resource-only controller; workspace artifact paths retained.'},
        'served_build':resource.served_build(benchmark,'http://127.0.0.1:4198'),'created_epoch_seconds':frozen,
        'deadline_epoch_seconds':frozen+LIMITS['calendar_seconds'],'authorization':'New explicit user request to measure all existing edits regardless of quality qualification; no new editor or quality runs.',
        'historical_studies_unchanged':{str(p):files[str(p)] for p in protected}}
    save(root/'study.json',study)
    sha=canonical_hash(study);save(root/'ready.json',{'study_sha256':sha})
    comparison={'kind':'five_artifact_resource_only_preset','session_order':ORDER,'settings':resource.RESOURCE_SETTINGS,
        'artifact_identities':config['expected_artifacts'],'study_sha256':sha,'limits':LIMITS,'cache_policy':config['cache_policy']}
    save(root/'resource-comparison-preset.json',comparison)
    save(root/'final-selection.json',{'kind':'adapter_schedule_binding_only_not_a_quality_selection','resource_preset_sha256':canonical_hash(comparison)})
    elapsed=time.monotonic()-start
    state={'study_sha256':sha,'deadline_epoch_seconds':study['deadline_epoch_seconds'],'halted':False,'active':None,
        'preparation_seconds':elapsed,'charged_local_seconds':elapsed,'charged_browser_seconds':0,'verification_stages':[],
        'sessions':[{'position':i+1,'artifact_key':key,'key':f'{i+1:02d}-{key}','status':'pending'} for i,key in enumerate(ORDER)]}
    atomic(root/'state.json',state)
    if retained_bytes(root)>LIMITS['retained_bytes'] or shutil.disk_usage(root).free<LIMITS['minimum_free_bytes']:
        state['halted']=True;atomic(root/'state.json',state);raise RuntimeError('Storage ceiling failed at freeze')
    return {'study_sha256':sha,'sessions':ORDER,'preparation_seconds':elapsed,'environment':env,'target_calls':0}


def load(root, verify=True):
    root=Path(root).resolve();study=read_json(root/'study.json');state=read_json(root/'state.json')
    sha=canonical_hash(study)
    if read_json(root/'ready.json')!={'study_sha256':sha} or state['study_sha256']!=sha:
        raise ValueError('Study/state freeze changed')
    validate_config(study['config'])
    if state['deadline_epoch_seconds']!=study['deadline_epoch_seconds']:raise ValueError('Calendar changed')
    if [r['artifact_key'] for r in state['sessions']]!=ORDER:raise ValueError('Session order changed')
    if verify:
        for p,digest in study['files'].items():ensure_file(p,digest)
    return root,study,state


def run_one(root, position):
    began=time.monotonic()
    root,study,state=load(root,verify=False)
    if state['halted'] or state['active'] is not None:raise RuntimeError('Stopped/interrupted study; no redispatch')
    row=state['sessions'][position-1];artifact=study['artifacts'][row['artifact_key']]
    if row['status']!='pending':raise ValueError('Session already attempted; no retries')
    # Full integrity/materialization/environment checks are charged to the local clock.
    try:
        preflight_limit=min(300,LIMITS['local_seconds']-state['charged_local_seconds'],state['deadline_epoch_seconds']-time.time())
        if preflight_limit<=0:raise TimeoutError('Local/calendar budget exhausted before verification')
        with bounded_python(preflight_limit):
            load(root)
            if resource.environment_identity()!=study['environment']:raise ValueError('Frozen device/browser environment changed')
            for name,value in artifact['files'].items():ensure_file(Path(artifact['path'])/name,value['sha256'])
    except BaseException as exc:
        elapsed=time.monotonic()-began
        state['charged_local_seconds']+=elapsed
        state['verification_stages'].append({'position':position,'charged_seconds':elapsed,'status':'failed'})
        row.update(status='stopped',error=type(exc).__name__+': '+str(exc),dispatch_absence_verified=True)
        state['halted']=True;atomic(root/'state.json',state)
        return {'position':position,'artifact_key':row['artifact_key'],'status':'stopped','continue':False,'error_type':type(exc).__name__}
    elapsed=time.monotonic()-began
    state['charged_local_seconds']+=elapsed
    state['verification_stages'].append({'position':position,'charged_seconds':elapsed,'status':'completed'})
    atomic(root/'state.json',state)
    start=time.monotonic()
    try:
        seconds=reserve(state,position,LIMITS,time.time(),shutil.disk_usage(root).free,retained_bytes(root))
    except BaseException as exc:
        row.update(status='stopped',error=type(exc).__name__+': '+str(exc),dispatch_absence_verified=True)
        state['halted']=True;atomic(root/'state.json',state)
        return {'position':position,'artifact_key':row['artifact_key'],'status':'stopped','continue':False,'error_type':type(exc).__name__}
    atomic(root/'state.json',state)
    execution=None;summary=None;error=None
    try:
        with bounded_python(seconds):
            session=resource.prepare_session(root,study,row['key'],None if row['artifact_key']=='pristine' else artifact)
            def on_load():
                if row['load_granted']:raise RuntimeError('Repeated load forbidden')
                # Keep 95 seconds inside this reservation for original bounded merger/finalization.
                remaining=seconds-(time.monotonic()-start)-95
                if remaining<80:raise TimeoutError('Insufficient request/cleanup/finalization allowance')
                row['load_granted']=True;atomic(root/'state.json',state)
                return remaining
            execution=resource.execute(session,on_load=on_load,preparation_seconds=min(300,max(.001,seconds-(time.monotonic()-start)-95)))
            summary_path=session/'execution'/('baseline-1' if row['artifact_key']=='pristine' else 'candidate-1')/'resource-summary.json'
            if summary_path.exists():summary=read_json(summary_path)
            if not completed_execution(execution,summary):error='Resource session incomplete or cleanup unconfirmed'
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        session=root/'sessions'/row['key'];arm='baseline-1' if row['artifact_key']=='pristine' else 'candidate-1'
        path=session/'execution'/arm/'supervisor-result.json'
        if path.exists():execution=read_json(path)
    elapsed=time.monotonic()-start
    good=finish(state,position,elapsed,execution,summary,error)
    atomic(root/'state.json',state)
    return {'position':position,'artifact_key':row['artifact_key'],'status':row['status'],'elapsed_seconds':elapsed,
        'measured_requests':None if summary is None else summary.get('measured_requests'),'continue':good}


def aggregate_sessions(sessions, matched_environment=True):
    result={}
    for key in dict.fromkeys(ORDER):
        rows=[r for r in sessions if r['artifact_key']==key]
        metrics={}
        for name in METRICS:
            values=[r['summary'].get(name) if r['status']=='completed' and r.get('summary') else None for r in rows]
            valid=len(values)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in values)
            metrics[name]={'values':values,'mean':sum(values)/2 if valid else None,'min':min(values) if valid else None,'max':max(values) if valid else None}
        result[key]={'n_scheduled':2,'n_completed':sum(r['status']=='completed' for r in rows),'metrics':metrics}
    comparisons={}
    for key in dict.fromkeys(ORDER):
        if key=='pristine':continue
        metrics={}
        for name in METRICS:
            base=result['pristine']['metrics'][name]['mean'];candidate=result[key]['metrics'][name]['mean']
            valid=matched_environment and base is not None and candidate is not None
            metrics[name]={'baseline_mean':base,'candidate_mean':candidate,
                'absolute_delta':candidate-base if valid else None,
                'relative_percent':100*(candidate-base)/base if valid and base!=0 else None}
        comparisons[key]={'replication_unit':'session','metrics':metrics}
    return result,comparisons


def verify_session_binding(session, study, row):
    manifest=read_json(session/'manifest.json')
    config=read_json(session/'baseline-config.json')
    expected=study['artifacts'][row['artifact_key']]
    arm='baseline' if row['artifact_key']=='pristine' else 'candidate'
    if not 1<=row['position']<=len(ORDER) or ORDER[row['position']-1]!=row['artifact_key']:
        raise ValueError('Session position differs from schedule')
    if (manifest['slot']!=row['key'] or row['key']!=f"{row['position']:02d}-{row['artifact_key']}"
        or manifest['study_manifest_sha256']!=canonical_hash(study)
        or manifest['artifact_sha256']!=expected['artifact_sha256'] or manifest['checkpoint']['sha256']!=expected['source_checkpoint_sha256']
        or manifest['arm']!=arm or manifest['environment']!=study['environment']
        or config['convertedArtifactSha256']!=expected['artifact_sha256']
        or config['sourceCheckpointSha256']!=expected['source_checkpoint_sha256']
        or config['runId']!=study['config']['study_id']+'-'+row['key']
        or config['servedBuild']!=study['served_build']):
        raise ValueError('Session evidence does not bind frozen study/position/artifact/environment/build')
    runtime.load(session)
    return manifest,config


def matched_environments(study, sessions, raw_summaries):
    if len(raw_summaries)!=10 or any(row['status']!='completed' for row in sessions):return False
    host=study['environment']
    browser=[s.get('environment') for s in raw_summaries]
    valid=all(e and e.get('userAgent') and e.get('platform') and e.get('hardwareConcurrency',0)>0 for e in browser)
    expected_chrome='Chrome/'+host['browser_version'].split('.')[0]+'.'
    return bool(valid and all(e==browser[0] for e in browser)
        and expected_chrome in browser[0]['userAgent']
        and str(browser[0]['hardwareConcurrency'])==host['hardware']['hw.logicalcpu']
        and all(s.get('runtime')==host['runtime'] and s.get('quantization')==host['quantization'] for s in raw_summaries)
        and len({s.get('model_lib') for s in raw_summaries})==1)


def report(root, out):
    began=time.monotonic()
    root,study,state=load(root,verify=False);sessions=[];raw_summaries=[]
    for row in state['sessions']:
        item={k:row[k] for k in ('position','artifact_key','status')};item.update(summary=None,requests=[],cleanup=None,evidence_sha256=None)
        session=root/'sessions'/row['key'];config=None
        if (session/'manifest.json').exists():_,config=verify_session_binding(session,study,row)
        target=session/'execution'/('baseline-1' if row['artifact_key']=='pristine' else 'candidate-1')
        supervisor=target/'supervisor-result.json'
        if supervisor.exists():
            value=read_json(supervisor);item['cleanup']=value['cleanup'];item['evidence_sha256']=ensure_file(supervisor)
            item['observed']={k:value.get(k) for k in ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
        summary_path=target/'resource-summary.json'
        if summary_path.exists():
            if config is None:raise ValueError('Summary has no frozen session identity')
            summary=read_json(summary_path);binding=read_json(target/'resource-summary-binding.json')
            if canonical_hash(summary)!=binding['summary_sha256']:raise ValueError('Summary binding changed')
            for path,digest in binding['input_files'].items():ensure_file(path,digest)
            if summary['model_lib']!=config['model']['modelLib'] or summary['model']!=config['model']['id']:
                raise ValueError('Reported model/library differs from frozen session')
            raw_summaries.append(summary)
            item['summary']={k:summary.get(k) for k in SUMMARY_KEYS}
            item['summary']['model_lib']=Path(urlparse(summary['model_lib']).path).name
            item['summary']['model_library_sha256']=study['artifacts'][row['artifact_key']]['files']['model.wasm']['sha256']
            n=summary['measured_requests'];ms=summary['response_ms_sum'];tokens=summary['scored_output_tokens']
            item['summary'].update(mean_response_ms=ms/n if ms is not None and n else None,
                full_response_tokens_per_second=tokens/(ms/1000) if tokens is not None and ms else None,
                response_ms_per_output_token=ms/tokens if ms is not None and tokens else None)
            browser=read_json(target/'browser-report.json');index=0
            for sample in browser['samples']:
                if not sample['warmup']:index+=1
                item['requests'].append({'request_index':0 if sample['warmup'] else index,'warmup':sample['warmup'],
                    **{dest:sample.get(source) for source,dest in (('inputTokens','input_tokens'),('outputTokens','output_tokens'),
                        ('responseMs','response_ms'),('firstOutputMs','first_output_ms'),('decodeTokensPerSecond','decode_tokens_per_second'),
                        ('finishReason','finish_reason'),('valid','valid'),('startedMs','started_ms'),('endedMs','ended_ms'))}})
            item['summary_sha256']=binding['summary_sha256'];item['served_environment']=summary.get('environment')
        if row.get('error'):item['error_type']=row['error'].split(':',1)[0]
        sessions.append(item)
    matched=matched_environments(study,sessions,raw_summaries)
    aggregates,comparisons=aggregate_sessions(sessions,matched)
    artifacts=[]
    labels={'pristine':'Pristine Qwen3.5-0.8B','v1':'V1 manual edit','v2-r1':'Astra round 1','v2-r2':'Astra round 2','v2-r3':'Astra round 3'}
    for key,a in study['artifacts'].items():
        operations=a['weight_operations']
        context={'benchmark':'IFEval development','strict_passes':16,'denominator':32,'interpretation':'Historical quality observations; not rerun in resource study.'}
        if key=='v1':context={'benchmark':'Official IFEval held-out subset','strict_passes':70,'baseline_strict_passes':69,'denominator':128,'interpretation':'Closed V1 result was inconclusive.'}
        artifacts.append({'key':key,'label':labels[key],'source_sha256':a['source_checkpoint_sha256'],'artifact_sha256':a['artifact_sha256'],
            'model_library_sha256':a['files']['model.wasm']['sha256'],'weight_operations':operations,'quality_context':context})
    observations=[r.get('observed') for r in state['sessions'] if r.get('observed')]
    actual={k:sum(r.get(k) or 0 for r in observations) for k in ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
    token_counts=[r['generation_budget']['observed']['generated_tokens'] for r in observations]
    actual.update(sampled_tokens_including_warmups=sum(token_counts) if all(x is not None for x in token_counts) else None,
        completed_sessions=sum(r['status']=='completed' for r in state['sessions']),attempted_sessions=sum('capacity_reserved' in r for r in state['sessions']),
        unresolved_attempted_sessions=sum('capacity_reserved' in r and not r.get('observed') for r in state['sessions']),
        stopped_before_reservation=sum(r['status']=='stopped' and 'capacity_reserved' not in r for r in state['sessions']),
        observation_scope='Known observed supervisor counters; unresolved attempts remain reserved and are not asserted to have zero dispatches.',
        reserved_target_tokens=sum(r.get('capacity_reserved',{}).get('target_tokens',0) for r in state['sessions']),
        charged_browser_seconds=state['charged_browser_seconds'],charged_local_seconds=state['charged_local_seconds'],
        preparation_seconds=state['preparation_seconds'],frontier_calls=0,quality_generations=0,
        generation_retries=sum(r['generation_budget']['charged']['retries'] for r in observations),
        failed_target_sequences=sum(r['generation_budget']['charged']['failed_sequences'] for r in observations))
    complete=all(r['status']=='completed' for r in state['sessions'])
    value={'schema_version':1,'kind':'existing_edits_resource_only_result','study_sha256':state['study_sha256'],
        'status':'complete' if complete else 'stopped' if state['halted'] else 'in_progress',
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':{'session_order':ORDER,'settings':resource.RESOURCE_SETTINGS,'limits':LIMITS,'cache_policy':study['config']['cache_policy'],
            'counter_definitions':COUNTERS,'workload':'Browser LLM Bench standard60 requests /3×20 /90-second gaps',
            'upstream_revision':study['resource_preset']['upstream_revision'],'derivation_sha256':canonical_hash(study['resource_preset']),
            'analysis':study['config']['analysis']},'artifacts':artifacts,'sessions':sessions,'aggregates':aggregates,'comparisons':comparisons,
        'environment':study['environment'],'matched_environment':matched,'actual_usage':actual,
        'identities':{'source_files_sha256':canonical_hash(study['files']),'settings_sha256':canonical_hash(resource.RESOURCE_SETTINGS),
            'served_build_sha256':canonical_hash(study['served_build']),'source_code':study['source_code'],
            'historical_studies_preservation':{Path(p).parent.name+'/'+Path(p).name:h for p,h in study['historical_studies_unchanged'].items()}},
        'limitations':['Two sessions per artifact; requests are not independent experimental replications.',
            'Mirrored order reduces simple linear drift, but is not randomized or fully counterbalanced.',
            'Resource observations do not change historical quality results or establish quality preservation.',
            'Full-response rates depend on output lengths; inspect token counts, truncations and per-request data.',
            'Native and allocation counters have distinct scopes; missing power/signals remain unavailable, not zero.',
            'Fresh browser caches do not clear OS filesystem caches.']}
    elapsed=time.monotonic()-began
    state['charged_local_seconds']+=elapsed
    state.setdefault('reporting_stages',[]).append({'charged_seconds':elapsed,'completed_sessions':actual['completed_sessions']})
    if state['charged_local_seconds']>LIMITS['local_seconds'] or time.time()>state['deadline_epoch_seconds']:
        state['halted']=True;value['status']='stopped'
    atomic(root/'state.json',state)
    actual['charged_local_seconds']=state['charged_local_seconds']
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);atomic(out,value)
    return {'status':value['status'],'completed_sessions':actual['completed_sessions'],'report':str(out)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report'])
    parser.add_argument('--root',default='artifacts/resource-all-edits-v1')
    parser.add_argument('--config',default='configs/resource_all_edits.json')
    parser.add_argument('--out',default='reports/resource-all-edits-v1/result.json')
    args=parser.parse_args();root=Path(args.root).resolve()
    os.environ.setdefault('WSL_NATIVE_PYTHON',sys.executable)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.command=='prepare':result=prepare(root,args.config)
    elif args.command=='status':
        _,study,state=load(root,verify=False);result={'study_sha256':state['study_sha256'],'state':state}
    elif args.command=='report':result=report(root,args.out)
    else:
        with _lock(root):
            _,_,state=load(root,verify=False)
            if state['halted'] or state['active'] is not None:raise RuntimeError('Stopped/interrupted study cannot redispatch')
            for row in state['sessions']:
                if row['status']=='completed':continue
                event=run_one(root,row['position']);print(json.dumps(event),flush=True)
                report(root,args.out)
                if not event['continue']:break
            result={'status':'run_loop_finished','report':args.out}
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
