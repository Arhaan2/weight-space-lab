"""Auditable continuation of unused positions after a pre-reservation file timeout.

The stopped original root is read-only. Existing scientific settings, runtime,
controller, observations, accounting ceilings and calendar remain unchanged.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weight_space_lab import resource_followup as base

ORIGINAL=Path('artifacts/resource-all-edits-v1').resolve()
CONTINUATION=Path('artifacts/resource-all-edits-v1-continuation').resolve()
PUBLIC=Path('reports/resource-all-edits-v1/result.json')
IMPORT_RECOVERY=Path('artifacts/resource-all-edits-v1-import-recovery.json').resolve()
FAILED_PREPARATION=Path('artifacts/resource-all-edits-v1-continuation-preparation1').resolve()
PAYLOAD_RECOVERY=Path('artifacts/resource-all-edits-v1-payload-recovery.json').resolve()
FROZEN_RECOVERY=Path('artifacts/resource-all-edits-v1-frozen-input-recovery.json').resolve()


def total_retained(root):
    # Conservative: count retained bytes in both roots, including shared hard links.
    recovery=base.read_json(IMPORT_RECOVERY)
    extra=sum(base.retained_bytes(Path(base.read_json(p)['cache_path']))+p.stat().st_size for p in (PAYLOAD_RECOVERY,FROZEN_RECOVERY))
    return base.retained_bytes(ORIGINAL)+base.retained_bytes(root)+base.retained_bytes(FAILED_PREPARATION)+base.retained_bytes(Path(recovery['overlay_path']))+IMPORT_RECOVERY.stat().st_size+extra


def ensure_available(path, digest, cached, events):
    """Repair only unavailable bytes from an independently verified exact copy."""
    path=Path(path)
    unavailable=not path.exists() or bool(path.stat().st_flags&0x40000000)
    if not unavailable:return
    try:
        base.ensure_file(path,digest)
        return
    except (FileNotFoundError,subprocess.TimeoutExpired):
        pass
    # A readable hash mismatch raises ValueError above and is never overwritten.
    base.ensure_file(cached,digest)
    temporary=path.with_name(path.name+f'.resource-restore-{os.getpid()}')
    try:
        shutil.copyfile(cached,temporary)
        base.ensure_file(temporary,digest)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    base.ensure_file(path,digest)
    with Path(events).open('a') as stream:
        stream.write(json.dumps({'path':str(path),'sha256':digest,'size_bytes':path.stat().st_size,
            'method':'Exact frozen copy restored after missing/dataless normal-read unavailability; no changed values.',
            'epoch_seconds':time.time()})+'\n')


def cache_documents(study, original_files, partial_sha, events):
    expected_payload={str(Path(a['path'])/name):v['sha256'] for a in study['artifacts'].values() for name,v in a['files'].items()}
    frozen={};summaries=[]
    expected_frozen={**study['files'],**{p:d for p,d in original_files.items() if not Path(p).is_relative_to(FAILED_PREPARATION)},str(PUBLIC.resolve()):partial_sha}
    for log,expected in ((PAYLOAD_RECOVERY,expected_payload),(FROZEN_RECOVERY,expected_frozen)):
        document=base.read_json(log)
        if document['status']!='complete' or {r['path']:r['sha256'] for r in document['files']}!=expected:
            raise ValueError('Exact byte cache does not match the frozen input allowlist')
        frozen[str(log)]=base.ensure_file(log)
        blobs={r['sha256']:r['cache_file'] for r in document['unique_payloads']}
        for row in document['unique_payloads']:
            if Path(row['cache_file']).stat().st_size!=row['size_bytes']:raise ValueError('Exact cache payload size changed')
            frozen[row['cache_file']]=base.ensure_file(row['cache_file'],row['sha256'])
        for path,digest in expected.items():
            # This old public report is snapshot-only; the live public path is
            # intentionally updated with continuation results, never restored.
            if path!=str(PUBLIC.resolve()):ensure_available(path,digest,blobs[digest],events)
        summaries.append({'log_sha256':frozen[str(log)],'files_verified':len(expected),'unique_payloads':len(blobs),'cache_payload_bytes':document['cache_payload_bytes']})
    return frozen,summaries


def original_inputs(old):
    files={str(p):base.ensure_file(p) for p in (ORIGINAL/'state.json',ORIGINAL/'study.json',ORIGINAL/'ready.json',ORIGINAL/'resource-comparison-preset.json',ORIGINAL/'final-selection.json')}
    for name in ('preparation-intent.json','preparation-result.json'):
        path=FAILED_PREPARATION/name;files[str(path)]=base.ensure_file(path)
    for row in old['sessions'][:6]:
        session=ORIGINAL/'sessions'/row['key']
        for name in ('manifest.json','ready.json','baseline-config.json','generation-budget.sqlite3','accounting-before.json','accounting-after.json','load-grant.json'):
            path=session/name
            if path.exists():files[str(path)]=base.ensure_file(path)
        arm='baseline-1' if row['artifact_key']=='pristine' else 'candidate-1'
        for name in ('supervisor-result.json','resource-summary.json','resource-summary-binding.json'):
            path=session/'execution'/arm/name;files[str(path)]=base.ensure_file(path)
        for path,digest in base.read_json(session/'execution'/arm/'resource-summary-binding.json')['input_files'].items():
            files[path]=base.ensure_file(path,digest)
    return files


def dependency_files():
    import numpy
    recovery=base.read_json(IMPORT_RECOVERY);overlay=Path(recovery['overlay_path'])
    if (recovery['status']!='complete' or recovery['versions_changed'] or recovery['installed_environment_modified']
        or recovery['version']!='2.3.5' or numpy.__version__!='2.3.5' or not Path(numpy.__file__).is_relative_to(overlay)):
        raise ValueError('Pinned import overlay identity changed')
    files={str(IMPORT_RECOVERY):base.ensure_file(IMPORT_RECOVERY)}
    for row in recovery['files']:
        path=overlay/row['relative_path']
        if path.stat().st_size!=row['size_bytes']:raise ValueError('Pinned overlay size changed')
        files[str(path)]=base.ensure_file(path,row['sha256'])
    return recovery,files


def check_original(state):
    if not state['halted'] or state['active'] is not None:raise ValueError('Original is not stopped cleanly')
    if [r['status'] for r in state['sessions']]!=['completed']*6+['stopped']+['pending']*3:
        raise ValueError('Only the observed six-complete preflight stop is eligible')
    row=state['sessions'][6]
    if not row.get('dispatch_absence_verified') or any(k in row for k in ('capacity_reserved','load_granted','observed')):
        raise ValueError('Position7 already reserved/dispatched')
    if (ORIGINAL/'sessions'/row['key']).exists():raise ValueError('Position7 session exists')
    if not row['error'].startswith('TimeoutExpired:'):raise ValueError('Unexpected original failure')
    if not all(r['observed']['cleanup']['retained_owned_pids_alive']==[] and
        all(r['observed']['cleanup'].get(k) is True for k in (*base.CLEANUP,'native_capture_exit_confirmed'))
        for r in state['sessions'][:6]):raise ValueError('Prior cleanup is unconfirmed')


def prepare(root):
    began=time.monotonic();root=Path(root).resolve()
    if root.exists():raise FileExistsError('Continuation root already exists; never replace it')
    _,study,old=base.load(ORIGINAL,verify=False);check_original(old)
    prior_public=base.read_json(PUBLIC)
    if prior_public['status']!='stopped' or prior_public['actual_usage']['completed_sessions']!=6:
        raise ValueError('Expected preserved six-session stopped public report')
    # Conservatively charge the whole pause, including offline recovery/checks,
    # rather than reset accounting or claim only a selectively measured subset.
    pause_seconds=max(0,time.time()-datetime.fromisoformat(prior_public['updated_at_utc']).timestamp())
    root.mkdir(parents=True,mode=0o700)
    original_state_sha=base.ensure_file(ORIGINAL/'state.json')
    base.save(root/'preparation-intent.json',{'original_state_sha256':original_state_sha,'started_epoch_seconds':time.time(),'target_calls':0})
    state=copy.deepcopy(old)
    try:
        remaining=min(600,base.LIMITS['local_seconds']-old['charged_local_seconds']-pause_seconds,old['deadline_epoch_seconds']-time.time())
        if remaining<=0:raise TimeoutError('No original preparation allowance remains')
        with base.bounded_python(remaining):
            original_files=original_inputs(old)
            cached_files,cache_summaries=cache_documents(study,original_files,base.ensure_file(PUBLIC),root/'restoration-events.jsonl')
            recovery,dependency=dependency_files()
            hydration=[]
            for p,digest in study['files'].items():base.ensure_file(p,digest,hydration)
            recovered=Path('artifacts/ifeval-browser-v2-collaboration-continuation/study-result.json').resolve()
            recovered_sha=base.ensure_file(recovered,study['files'][str(recovered)])
            for a in study['artifacts'].values():
                for name,value in a['files'].items():base.ensure_file(Path(a['path'])/name,value['sha256'],hydration)
            if base.resource.environment_identity()!=study['environment']:raise ValueError('Frozen environment changed')
            (root/'inherited-state.json').write_bytes((ORIGINAL/'state.json').read_bytes())
            (root/'inherited-partial-result.json').write_bytes(PUBLIC.read_bytes())
            for name in ('study.json','ready.json','resource-comparison-preset.json','final-selection.json'):
                (root/name).write_bytes((ORIGINAL/name).read_bytes())
            benchmark=base.prepare_benchmark(root,{'benchmark_source':str(ORIGINAL/'benchmark'),'baseline':study['baseline']})
            benchmark_files={}
            for p,digest in study['files'].items():
                if Path(p).is_relative_to(ORIGINAL/'benchmark'):
                    target=benchmark/Path(p).relative_to(ORIGINAL/'benchmark')
                    benchmark_files[str(target)]=base.ensure_file(target,digest,hydration)
            (root/'sessions').mkdir()
            for row in old['sessions'][:6]:(root/'sessions'/row['key']).symlink_to(ORIGINAL/'sessions'/row['key'],target_is_directory=True)
            failed_preparation=base.read_json(FAILED_PREPARATION/'preparation-result.json')
            if failed_preparation['status']!='failed' or failed_preparation['target_calls']!=0:
                raise ValueError('Unexpected prior no-load preparation result')
            if original_files[str(ORIGINAL/'state.json')]!=original_state_sha:raise ValueError('Original changed during preparation')
            state['sessions'][6]={'position':7,'artifact_key':'v2-r2','key':'07-v2-r2','status':'pending',
                'inherited_preflight_failure':copy.deepcopy(old['sessions'][6]),'dispatch_absence_verified_at_continuation':True}
            state['halted']=False
            frozen=time.time()
            manifest={'schema_version':1,'kind':'resource_pre_dispatch_continuation','scientific_study_sha256':old['study_sha256'],
                'original_state_sha256':original_state_sha,'inherited_partial_result_sha256':base.ensure_file(root/'inherited-partial-result.json'),
                'original_files':original_files,'benchmark_files':benchmark_files,
                'driver_sha256':base.ensure_file(Path(__file__).resolve()),'driver_path':str(Path(__file__).resolve()),
                'dependency_files':dependency,'import_recovery':{k:recovery[k] for k in ('kind','distribution','version','archive_sha256','installed_record_sha256','payload_files','installed_record_matches','overlay_payload_bytes','versions_changed','installed_environment_modified')},
                'cached_files':cached_files,'cache_summaries':cache_summaries,
                'original_deadline_epoch_seconds':old['deadline_epoch_seconds'],'limits':base.LIMITS,
                'inherited_completed_positions':list(range(1,7)),'remaining_positions':[7,8,9,10],
                'original_failure':old['sessions'][6],'original_verification_failure':old['verification_stages'][-1],
                'inherited_charged_local_seconds':old['charged_local_seconds'],'inherited_charged_browser_seconds':old['charged_browser_seconds'],
                'pause_local_seconds_conservatively_charged':pause_seconds,
                'inherited_reserved_target_tokens':sum(r.get('capacity_reserved',{}).get('target_tokens',0) for r in old['sessions']),
                'preserved_no_load_preparation_failures':[{'status':'failed','target_calls':0,'elapsed_seconds':failed_preparation['charged_seconds'],
                    'error_type':failed_preparation['error'].split(':',1)[0],'result_sha256':original_files[str(FAILED_PREPARATION/'preparation-result.json')],
                    'accounting':'Included in the conservative pause charge; not added twice.'}],
                'recovery':{'file_role':'Protected historical V2 terminal result','sha256':recovered_sha,'size_bytes':recovered.stat().st_size,
                    'method':'Ordinary filesystem materialization completed after bounded read timeout; exact frozen bytes verified; no replacement or reconstruction.'},
                'preparation_materializations':hydration,'created_epoch_seconds':frozen,
                'authorization':'Explicit completion of unused positions7–10, with original ceilings/calendar and six observations retained. No model-call retry or replacement.',
                'interruption_note':'A filesystem integrity-preflight pause separates completed position6 from unused position7; this is not an uninterrupted run. The slow completed R3 observation remains included.'}
            elapsed=time.monotonic()-began;state['charged_local_seconds']+=pause_seconds+elapsed
            state['continuation_preparation_seconds']=elapsed
            state['continuation_identity']=base.canonical_hash(manifest)
            if total_retained(root)>base.LIMITS['retained_bytes']:raise RuntimeError('Combined retained bytes exceed original ceiling')
            base.save(root/'continuation.json',manifest)
            base.save(root/'continuation-ready.json',{'continuation_sha256':state['continuation_identity']})
            base.atomic(root/'state.json',state)
            base.save(root/'preparation-result.json',{'status':'complete','charged_seconds':elapsed,'target_calls':0})
    except BaseException as exc:
        elapsed=time.monotonic()-began
        base.save(root/'preparation-result.json',{'status':'failed','charged_seconds':elapsed,'target_calls':0,'error':type(exc).__name__+': '+str(exc)})
        raise
    return {'continuation_sha256':state['continuation_identity'],'remaining_positions':[7,8,9,10],
        'inherited_browser_seconds':old['charged_browser_seconds'],'inherited_local_seconds':old['charged_local_seconds'],
        'preparation_seconds':elapsed,'deadline_unchanged':True,'target_calls':0}


def verify(root):
    manifest=base.read_json(root/'continuation.json');_,study,state=base.load(root,verify=False)
    digest=base.canonical_hash(manifest)
    if base.read_json(root/'continuation-ready.json')!={'continuation_sha256':digest} or state.get('continuation_identity')!=digest:
        raise ValueError('Continuation binding changed')
    if manifest['scientific_study_sha256']!=state['study_sha256'] or manifest['limits']!=base.LIMITS or manifest['original_deadline_epoch_seconds']!=state['deadline_epoch_seconds']:
        raise ValueError('Original scientific identity/ceilings/calendar changed')
    base.ensure_file(manifest['driver_path'],manifest['driver_sha256'])
    for p,d in manifest['cached_files'].items():base.ensure_file(p,d)
    cache_documents(study,manifest['original_files'],manifest['inherited_partial_result_sha256'],root/'restoration-events.jsonl')
    for p,d in {**manifest['original_files'],**manifest['benchmark_files'],**manifest['dependency_files']}.items():base.ensure_file(p,d)
    base.ensure_file(root/'inherited-partial-result.json',manifest['inherited_partial_result_sha256'])
    check_original(base.read_json(ORIGINAL/'state.json'))
    if total_retained(root)>base.LIMITS['retained_bytes']:raise RuntimeError('Combined retained bytes exceed original ceiling')
    return manifest,state


def export(root):
    result=base.report(root,PUBLIC);began=time.monotonic()
    value=base.read_json(PUBLIC);m=base.read_json(root/'continuation.json');state=base.read_json(root/'state.json')
    seventh=state['sessions'][6]
    prior_end=state['sessions'][5]['started_epoch_seconds']+state['sessions'][5]['charged_seconds']
    gap=seventh.get('started_epoch_seconds')
    value['continuation']={'identity':base.canonical_hash(m),'original_state_sha256':m['original_state_sha256'],
        'preserved_partial_result_sha256':m['inherited_partial_result_sha256'],'scientific_study_sha256':m['scientific_study_sha256'],
        'inherited_completed_positions':m['inherited_completed_positions'],'remaining_positions':m['remaining_positions'],
        'inherited_charged_browser_seconds':m['inherited_charged_browser_seconds'],'inherited_charged_local_seconds':m['inherited_charged_local_seconds'],
        'pause_local_seconds_conservatively_charged':m['pause_local_seconds_conservatively_charged'],
        'original_calendar_and_total_limits_unchanged':True,'preserved_failed_preflight_count':1,
        'failure_stage':'Integrity verification before position7 reservation; no session7 directory, load or generation existed.',
        'failure_type':'TimeoutExpired','recovery':m['recovery'],'driver_sha256':m['driver_sha256'],
        'dependency_recovery':m['import_recovery'],
        'verified_exact_byte_caches':m['cache_summaries'],
        'preserved_no_load_preparation_failures':m['preserved_no_load_preparation_failures'],
        'inter_session_gap_after_position6_seconds':gap-prior_end if gap is not None else None,
        'interruption_note':m['interruption_note'],'replaced_completed_sessions':0,'model_dispatch_retries':0}
    value['matched_current_environment']=value['matched_environment'];value['uninterrupted']=False
    value['actual_usage']['preserved_failed_pre_dispatch_verifications']=1
    value['actual_usage']['stopped_before_reservation']+=1
    value['actual_usage']['offline_preparation_failures']=len(m['preserved_no_load_preparation_failures'])
    value['continuation']['post_cache_exact_restorations']=len((root/'restoration-events.jsonl').read_text().splitlines()) if (root/'restoration-events.jsonl').exists() else 0
    value['actual_usage']['continuation_preparation_seconds']=state['continuation_preparation_seconds']
    value['actual_usage']['aggregate_retained_bytes_conservative']=total_retained(root)
    value['limitations'].extend([m['interruption_note'],
        'Matched recorded device/browser/runtime identity does not establish constant background host load or uninterrupted conditions.',
        'Lower CPU/GPU activity rates do not establish lower total compute or energy, particularly when response time rises.'])
    state['charged_local_seconds']+=time.monotonic()-began
    if state['charged_local_seconds']>base.LIMITS['local_seconds'] or time.time()>state['deadline_epoch_seconds']:
        state['halted']=True;value['status']='stopped'
    value['actual_usage']['charged_local_seconds']=state['charged_local_seconds']
    base.atomic(root/'state.json',state)
    base.atomic(PUBLIC,value)
    return result


def run(root):
    with base._lock(root):
        _,_,state=base.load(root,verify=False)
        if state['halted'] or state['active'] is not None:raise RuntimeError('Stopped/interrupted continuation cannot redispatch')
        for position in (7,8,9,10):
            _,_,state=base.load(root,verify=False)
            if state['halted'] or state['active'] is not None:break
            if state['sessions'][position-1]['status']=='completed':continue
            started=time.monotonic()
            try:
                allowance=min(300,base.LIMITS['local_seconds']-state['charged_local_seconds'],state['deadline_epoch_seconds']-time.time())
                if allowance<=0:raise TimeoutError('Original verification allowance exhausted')
                with base.bounded_python(allowance):verify(root)
            except BaseException as exc:
                state['charged_local_seconds']+=time.monotonic()-started;state['halted']=True
                state['continuation_guard_failure']={'position':position,'error':type(exc).__name__+': '+str(exc)}
                base.atomic(root/'state.json',state);export(root);raise
            state['charged_local_seconds']+=time.monotonic()-started;base.atomic(root/'state.json',state)
            event=base.run_one(root,position);print(json.dumps(event),flush=True);export(root)
            if not event['continue']:break
    return {'status':'continuation_loop_finished','report':str(PUBLIC)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','report'])
    args=parser.parse_args();os.environ.setdefault('WSL_NATIVE_PYTHON',sys.executable);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    result=prepare(CONTINUATION) if args.command=='prepare' else run(CONTINUATION) if args.command=='run' else export(CONTINUATION)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
