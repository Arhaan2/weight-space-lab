"""Fresh, separately bounded six-artifact comparison. No old ledger is reopened."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from . import browser_resource as resource, ifeval_v2_runtime as runtime
from . import resource_followup as historical
from .browser_check import _collector
from .browser_candidate import _files
from .grading import grader_identity, verify_ifeval_assets
from .ifeval_baseline import SETTINGS, POLICY, save
from .ifeval_v2_study import atomic, bounded_python, retained_bytes, CLEANUP
from .inventory import read_json, file_sha256
from .native_runner import _lock
from .protocol import canonical_hash

ID='fresh-baseline-r4-v1'
KEYS=['pristine','v1','v2-r1','v2-r2','v2-r3','astra-r4']
ORDER=KEYS+list(reversed(KEYS))
LABELS={'pristine':'Pristine','v1':'V1 edit','v2-r1':'Astra round 1','v2-r2':'Astra round 2','v2-r3':'Astra round 3','astra-r4':'Astra round 4'}
LIMITS={'engine_loads':18,'warmups':18,'quality_requests':192,'resource_requests':720,'dispatches':930,
 'reserved_target_tokens':242976,'browser_seconds':10800,'local_seconds':14400,'calendar_seconds':86400,
 'retained_bytes':34359738368,'minimum_free_bytes':8589934592,'editor_proposals':1,'imports':1,
 'builds':1,'conversions':1,'packages':1,'automatic_retries':0}
SOURCE_ROOT=Path(__file__).resolve().parents[1]

def expected_rows():
    return ([{'key':f'quality-{i+1:02d}-{key}','artifact_key':key,'kind':'quality','position':i+1,'status':'pending'} for i,key in enumerate(KEYS)]
        +[{'key':f'resource-{i+1:02d}-{key}','artifact_key':key,'kind':'resource','position':i+1,'status':'pending'} for i,key in enumerate(ORDER)])

def validate_config(config):
    if (config['study_id']!=ID or config['quality_order']!=KEYS or config['resource_order']!=ORDER or
        config['limits']!=LIMITS or config['quality_settings']!=SETTINGS or config['resource_settings']!=resource.RESOURCE_SETTINGS):
        raise ValueError('Frozen schedule/settings/limits differ')

def total_retained(root,study):
    roots=[Path(p).resolve() for p in study.get('retained_roots',[str(root)])]
    if any(a!=b and a.is_relative_to(b) for a in roots for b in roots):raise ValueError('Retained roots must be disjoint')
    return sum(retained_bytes(p) for p in roots)

def available(root,study,state):
    if state['halted'] or state['active'] is not None or state.get('reporting_active'):raise RuntimeError('Stopped or unresolved study; no automatic retry')
    if time.time()>=study['deadline_epoch_seconds']:raise TimeoutError('Calendar exhausted')
    if state['charged_local_seconds']>=LIMITS['local_seconds'] or state['charged_browser_seconds']>=LIMITS['browser_seconds']:raise TimeoutError('Time budget exhausted')
    if shutil.disk_usage(root).free<LIMITS['minimum_free_bytes'] or total_retained(root,study)>LIMITS['retained_bytes']:raise RuntimeError('Storage limit')

def reserve(state,key,seconds,now):
    if state['halted'] or state['active'] is not None or state.get('reporting_active'):raise RuntimeError('No redispatch after halt/unresolved attempt')
    row=next(r for r in state['sessions'] if r['key']==key)
    index=state['sessions'].index(row)
    if row['status']!='pending' or any(r['status']!='completed' for r in state['sessions'][:index]):raise ValueError('Fixed predecessor order or single attempt violated')
    n,cap=(32,1024) if row['kind']=='quality' else (60,64)
    if seconds<max(300,cap/10):raise TimeoutError('Insufficient request/cleanup allowance')
    row.update(status='running',reserved_seconds=seconds,started_epoch_seconds=now,load_granted=False,
        capacity_reserved={'engine_loads':1,'warmups':1,'measured_requests':n,'dispatches':n+1,'target_tokens':n*cap+16})
    state['active']={'key':key,'pid':os.getpid(),'started_epoch_seconds':now,'reserved_seconds':seconds}
    return row

def verify_files(files):
    for path,digest in files.items():
        if file_sha256(path)!=digest:raise ValueError('Frozen bytes changed: '+Path(path).name)

def verify_converter_interpreter(build):
    binding=build['converter_interpreter_binding'];selector=Path(build['converter_python'])
    if (str(selector)!=binding['selector'] or not selector.is_symlink() or os.readlink(selector)!=binding['symlink_target'] or
        str(selector.resolve())!=binding['resolved_executable'] or file_sha256(selector.resolve())!=binding['resolved_executable_sha256'] or
        file_sha256(binding['pyvenv_config'])!=binding['pyvenv_config_sha256']):
        raise ValueError('Frozen converter interpreter selector, binary or venv metadata changed')

def preparation_summary(root,staged,build,elapsed):
    receipt=read_json(Path(build['source_model']).parent/'converter-staging.json')
    failures=[]
    for name,stage in [('grader-preflight-preparation-failure.json','Tokenizer asset staging'),('freeze-preparation-failure-01.json','Interpreter selector binding')]:
        p=root/name;item=read_json(p)
        failures.append({'stage':stage,'target_calls':item['target_calls'],'receipt_sha256':file_sha256(p),
            'charged_seconds':item['charged_conservative_seconds']})
    return {'target_calls':0,'build_preflight_sha256':file_sha256(root/'build-preflight.json'),
        'converter_staging_sha256':file_sha256(Path(build['source_model']).parent/'converter-staging.json'),
        'source_file_count':build['source_file_count'],'converter_distribution_count':len(receipt['packages']),
        'unavailable_original_installed_records':sum(p['installed_record_status']!='read' for p in receipt['packages']),
        'dependency_verification_counts':receipt['counts'],'signature_transformations':build['signature_repair_summary'],
        'compiled_parameter_metadata':build['metadata_summary'],'failed_preparation_stages':failures,
        'charged_staging_seconds':staged['charged_preparation_seconds'],'charged_build_preparation_seconds':build['charged_preparation_seconds'],
        'charged_successful_freeze_seconds':elapsed,'full_old_dependency_byte_equivalence_claimed':False,
        'notes':['All preparation used zero target-model calls. Exact source bytes and deployed artifacts were verified before the first model load.',
            'The converter was reconstructed from 75 same-version cached wheels. Wheel RECORD digests and available installed RECORD digests were checked; 27 historical installed RECORDs were unavailable, so full old-environment byte equality is not claimed.',
            'Ten new local library copies received signature-only ad-hoc signing, with unchanged code/data payloads verified; three readable historical signed libraries matched byte for byte.',
            'The 435-parameter metadata was extracted from the unchanged, verified compiled WASM; equality to the unavailable historical JSON bytes is not claimed.',
            'Preparation rejected 73 extra language files in the new tokenizer staging and a converter executable symlink. The English-only allowlist and explicit interpreter selector/binary/config binding were corrected before freezing; original inputs were preserved.',
            'Earlier console-script and signature-metadata preparation failures were preserved. All failed preparation and accounting amendments are included in conservative local elapsed-time charges.']}

def load(root,verify=False):
    root=Path(root).resolve();study=read_json(root/'study.json');state=read_json(root/'state.json')
    if read_json(root/'ready.json')!={'study_sha256':canonical_hash(study)} or state['study_sha256']!=canonical_hash(study):raise ValueError('Study binding changed')
    validate_config(study['config'])
    expected=expected_rows()
    if [(r['key'],r['artifact_key'],r['kind']) for r in state['sessions']]!=[(r['key'],r['artifact_key'],r['kind']) for r in expected]:raise ValueError('Schedule changed')
    if verify:
        verify_files(study['files'])
        verify_files({str(SOURCE_ROOT/k):v for k,v in study['source_files'].items()})
        verify_converter_interpreter(read_json(study['build_preflight_path']))
        if resource.environment_identity()!=study['environment']:raise ValueError('Device/browser/runtime changed')
        if study['python_identity']!={'version':sys.version,'executable_sha256':file_sha256(Path(sys.executable).resolve())}:raise ValueError('Controller Python identity changed')
        if (root/'candidate-binding.json').exists():
            binding=read_json(root/'candidate-binding.json')
            if binding['parent_study_sha256']!=canonical_hash(study) or state.get('candidate_binding_sha256')!=canonical_hash(binding):raise ValueError('Candidate binding changed')
            verify_files(binding['files'])
    return root,study,state

def prepare(root,config_path,build_preflight):
    root=Path(root).resolve();began=time.monotonic()
    if (root/'study.json').exists() or (root/'state.json').exists():raise FileExistsError('Study already frozen')
    staged=read_json(root/'staged-inputs.json');config=read_json(config_path);validate_config(config)
    verify_files(staged['files'])
    grader_check=read_json(root/'grader-preflight.json')
    if grader_check.get('status')!='passed' or grader_check.get('target_calls')!=0 or len(grader_check.get('cases',[]))!=2:raise ValueError('Require no-model known-answer grader preflight')
    build=read_json(build_preflight)
    if build.get('status')!='ready' or build.get('target_calls',0)!=0:raise ValueError('Require completed no-inference build readiness')
    verify_files(build['files'])
    verify_converter_interpreter(build)
    if set(staged['artifacts'])!=set(KEYS[:-1]):raise ValueError('Five original artifacts required')
    for key,a in staged['artifacts'].items():
        if canonical_hash(a['files'])!=a['artifact_sha256'] or _files(a['path'])!=a['files']:raise ValueError('Artifact differs: '+key)
    if verify_ifeval_assets(staged['baseline']['official_ifeval_root'])!=staged['baseline']['official_assets']:raise ValueError('Grader assets differ')
    env=resource.environment_identity();collector=_collector(root/'benchmark');signals,unavailable=collector.host_signals()
    if signals.get('hostMemoryPressureLevel') is None or signals.get('hostThermalState') is None or signals['hostMemoryPressureLevel']>=4 or signals['hostThermalState']>=2:raise RuntimeError('Host resource guard')
    save(root/'native-preflight.json',{'environment':env,'host_signals':signals,'unavailable':unavailable,'target_calls':0})
    sources={str(p.relative_to(SOURCE_ROOT)):file_sha256(p) for p in (SOURCE_ROOT/'weight_space_lab').glob('*.py')}
    for p in (SOURCE_ROOT/'scripts/stage_fresh_baseline_inputs.py',SOURCE_ROOT/'scripts/build_fresh_astra_packet.py',SOURCE_ROOT/'scripts/browser_convert.py',SOURCE_ROOT/'configs/fresh_baseline_r4.json',SOURCE_ROOT/'docs/fresh-baseline-r4-protocol.md',SOURCE_ROOT/'docs/astra-round4-editor.md',SOURCE_ROOT/'configs/grading-requirements.txt'):
        sources[str(p.relative_to(SOURCE_ROOT))]=file_sha256(p)
    files={**staged['files'],**build['files'],str(Path(build_preflight).resolve()):file_sha256(build_preflight)}
    for name in ('staged-inputs.json','python-dependencies.json','native-preflight.json','grader-preflight.json'):
        files[str(root/name)]=file_sha256(root/name)
    now=time.time();elapsed=time.monotonic()-began
    study={'schema_version':1,'kind':'fresh_quality_and_resource_comparison','config':config,'baseline':staged['baseline'],
        'artifacts':staged['artifacts'],'benchmark_source':str(root/'benchmark'),'environment':env,'files':files,'source_files':sources,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE_ROOT,text=True).strip(),
        'python_identity':{'version':sys.version,'executable_sha256':file_sha256(Path(sys.executable).resolve())},
        'served_build':resource.served_build(root/'benchmark','http://127.0.0.1:4198'),
        'resource_preset':resource.inspect_preset(root/'benchmark'),'created_epoch_seconds':now,'deadline_epoch_seconds':now+LIMITS['calendar_seconds'],
        'build_preflight_sha256':file_sha256(build_preflight),'build_preflight_path':str(Path(build_preflight).resolve()),
        'retained_roots':[str(root),str(Path(build['source_model']).resolve().parent)],
        'authorization':'User requested a new Astra round, fresh baseline, quality/resource comparisons and publication; no old capacity reused.',
        'quality_exposure':'All 32 prompts were previously exposed development data; no untouched final confirmation.',
        'candidate_binding_policy':'Exactly one new independent proposal and artifact bound after five fresh quality sessions, before its own evaluation; no parent-study mutation.',
        'preparation':preparation_summary(root,staged,build,elapsed)}
    save(root/'study.json',study);digest=canonical_hash(study);save(root/'ready.json',{'study_sha256':digest})
    charged=staged['charged_preparation_seconds']+float(build.get('charged_preparation_seconds',0))+grader_check['charged_seconds']+elapsed
    state={'study_sha256':digest,'halted':False,'active':None,'sessions':expected_rows(),'charged_local_seconds':charged,
        'charged_browser_seconds':0,'stages':[{'kind':'preparation','status':'completed','charged_seconds':charged}],
        'candidate_binding_sha256':None,'editor_proposals_consumed':0,'build_consumed':0,'action_attempts':{}}
    atomic(root/'state.json',state);available(root,study,state)
    return {'status':'ready','study_sha256':digest,'quality_order':KEYS,'resource_order':ORDER,'target_calls':0,'local_seconds_used':charged}

def artifact_for(root,study,key):
    if key in study['artifacts']:return study['artifacts'][key]
    return read_json(root/'candidate-binding.json')['artifact']

def begin_action(root,study,state,kind,maximum,reserve_bytes=0):
    available(root,study,state)
    if kind in state['action_attempts']:raise ValueError('Action already attempted; no retry: '+kind)
    if shutil.disk_usage(root).free-LIMITS['minimum_free_bytes']<reserve_bytes:raise RuntimeError('Insufficient reserved action storage')
    seconds=min(maximum,LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())
    if seconds<30:raise TimeoutError('Insufficient action allowance')
    intent={'kind':kind,'status':'running','started_epoch_seconds':time.time(),'reserved_seconds':seconds,'reserve_bytes':reserve_bytes,'pid':os.getpid()}
    state['action_attempts'][kind]=intent;state['active']={'key':kind,**intent}
    if kind=='editor':state['editor_proposals_consumed']=1
    if kind=='checkpoint_build':state['build_consumed']=1
    atomic(root/'state.json',state)
    return seconds

def charged_load(root,label):
    began=time.monotonic();root,study,state=load(root);available(root,study,state);error=None
    try:
        with bounded_python(min(600,LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())):load(root,verify=True)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    elapsed=time.monotonic()-began;state['charged_local_seconds']+=elapsed
    state['stages'].append({'kind':'verification','key':label,'charged_seconds':elapsed,'status':'failed' if error else 'completed'})
    if error:state['halted']=True
    atomic(root/'state.json',state)
    if error:raise RuntimeError(error)
    return root,study,state

def finish_action(root,state,kind,elapsed,error=None):
    if not state['active'] or state['active']['key']!=kind:raise ValueError('Mismatched active action')
    row=state['action_attempts'][kind]
    if elapsed>row['reserved_seconds']:error=error or 'TimeoutError: action exceeded its reservation'
    row.update(status='failed' if error else 'completed',charged_seconds=elapsed,error=error)
    state['charged_local_seconds']+=elapsed;state['stages'].append(dict(row));state['active']=None
    if error:state['halted']=True
    atomic(root/'state.json',state)
    if error:raise RuntimeError(error)

def packet_files(packet,study):
    packet=Path(packet).resolve();request=read_json(packet/'request.json');ready=read_json(packet/'ready.json')
    files={str(packet/name):file_sha256(packet/name) for name in ('request.json','ready.json','editor-packet.txt')}
    if (request.get('study_id')!=ID or request.get('study_sha256')!=canonical_hash(study) or
        request.get('submission_expires_at_epoch_seconds')!=study['deadline_epoch_seconds'] or
        ready.get('request_sha256')!=canonical_hash(request) or ready.get('request_file_sha256')!=files[str(packet/'request.json')] or
        ready.get('packet_sha256')!=files[str(packet/'editor-packet.txt')]):raise ValueError('Editor packet binding differs')
    chunks=[]
    for index,part in enumerate(ready['parts'],1):
        if part['file']!=f'part-{index:03d}.txt':raise ValueError('Unordered or unsafe packet part')
        path=packet/part['file'];payload=path.read_bytes()
        if not 0<len(payload)<=12000 or len(payload)!=part['bytes'] or hashlib.sha256(payload).hexdigest()!=part['sha256']:raise ValueError('Packet part changed')
        files[str(path)]=part['sha256'];chunks.append(payload)
    if b''.join(chunks)!=(packet/'editor-packet.txt').read_bytes() or sum(map(len,chunks))!=ready['packet_bytes']:raise ValueError('Part delivery differs from exact full packet')
    return request,ready,files

def begin_editor(root,packet):
    root,study,state=charged_load(root,'before_editor')
    if any(r['status']!='completed' for r in state['sessions'][:5]) or state['sessions'][5]['status']!='pending':raise ValueError('Require exactly the five fresh preceding quality sessions')
    packet=Path(packet).resolve();request,ready,files=packet_files(packet,study)
    if state.get('packet_preparation_charged'):raise ValueError('Editor packet already admitted')
    state['charged_local_seconds']+=ready['charged_export_seconds'];state['packet_preparation_charged']=True
    state['stages'].append({'kind':'editor_packet_export','charged_seconds':ready['charged_export_seconds'],'status':'completed'});atomic(root/'state.json',state)
    seconds=begin_action(root,study,state,'editor',3600)
    state['editor_packet']={'path':str(packet),'request_sha256':canonical_hash(request),'packet_sha256':ready['packet_sha256'],'files':files}
    atomic(root/'state.json',state)
    save(root/'editor-turn-intent.json',{'study_sha256':state['study_sha256'],**state['editor_packet'],'reserved_seconds':seconds,
        'route':'one actual fresh-context gpt-6-astra collaboration agent','model_identity_and_provider_usage':'selector observed; immutable version, accepted effort and billing unavailable',
        'target_calls':0,'proposal_slot_consumed_before_dispatch':True})
    return {'status':'editor_slot_reserved','reserved_seconds':seconds,'request_sha256':canonical_hash(request)}

def import_response(root,response_path,rationale_path=None):
    from .ifeval_handoff import validate_response,MAX_RESPONSE_BYTES
    from .editor_contract import read_json_bounded
    from .ifeval_v2_controls import validate_v2_proposal
    root,study,state=load(root)
    if not state['active'] or state['active']['key']!='editor':raise ValueError('No reserved editor turn')
    # The supplied response closes the already-consumed turn; an invalid response cannot reopen it.
    elapsed=time.time()-state['action_attempts']['editor']['started_epoch_seconds']
    finish_action(root,state,'editor',elapsed)
    seconds=begin_action(root,study,state,'response_import',300);began=time.monotonic();error=None
    try:
        with bounded_python(seconds):
            source=Path(response_path)
            with source.open('rb') as stream:raw=stream.read(MAX_RESPONSE_BYTES+1)
            packet=Path(state['editor_packet']['path']);original=packet/'original-response.json'
            with original.open('xb') as stream:stream.write(raw)
            if len(raw)>MAX_RESPONSE_BYTES:raise ValueError('Response exceeded frozen byte bound')
            request=read_json(packet/'request.json')
            _,_,files=packet_files(packet,study)
            if files!=state['editor_packet']['files']:raise ValueError('Delivered packet inputs changed')
            if canonical_hash(request)!=state['editor_packet']['request_sha256'] or file_sha256(packet/'editor-packet.txt')!=state['editor_packet']['packet_sha256']:raise ValueError('Packet changed after dispatch')
            proposal=validate_v2_proposal(validate_response(read_json_bounded(original,max_bytes=MAX_RESPONSE_BYTES),request),study['baseline']['source_sha256'])
            rationale_path=Path(rationale_path) if rationale_path else source.with_name('rationale.txt')
            with rationale_path.open('rb') as stream:rationale_bytes=stream.read(12001)
            rationale=rationale_bytes.decode('utf-8').strip()
            if len(rationale_bytes)>12000 or not rationale or len(rationale.split())>150 or '\x00' in rationale:raise ValueError('Require a bounded pre-evaluation rationale of at most 150 words')
            candidate=root/'candidate';candidate.mkdir(mode=0o700)
            save(candidate/'proposal.json',proposal);state['proposal_sha256']=canonical_hash(proposal)
            (candidate/'editor-rationale.txt').write_bytes(rationale_bytes)
            state['pre_evaluation_rationale_sha256']=file_sha256(candidate/'editor-rationale.txt')
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finish_action(root,state,'response_import',time.monotonic()-began,error)
    return {'accepted':True,'proposal_sha256':state['proposal_sha256']}

def build_candidate(root,editor_provenance_path):
    from .editor_tools import run_checkpoint_action
    from .ifeval_v2_candidate import convert,package_candidate
    from .ifeval_v2_controls import validate_v2_proposal
    root,study,state=charged_load(root,'before_candidate_build');available(root,study,state)
    provenance=read_json(editor_provenance_path)
    if provenance.get('requested_model')!='gpt-6-astra' or provenance.get('actual_collaboration_agent') is not True:raise ValueError('Require recorded actual Astra collaboration provenance before build')
    if state.get('action_attempts',{}).get('response_import',{}).get('status')!='completed':raise ValueError('Import a single bound response first')
    inputs=read_json(study['build_preflight_path']);target=root/'candidate';policy=Path(study['baseline']['baseline_root'])/'weight-information/policy.json'
    proposal=read_json(target/'proposal.json')
    if canonical_hash(proposal)!=state['proposal_sha256']:raise ValueError('Proposal changed')
    if file_sha256(target/'editor-rationale.txt')!=state['pre_evaluation_rationale_sha256']:raise ValueError('Pre-evaluation rationale changed')
    validate_v2_proposal(proposal,study['baseline']['source_sha256'])
    audit=None;artifact=None
    def action(kind,maximum,reserve_bytes,function):
        seconds=begin_action(root,study,state,kind,maximum,reserve_bytes);began=time.monotonic();error=None;value=None
        try:
            with bounded_python(seconds):value=function(seconds)
        except BaseException as exc:error=type(exc).__name__+': '+str(exc)
        finish_action(root,state,kind,time.monotonic()-began,error)
        return value
    def checkpoint_build(seconds):
        audit=run_checkpoint_action(model=inputs['source_model'],policy=read_json(policy),exposure_log=target/'exposures.jsonl',output=target/'source',
            tool='propose',arguments={'proposal':proposal},timeout_seconds=max(.001,seconds-2),check_active=lambda:None,expected_original_sha256=study['baseline']['source_sha256'])
        full=read_json(target/'source/audit.json')
        if not full['complete'] or full['artifact_verification']!='passed':raise ValueError('Full saved source audit failed')
        return audit,full
    audit,full_audit=action('checkpoint_build',900,2*1024**3,checkpoint_build)
    save(target/'source-audit.json',audit)
    def convert_once(seconds):
        deadline=time.monotonic()+seconds
        return convert(inputs['source_model'],target/'source',inputs['conversion_manifest'],target/'converted',inputs['converter_python'],
            policy_path=policy,precheck_timeout=min(30,seconds-10),conversion_timeout=min(1800,seconds),on_conversion=lambda:max(.001,min(1800,deadline-time.monotonic()-2)))
    action('conversion',1800,1024**3,convert_once)
    artifact=action('package',600,1024**3,lambda seconds:package_candidate(inputs['source_model'],target/'source',inputs['conversion_manifest'],target/'converted',target/'deployed',policy_path=policy))
    if artifact.get('conversion_erases_edit'):
        state['halted']=True;atomic(root/'state.json',state);raise ValueError('Conversion erased edit; capacity spent')
    provenance.update(proposals=1,target_generations=0,weight_operations=proposal['operations'],proposal_sha256=state['proposal_sha256'],
        request_sha256=state['editor_packet']['request_sha256'],packet_sha256=state['editor_packet']['packet_sha256'],
        charged_local_seconds=sum(r.get('charged_seconds',0) for r in state['action_attempts'].values()),
        source_audit_sha256=canonical_hash(full_audit),new_proposal_from_pristine=True,
        pre_evaluation_rationale=(target/'editor-rationale.txt').read_text().strip(),pre_evaluation_rationale_sha256=state['pre_evaluation_rationale_sha256'],
        source_edit={k:full_audit[k] for k in ('result_sha256','changed_stored_entries','changed_numerical_entries','targeted_coordinates','unchanged_tensor_count','artifact_verification','complete')},
        source_tensor_contract={op['tensor']:{k:full_audit['model_inventory']['tensors'][op['tensor']][k] for k in ('shape','dtype')} for op in proposal['operations']})
    save(target/'editor-provenance.json',provenance)
    return bind_candidate(root,artifact['artifact_manifest'],target/'editor-provenance.json')

def bind_candidate(root,artifact_path,provenance_path):
    root,study,state=charged_load(root,'candidate_binding')
    available(root,study,state)
    if (root/'candidate-binding.json').exists() or state['candidate_binding_sha256']:raise ValueError('Candidate already bound')
    if any(state.get('action_attempts',{}).get(k,{}).get('status')!='completed' for k in ('editor','response_import','checkpoint_build','conversion','package')):raise ValueError('All single-use candidate actions must be durably completed')
    if [r['status'] for r in state['sessions'][:5]]!=['completed']*5 or state['sessions'][5]['status']!='pending':raise ValueError('Five fresh quality observations required')
    a=read_json(artifact_path);provenance=read_json(provenance_path)
    if canonical_hash(a['files'])!=a['artifact_sha256'] or _files(a['path'])!=a['files']:raise ValueError('Candidate deployment differs')
    if provenance.get('target_generations',0)!=0 or provenance.get('proposals')!=1:raise ValueError('Require one actual Astra proposal, no hidden evaluation')
    base=study['artifacts']['pristine']
    for name in ('model.wasm','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','mlc-chat-config.json'):
        if a['files'][name]!=base['files'][name]:raise ValueError('Deployment controls differ')
    if a['artifact_sha256'] in [x['artifact_sha256'] for x in study['artifacts'].values()]:raise ValueError('R4 must be distinct from existing artifacts')
    a['weight_operations']=provenance['weight_operations']
    a['source_edit']=provenance['source_edit'];a['source_tensor_contract']=provenance['source_tensor_contract']
    files={str(Path(a['path'])/name):v['sha256'] for name,v in a['files'].items()}
    files.update({str(Path(artifact_path).resolve()):file_sha256(artifact_path),str(Path(provenance_path).resolve()):file_sha256(provenance_path)})
    files[str(root/'candidate/editor-rationale.txt')]=state['pre_evaluation_rationale_sha256']
    binding={'schema_version':1,'parent_study_sha256':state['study_sha256'],'artifact_key':'astra-r4','artifact':a,
        'editor_provenance':provenance,'files':files,'bound_epoch_seconds':time.time()}
    save(root/'candidate-binding.json',binding);state['candidate_binding_sha256']=canonical_hash(binding)
    atomic(root/'state.json',state);return {'candidate_binding_sha256':state['candidate_binding_sha256'],'artifact_sha256':a['artifact_sha256']}

def quality_session(root,study,row,a):
    session=root/'sessions'/row['key']
    if session.exists():raise FileExistsError('No session replacement')
    baseline=Path(study['baseline']['baseline_root']);benchmark=root/'benchmark'
    config=copy.deepcopy(read_json(baseline/'baseline-config.json'));config.pop('configurationSha256',None)
    config.update(runId=ID+'-'+row['key'],arm='baseline' if row['artifact_key']=='pristine' else 'candidate')
    if row['artifact_key']!='pristine':
        relative='models/v2-'+a['artifact_sha256']+'/resolve/'+a['artifact_sha256']+'/'
        target=benchmark/'public'/relative
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(a['path'],target,copy_function=os.link)
        origin=urlparse(config['model']['modelUrl']);url=origin.scheme+'://'+origin.netloc+'/'+relative
        config['model'].update(id=a['model_id'],modelUrl=url)
        config.update(sourceCheckpointSha256=a['source_checkpoint_sha256'],convertedArtifactSha256=a['artifact_sha256'])
        names=['mlc-chat-config.json','tensor-cache.json','tokenizer.json',*[x['dataPath'] for x in read_json(target/'tensor-cache.json')['records']]]
        config['assets']=[{'url':url+n,'sha256':a['files'][n]['sha256']} for n in names]+[{'url':config['model']['modelLib'],'sha256':a['files']['model.wasm']['sha256']}]
    config['servedBuild']=study['served_build'];config['configurationSha256']=canonical_hash(config)
    items_path=baseline/'data/development-items.json';items=read_json(items_path)
    session.mkdir(parents=True,mode=0o700);(session/'benchmark').symlink_to(benchmark,target_is_directory=True)
    save(session/'baseline-config.json',config)
    files={str(session/'baseline-config.json'):file_sha256(session/'baseline-config.json'),str(items_path):file_sha256(items_path)}
    files.update({p:d for p,d in study['files'].items() if Path(p).is_relative_to(benchmark) and not Path(p).is_relative_to(benchmark/'node_modules') and 'models' not in Path(p).relative_to(benchmark).parts})
    manifest={'schema_version':1,'kind':'fresh_baseline_quality_session','arm':config['arm'],'run_id':config['runId'],'slot':row['key'],
        'study_manifest_sha256':canonical_hash(study),'configuration_sha256':config['configurationSha256'],
        'artifact_sha256':a['artifact_sha256'],'checkpoint':{**read_json(baseline/'manifest.json')['checkpoint'],'sha256':a['source_checkpoint_sha256']},
        'scheduled_prompts':32,'scheduled_instructions':sum(len(i['target']['instruction_id_list']) for i in items),
        'items_path':str(items_path),'items_sha256':canonical_hash(items),'settings':SETTINGS,'policy':POLICY,'workload_identity':'ifeval-task','browser_seconds':3600,
        'grader':study['baseline']['grader'],'official_ifeval_root':study['baseline']['official_ifeval_root'],'official_assets':study['baseline']['official_assets'],
        'files':files,'environment':study['environment'],'legacy_transport_discriminator':'Unchanged client schema; new run and study identities identify this separate comparison.'}
    save(session/'manifest.json',manifest);save(session/'ready.json',{'manifest_sha256':canonical_hash(manifest)})
    with runtime.ledger(session,manifest,create=True) as book:save(session/'accounting-before.json',book.summary())
    return session

def freeze_resource(root,study,state):
    if (root/'resource-comparison-preset.json').exists():return
    if any(r['status']!='completed' for r in state['sessions'][:6]):raise ValueError('Quality phase incomplete')
    comparison={'kind':'fresh_six_artifact_resource_preset','study_sha256':canonical_hash(study),'candidate_binding_sha256':state['candidate_binding_sha256'],
        'session_order':ORDER,'artifact_identities':{key:artifact_for(root,study,key)['artifact_sha256'] for key in KEYS},'settings':resource.RESOURCE_SETTINGS,'limits':LIMITS}
    save(root/'resource-comparison-preset.json',comparison)
    save(root/'final-selection.json',{'kind':'adapter_binding_only_no_quality_gate','resource_preset_sha256':canonical_hash(comparison)})

def observed_environment(target):
    for name in ('browser-report.json','loaded.json'):
        path=target/name
        if path.exists():
            obj=read_json(path)
            if obj.get('environment'):return obj['environment']
    result=read_json(target/'result.json') if (target/'result.json').exists() else {}
    direct=result.get('environment') or result.get('driver',{}).get('environment')
    if direct:return direct
    if result.get('user_agent'):
        version=target/'chrome-profile/Last Version'
        return {'userAgent':result['user_agent'],'platform':result.get('platform'),'hardwareConcurrency':result.get('hardware_concurrency'),
            'chrome_version':version.read_text().strip() if version.exists() else None}
    return None

def run_one(root,key):
    root,study,state=load(root);available(root,study,state);row=next(r for r in state['sessions'] if r['key']==key)
    began=time.monotonic()
    try:
        with bounded_python(min(600,LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())):load(root,verify=True)
    except BaseException as exc:
        elapsed=time.monotonic()-began;state['charged_local_seconds']+=elapsed
        row.update(status='stopped',error=type(exc).__name__+': '+str(exc),dispatch_absence_verified=True)
        state['halted']=True;state['stages'].append({'kind':'verification','key':key,'charged_seconds':elapsed,'status':'failed'})
        atomic(root/'state.json',state);return {'status':'stopped','error':row['error']}
    elapsed=time.monotonic()-began;state['charged_local_seconds']+=elapsed
    state['stages'].append({'kind':'verification','key':key,'charged_seconds':elapsed,'status':'completed'})
    atomic(root/'state.json',state)
    try:
        a=artifact_for(root,study,row['artifact_key']);limit=3600 if row['kind']=='quality' else 1800
        seconds=min(limit,LIMITS['browser_seconds']-state['charged_browser_seconds'],LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())
        row=reserve(state,key,seconds,time.time())
    except BaseException as exc:
        row.update(status='stopped',error=type(exc).__name__+': '+str(exc),dispatch_absence_verified=True)
        state['halted']=True;atomic(root/'state.json',state);return {'status':'stopped','error':row['error']}
    atomic(root/'state.json',state)
    start=time.monotonic();execution=None;summary=None;error=None;session=None
    try:
        with bounded_python(seconds):
            if row['kind']=='quality':session=quality_session(root,study,row,a)
            else:
                freeze_resource(root,study,state)
                session=resource.prepare_session(root,study,row['key'],None if row['artifact_key']=='pristine' else a)
            def on_load():
                if row['load_granted']:raise RuntimeError('Repeated load forbidden')
                row['load_granted']=True;atomic(root/'state.json',state)
                return seconds-(time.monotonic()-start)-(95 if row['kind']=='resource' else 10)
            runner=runtime if row['kind']=='quality' else resource
            execution=runner.execute(session,on_load=on_load,preparation_seconds=min(300,max(.001,seconds-(time.monotonic()-start)-95)))
            if execution['status']!='completed' or not all(execution['cleanup'].get(k) is True for k in CLEANUP) or execution['cleanup'].get('retained_owned_pids_alive')!=[]:raise RuntimeError('Execution incomplete or cleanup unconfirmed')
            target=session/'execution'/('baseline-1' if row['artifact_key']=='pristine' else 'candidate-1')
            if row['kind']=='resource':
                summary=read_json(target/'resource-summary.json')
                if not historical.completed_execution(execution,summary):raise RuntimeError('Resource evidence incomplete')
                config=read_json(session/'baseline-config.json')
                if summary['runtime']!=study['environment']['runtime'] or summary['quantization']!=study['environment']['quantization'] or summary['model_lib']!=config['model']['modelLib']:raise ValueError('Observed resource runtime or library mismatch')
            row['observed_environment']=observed_environment(target)
            if not environment_matches(study['environment'],row['observed_environment']):raise ValueError('Observed browser/device identity unsupported or mismatched')
            peers=[r for r in state['sessions'] if r['kind']==row['kind'] and r['status']=='completed']
            if any(r.get('observed_environment')!=row['observed_environment'] for r in peers):raise ValueError('Observed browser/device identity changed between sessions')
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    elapsed=time.monotonic()-start;state['charged_browser_seconds']+=elapsed;state['charged_local_seconds']+=elapsed
    if execution is None and session:
        p=session/'execution'/('baseline-1' if row['artifact_key']=='pristine' else 'candidate-1')/'supervisor-result.json'
        if p.exists():execution=read_json(p)
    row.update(charged_browser_seconds=elapsed,error=error)
    if execution:row['observed']={k:execution.get(k) for k in ('status','engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches','generation_budget','cleanup')}
    state['active']=None
    if not error and row['kind']=='quality':
        started=time.monotonic();grading_seconds=min(900,LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())
        state['active']={'key':key+':grading','pid':os.getpid(),'started_epoch_seconds':time.time(),'reserved_seconds':grading_seconds};atomic(root/'state.json',state)
        try:
            deadline=time.monotonic()+grading_seconds
            def before_grade():
                if deadline-time.monotonic()<20:raise TimeoutError('Grading allowance exhausted')
            with bounded_python(grading_seconds):summary=runtime.grade(session,before_grade=before_grade)
            if not summary['technically_valid'] or not summary['semantic_grading_complete'] or summary.get('deployment_limits_satisfied') is not True or summary.get('memory_limits_satisfied') is not True:raise RuntimeError('Official grading or deployment/memory validity incomplete')
        except BaseException as exc:error=type(exc).__name__+': '+str(exc)
        grading_elapsed=time.monotonic()-started;state['charged_local_seconds']+=grading_elapsed
        state['stages'].append({'kind':'grading','key':key,'charged_seconds':grading_elapsed,'status':'failed' if error else 'completed'});state['active']=None
    good=not error and elapsed<=seconds
    row.update(status='completed' if good else 'stopped',error=error,summary=summary)
    state['halted']=not good;atomic(root/'state.json',state)
    return {'key':key,'status':row['status'],'strict_passes':None if row['kind']!='quality' or not summary else summary['metrics']['prompt_level_strict_accuracy']['passed'],'elapsed_seconds':elapsed,'error':error}

def finite(x):return type(x) in (int,float) and math.isfinite(x)
def environment_matches(expected,observed):
    return bool(observed and observed.get('userAgent') and observed.get('platform') and 'Chrome/'+expected['browser_version'].split('.')[0]+'.' in observed['userAgent'] and
        str(observed.get('hardwareConcurrency'))==expected['hardware']['hw.logicalcpu'] and
        ('chrome_version' not in observed or observed['chrome_version']==expected['browser_version']))

def aggregate_resources(sessions,matched):
    aggregates={}
    for key in KEYS:
        rows=[r for r in sessions if r['artifact_key']==key];metrics={}
        for name in historical.METRICS:
            values=[r['summary'].get(name) if r['status']=='completed' and r.get('summary') else None for r in rows]
            valid=len(values)==2 and all(finite(x) for x in values)
            metrics[name]={'values':values,'mean':sum(values)/2 if valid else None,'min':min(values) if valid else None,'max':max(values) if valid else None}
        aggregates[key]={'n_scheduled':2,'n_completed':sum(r['status']=='completed' for r in rows),'metrics':metrics}
    comparisons={}
    for key in KEYS[1:]:
        metrics={}
        for name in historical.METRICS:
            base=aggregates['pristine']['metrics'][name]['mean'];candidate=aggregates[key]['metrics'][name]['mean'];valid=matched and base is not None and candidate is not None
            metrics[name]={'baseline_mean':base,'candidate_mean':candidate,'absolute_delta':candidate-base if valid else None,'relative_percent':100*(candidate-base)/base if valid and base!=0 else None}
        comparisons[key]={'replication_unit':'session','metrics':metrics}
    return aggregates,comparisons

def verify_session(root,study,state,row):
    session=root/'sessions'/row['key'];manifest=read_json(session/'manifest.json');config=read_json(session/'baseline-config.json')
    a=artifact_for(root,study,row['artifact_key']);arm='baseline' if row['artifact_key']=='pristine' else 'candidate'
    runtime.load(session)
    if (manifest['study_manifest_sha256']!=state['study_sha256'] or manifest['artifact_sha256']!=a['artifact_sha256'] or
        manifest['checkpoint']['sha256']!=a['source_checkpoint_sha256'] or manifest['slot']!=row['key'] or
        manifest['run_id']!=ID+'-'+row['key'] or config['runId']!=manifest['run_id'] or manifest['arm']!=arm or config['arm']!=arm or
        config['convertedArtifactSha256']!=a['artifact_sha256'] or config['sourceCheckpointSha256']!=a['source_checkpoint_sha256'] or
        config['servedBuild']!=study['served_build'] or manifest.get('environment')!=study['environment']):
        raise ValueError('Session identity/configuration differs from frozen study')
    return manifest,config

def report_value(root,study,state):
    quality=[];sessions=[];artifacts=[]
    for key in KEYS:
        if key=='astra-r4' and not state['candidate_binding_sha256']:continue
        a=artifact_for(root,study,key)
        source_edit=a.get('source_edit') or {'changed_stored_entries':0 if key=='pristine' else 122880,
            'changed_numerical_entries':0 if key=='pristine' else 122880,'targeted_coordinates':0 if key=='pristine' else 122880,
            'scope':'Previously audited fixed artifact; no source reconstruction in this new cohort.'}
        artifacts.append({'key':key,'label':LABELS[key],'artifact_sha256':a['artifact_sha256'],'source_sha256':a['source_checkpoint_sha256'],
            'model_library_sha256':a['files']['model.wasm']['sha256'],'weight_operations':a['weight_operations'],
            'source_edit':source_edit,'source_tensor_contract':a.get('source_tensor_contract'),
            'targeted_source_coordinates':source_edit['targeted_coordinates'],'observed_changed_source_entries':source_edit['changed_stored_entries'],
            'changed_source_entries':source_edit['changed_stored_entries']})
    for row in state['sessions']:
        session=root/'sessions'/row['key'];arm='baseline' if row['artifact_key']=='pristine' else 'candidate';target=session/'execution'/(arm+'-1')
        public={'artifact_key':row['artifact_key'],'status':row['status'],'position':row['position'],'observed_environment':row.get('observed_environment'),
            'expected_environment':study['environment'],'cleanup':row.get('observed',{}).get('cleanup')}
        if row.get('observed'):
            execution=read_json(target/'supervisor-result.json')
            if execution['study_manifest_sha256']!=state['study_sha256'] or any(row['observed'][k]!=execution.get(k) for k in row['observed']):raise ValueError('State counters disagree with saved execution')
            public['observed']={k:execution.get(k) for k in ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
            public['generation_budget']={k:execution['generation_budget'][k] for k in ('observed','charged')}
            public['warmup']={'observed_count':len(execution.get('warmup_outputs',[])),
                'output_tokens':sum(x['generated_tokens'] for x in execution.get('warmup_outputs',[])),
                'finish_reasons':[x['finish_reason'] for x in execution.get('warmup_outputs',[])]}
        if row.get('error'):public['error_type']=row['error'].split(':',1)[0]
        if (session/'manifest.json').exists():
            manifest,config=verify_session(root,study,state,row)
            public['manifest_sha256']=canonical_hash(manifest)
        if row['kind']=='quality':
            public.update(metrics=None,item_outcomes=[])
            path=session/arm/'graded.json'
            if path.exists():
                envelope=read_json(path);s=envelope['summary'];m=s['metrics']
                if envelope['study_manifest_sha256']!=state['study_sha256'] or envelope['manifest_sha256']!=canonical_hash(manifest):raise ValueError('Grade envelope is not bound to this fresh session')
                public['metrics']={'strict_passes':m['prompt_level_strict_accuracy']['passed'],'loose_passes':m['prompt_level_loose_accuracy']['passed'],
                    'denominator':32,'strict_instruction_passes':m['instruction_level_strict_accuracy']['passed'],'loose_instruction_passes':m['instruction_level_loose_accuracy']['passed'],
                    'instruction_denominator':s['scheduled_instructions'],'scored_output_tokens':s['known_generated_tokens'],
                    'generation_seconds':s['generation_seconds'],'natural_completions':s['natural_completions'],'truncations':s['truncations'],
                    'attempted':s['attempted'],'unexecuted':s['unexecuted'],'unresolved_grader_errors':s['unresolved_grader_errors'],
                    'technically_valid':s['technically_valid'],'semantic_grading_complete':s['semantic_grading_complete'],
                    'deployment_limits_satisfied':s['deployment_limits_satisfied'],'memory_limits_satisfied':s['memory_limits_satisfied']}
                public['graded_evidence_sha256']=file_sha256(path)
                for record,grade in zip(envelope['records'],envelope['grades'],strict=True):
                    public['item_outcomes'].append({'item_id':record['item_id'],'strict_pass':grade['metrics'].get('prompt_level_strict_accuracy') if grade['status']=='graded' else None,
                        'loose_pass':grade['metrics'].get('prompt_level_loose_accuracy') if grade['status']=='graded' else None,'status':record['status'],'grading_status':grade['status'],
                        'output_tokens':record['token_counts']['output'],'duration_seconds':record['duration_seconds'],'finish_reason':record.get('finish_reason')})
            quality.append(public)
        else:
            public.update(summary=None,requests=[],evidence_sha256=None)
            if (target/'supervisor-result.json').exists():
                public['evidence_sha256']=file_sha256(target/'supervisor-result.json')
                public['observed']={k:row.get('observed',{}).get(k) for k in ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
            if (target/'resource-summary.json').exists():
                summary=read_json(target/'resource-summary.json');binding=read_json(target/'resource-summary-binding.json')
                if canonical_hash(summary)!=binding['summary_sha256']:raise ValueError('Summary changed')
                verify_files(binding['input_files'])
                if summary['model']!=config['model']['id'] or summary['model_lib']!=config['model']['modelLib']:raise ValueError('Resource report model/library differs from session')
                public['summary']={k:summary.get(k) for k in historical.SUMMARY_KEYS};s=public['summary'];s['model_lib']=Path(urlparse(s['model_lib']).path).name
                s['model_library_sha256']=artifact_for(root,study,row['artifact_key'])['files']['model.wasm']['sha256'];n=s['measured_requests'];ms=s['response_ms_sum'];tokens=s['scored_output_tokens']
                s.update(mean_response_ms=ms/n if ms is not None and n else None,full_response_tokens_per_second=tokens/(ms/1000) if tokens is not None and ms else None,response_ms_per_output_token=ms/tokens if ms is not None and tokens else None)
                index=0
                for sample in read_json(target/'browser-report.json')['samples']:
                    if not sample['warmup']:index+=1
                    public['requests'].append({'request_index':0 if sample['warmup'] else index,'warmup':sample['warmup'],**{dest:sample.get(src) for src,dest in (('inputTokens','input_tokens'),('outputTokens','output_tokens'),('responseMs','response_ms'),('firstOutputMs','first_output_ms'),('decodeTokensPerSecond','decode_tokens_per_second'),('finishReason','finish_reason'),('valid','valid'),('startedMs','started_ms'),('endedMs','ended_ms'))}})
                public['summary_sha256']=binding['summary_sha256'];public['observed_environment']=summary['environment']
            sessions.append(public)
    qmatch=all(r['status']=='completed' and environment_matches(study['environment'],r['observed_environment']) for r in quality) and all(r['observed_environment']==quality[0]['observed_environment'] for r in quality)
    lib=study['artifacts']['pristine']['files']['model.wasm']['sha256']
    rmatch=(all(r['status']=='completed' and r.get('summary') and environment_matches(study['environment'],r['observed_environment']) and
        r['expected_environment']==study['environment'] and r['summary']['runtime']==study['environment']['runtime'] and
        r['summary']['quantization']==study['environment']['quantization'] and r['summary']['model_library_sha256']==lib for r in sessions)
        and all(r['observed_environment']==sessions[0]['observed_environment'] for r in sessions))
    aggregates,comparisons=aggregate_resources(sessions,rmatch)
    observed=[r['observed'] for r in state['sessions'] if r.get('observed')]
    usage={k:sum(x.get(k) or 0 for x in observed) for k in ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
    token_counts=[x['generation_budget']['observed']['generated_tokens'] for x in observed]
    usage.update(completed_sessions=sum(r['status']=='completed' for r in state['sessions']),completed_quality_sessions=sum(r['status']=='completed' for r in quality),completed_resource_sessions=sum(r['status']=='completed' for r in sessions),
        attempted_sessions=sum('capacity_reserved' in r for r in state['sessions']),reserved_target_tokens=sum(r.get('capacity_reserved',{}).get('target_tokens',0) for r in state['sessions']),
        unresolved_attempted_sessions=sum('capacity_reserved' in r and not r.get('observed') for r in state['sessions']),
        stopped_before_reservation=sum(r['status']=='stopped' and 'capacity_reserved' not in r for r in state['sessions']),
        observation_scope='Known observed supervisor counters; unresolved attempts remain reserved and are not asserted to have zero dispatches.',
        sampled_tokens_including_warmups=sum(token_counts) if all(type(x) is int for x in token_counts) else None,
        generation_retries=sum(x['generation_budget']['charged']['retries'] for x in observed),failed_target_sequences=sum(x['generation_budget']['charged']['failed_sequences'] for x in observed),
        charged_browser_seconds=state['charged_browser_seconds'],charged_local_seconds=state['charged_local_seconds'],editor_proposals=state['editor_proposals_consumed'],builds=state['build_consumed'],
        imports=int('response_import' in state.get('action_attempts',{})),conversions=int('conversion' in state.get('action_attempts',{})),packages=int('package' in state.get('action_attempts',{})),retained_bytes=total_retained(root,study))
    complete=all(r['status']=='completed' for r in state['sessions']);status='complete' if complete else 'stopped' if state['halted'] else 'in_progress'
    objectives=None
    if qmatch and all(r['status']=='completed' and r['metrics'] for r in quality):
        base=quality[0]['metrics'];new=quality[-1]['metrics'];best=max(r['metrics']['strict_passes'] for r in quality[:-1])
        objectives={'fresh_baseline_strict_passes':base['strict_passes'],'best_prior_fresh_strict_passes':best,'candidate_strict_passes':new['strict_passes'],
            'quality_gain_over_baseline':new['strict_passes']>base['strict_passes'],'quality_gain_over_best_prior':new['strict_passes']>best,
            'output_tokens_no_greater_than_baseline':new['scored_output_tokens']<=base['scored_output_tokens'],
            'strict_and_token_target_met':new['strict_passes']>best and new['scored_output_tokens']<=base['scored_output_tokens'],
            'resource_eligibility':'All technically valid artifacts measured regardless of quality; no retrospective gate.'}
    provenance=read_json(root/'candidate-binding.json')['editor_provenance'] if state['candidate_binding_sha256'] else None
    if provenance:provenance={k:v for k,v in provenance.items() if k not in ('files','paths','artifact_path','source_path','packet_path','response_path')}
    value={'schema_version':1,'kind':'fresh_baseline_r4_result','id':ID,'label':'Fresh baseline and fourth Astra edit','status':status,
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),'study_sha256':state['study_sha256'],'candidate_binding_sha256':state['candidate_binding_sha256'],
        'artifacts':artifacts,'environment':study['environment'],'limits':LIMITS,'actual_usage':usage,'preparation':study['preparation'],
        'quality':{'status':'complete' if all(r['status']=='completed' for r in quality) else status,'matched_environment':qmatch,'settings':SETTINGS,'order':KEYS,
            'benchmark':'IFEval exposed 32-prompt development set','untouched_final_prompts':0,'grader':study['baseline']['grader'],'artifacts':quality,'objectives':objectives},
        'resource':{'status':'complete' if all(r['status']=='completed' for r in sessions) else status,'matched_environment':rmatch,'settings':resource.RESOURCE_SETTINGS,'session_order':ORDER,
            'sessions':sessions,'aggregates':aggregates,'comparisons':comparisons,'counter_definitions':historical.COUNTERS,'upstream_revision':study['resource_preset']['upstream_revision']},
        'editor_provenance':provenance,'source_identity':{'base_commit':study['source_commit'],'files':study['source_files'],'python':study['python_identity'],'frozen_input_manifest_sha256':canonical_hash(study['files'])},
        'limitations':['All 32 quality prompts were previously exposed; adaptive development evidence, not untouched confirmation.',
            'One quality run per artifact and two resource sessions on one device; no significance or resource-equivalence claim.',
            'Mirrored resource order balances linear ordinal drift, not arbitrary elapsed-time or host-condition changes.',
            'CPU/GPU activity rates are not cumulative compute; allocation descriptors are not physical VRAM; missing power remains null.',
            'Session throughput ratios are averaged equally, not pooled across sessions.','Historical studies and ledgers remain unchanged.']}
    return value

def export(root,out):
    began=time.monotonic();root,study,state=load(root);value=None;error=None
    if state.get('reporting_active'):raise RuntimeError('Interrupted export reservation remains unresolved')
    seconds=min(900,LIMITS['local_seconds']-state['charged_local_seconds'],study['deadline_epoch_seconds']-time.time())
    if seconds<=0:raise TimeoutError('No reporting allowance remains')
    state['reporting_active']={'pid':os.getpid(),'started_epoch_seconds':time.time(),'reserved_seconds':seconds};atomic(root/'state.json',state)
    try:
        with bounded_python(seconds):value=report_value(root,study,state)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        elapsed=time.monotonic()-began;state['charged_local_seconds']+=elapsed
        over=elapsed>seconds or state['charged_local_seconds']>LIMITS['local_seconds'] or time.time()>study['deadline_epoch_seconds']
        if over:error=error or 'TimeoutError: reporting exceeded the unchanged study budget'
        state['stages'].append({'kind':'export','status':'failed' if error else 'completed','charged_seconds':elapsed,'error':error})
        state['reporting_active']=None
        if error:state['halted']=True
        atomic(root/'state.json',state)
    if value is not None:
        value['actual_usage']['charged_local_seconds']=state['charged_local_seconds']
        if error:value['status']='stopped';value['reporting_error_type']=error.split(':',1)[0]
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);atomic(out,value)
    if error:raise RuntimeError(error)
    return {'status':value['status'],'quality_completed':value['actual_usage']['completed_quality_sessions'],'resource_completed':value['actual_usage']['completed_resource_sessions'],'out':str(out)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','status','run-quality-prefix','run-rest','begin-editor','import-response','build-candidate','report'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,default=SOURCE_ROOT/'configs/fresh_baseline_r4.json')
    parser.add_argument('--build-preflight',type=Path);parser.add_argument('--artifact',type=Path);parser.add_argument('--provenance',type=Path)
    parser.add_argument('--packet',type=Path);parser.add_argument('--response',type=Path);parser.add_argument('--rationale',type=Path)
    parser.add_argument('--out',type=Path,default=SOURCE_ROOT/'reports/fresh-baseline-r4-v1/result.json')
    args=parser.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['WSL_NATIVE_PYTHON']=sys.executable
    if args.command=='status':result=load(args.root)[2]
    else:
        with _lock(args.root):
            if args.command=='prepare':result=prepare(args.root,args.config,args.build_preflight)
            elif args.command=='begin-editor':result=begin_editor(args.root,args.packet)
            elif args.command=='import-response':result=import_response(args.root,args.response,args.rationale)
            elif args.command=='build-candidate':result=build_candidate(args.root,args.provenance)
            elif args.command=='report':result=export(args.root,args.out)
            else:
                root,study,state=load(args.root);available(root,study,state)
                targets=state['sessions'][:5] if args.command=='run-quality-prefix' else state['sessions'][5:]
                for row in targets:
                    if row['status']=='completed':continue
                    result=run_one(root,row['key']);print(json.dumps(result),flush=True);export(root,args.out)
                    if result['status']!='completed':break
                result=export(root,args.out)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
