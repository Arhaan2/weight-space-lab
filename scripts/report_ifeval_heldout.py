#!/usr/bin/env python3
"""Write a sanitized final subset report from preserved scored evidence. Arhaan2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weight_space_lab.ifeval_baseline import read_events, save
from weight_space_lab.ifeval_heldout_analysis import analyze
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash

AUTHOR = '[Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2)'
DEVELOPMENT = ('The sole development improvement (schedule item 17) required at least 22 highlighted sections. '
    'The baseline echoed long plot text with 3 highlighted spans; the candidate repeatedly emitted the same short title, '
    'producing 169 complete highlighted spans before truncation. Both used 1,020 sampled tokens. The official strict and '
    'loose fail-to-pass grades remain unchanged. This gain reflects repetitive content satisfying the formatting checker; '
    'it does not demonstrate better substantive answering or reduced repetition.')


def resources(directory):
    if not (directory/'events.jsonl').exists() and not (directory/'resources.jsonl').exists():
        return {'available':False}
    events = read_events(directory)
    points = [e['data'] for e in events if e['type']=='point']
    peaks = {key:max([p[key] for p in points if type(p.get(key)) in (int,float)],default=None)
             for key in ('gpuBufferBytes','wasmBytes','heapBytes','eventLoopDelayMs')}
    host,roles={},{}
    path=directory/'resources.jsonl'
    data=path.read_bytes() if path.exists() else b''
    complete=data[:data.rfind(b'\n')+1]
    rows=[json.loads(line) for line in complete.splitlines()]
    for row in rows:
        for key,value in row.get('host',{}).items():
            if type(value) in (int,float):
                previous=host.setdefault(key,{'min':value,'max':value,'samples':0})
                previous.update(min=min(previous['min'],value),max=max(previous['max'],value),samples=previous['samples']+1)
        for process in row.get('owned_processes',[]):
            role=roles.setdefault(process['role'],{})
            for key in ('footprintBytes','rssBytes','wiredBytes','userCpuNs','systemCpuNs',
                        'diskReadBytes','diskWrittenBytes','kernelEnergyNanojoules'):
                if type(process.get(key)) in (int,float):
                    role[key]=max(role.get(key,0),process[key])
    return {'available':True,'worker_sampled_maxima':peaks,'worker_sample_count':len(points),
        'host_sampled_numeric_ranges':host,'host_sample_count':len(rows),
        'trailing_incomplete_resource_record':len(complete)!=len(data),'owned_role_single_process_maxima':roles,
        'owned_process_counter_meaning':'CPU time, disk bytes and kernel energy are cumulative counters since each process was created. Values are maximum single observed process counters per role, not task-only deltas or sums across processes. Kernel energy is not battery energy and these observations do not establish a controlled energy difference.',
        'meaning':'GPUBuffer descriptor bytes and WASM address space are not physical VRAM. Host readings include unrelated work; process memory can include shared allocations; sampled maxima are not absolute peaks.'}


def arm_summary(root,arm,accounting):
    folder=root/arm
    path=folder/'execution'/(arm+'-1')/'supervisor-result.json'
    run=read_json(path) if path.exists() else None
    if run is None and (folder/'execution').exists():
        raise ValueError('Execution started without reconciled supervisor evidence; preserve and reconcile before reporting')
    manifest=read_json(folder/'manifest.json')
    book=run['generation_budget'] if run else read_json(folder/'accounting-before.json')
    cleanup=(run or {}).get('cleanup',{})
    return {'source_checkpoint_sha256':manifest['checkpoint']['sha256'],
        'artifact_sha256':manifest['artifact_sha256'],'configuration_sha256':manifest['configuration_sha256'],
        'manifest_sha256':canonical_hash(manifest),
        'accounting':{k:v for k,v in accounting.items() if not k.endswith('_item_ids')},
        'execution_status':run['status'] if run else 'unexecuted',
        'runtime_failure_recorded':bool(run and run.get('error')),
        'cleanup_applicable':bool(run and run.get('engine_load_attempts')),
        'budget':{**{key:(run or {}).get(key,0) for key in
            ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')},
            'first_load_through_cleanup_seconds':(run or {}).get('first_load_through_cleanup_seconds')},
        'ledger':{k:book[k] for k in ('limits','reserved','observed','charged','cancelled')},
        'cleanup':{key:cleanup.get(key) for key in ('unload_confirmed','worker_terminated','chrome_exit_confirmed',
            'local_harness_exit_confirmed','controller_exit_verified','chrome_os_exit_verified',
            'forced_termination','owned_browser_and_worker_resources_released')},
        'remaining_owned_processes':len(cleanup.get('retained_owned_pids_alive',[])) if run else 0,
        'warmup':[{k:o.get(k) for k in ('generated_tokens','input_tokens','response_latency_ms','finish_reason')}
                  for o in (run or {}).get('warmup_outputs',[])],
        'resources':resources(folder/'execution'/(arm+'-1'))}



def public_interpretation(value):
    """Publish qualitative summaries and schedule positions, never raw evidence."""
    rows=[]
    for row in value.get('changed_outcomes',[]):
        position=row.get('schedule_position')
        if type(position) is not int or not 1 <= position <= 128:
            raise ValueError('Interpretation requires a frozen 1-based schedule position')
        clean={key:row[key] for key in ('schedule_position','strict_change','loose_change','observation')}
        if any(not isinstance(clean[key],str) for key in ('strict_change','loose_change','observation')):
            raise ValueError('Interpretation notes must be text')
        rows.append(clean)
    if len({r['schedule_position'] for r in rows}) != len(rows) or not isinstance(value.get('summary'),str):
        raise ValueError('Interpretation requires a summary and unique schedule positions')
    return {'summary':value['summary'],'changed_outcomes':rows}


def aggregate_budget(root,arms):
    result={key:sum(a['budget'][key] for a in arms.values()) for key in
        ('engine_load_attempts','warmup_dispatches','scored_dispatches','generation_dispatches')}
    known=sum(a['ledger']['observed']['generated_tokens'] or 0 for a in arms.values())
    unknown=sum(a['budget']['generation_dispatches']-a['ledger']['observed']['token_measurements'] for a in arms.values())
    if unknown < 0:
        raise ValueError('Token measurements exceed generation dispatches')
    result.update(known_reported_generated_tokens=known,
        reported_generated_tokens=known if unknown == 0 else None,
        requests_with_unknown_generated_tokens=unknown,
        reserved_generated_tokens=sum(a['ledger']['reserved']['generated_tokens'] for a in arms.values()),
        charged_generated_tokens=sum(a['ledger']['charged']['generated_tokens'] for a in arms.values()),
        automatic_retries=sum(a['ledger']['charged']['retries'] for a in arms.values()),
        pending_requests=sum(a['ledger']['charged']['pending_requests'] for a in arms.values()),
        limit_violations=sum(a['ledger']['charged']['limit_violations'] for a in arms.values()),
        new_weight_proposals=0,development_reruns=0,experimental_frontier_requests=0)
    grants={a:read_json(root/a/'load-grant.json') for a in arms if (root/a/'load-grant.json').exists()}
    durations={a:arms[a]['budget']['first_load_through_cleanup_seconds'] for a in grants}
    if grants and all(type(v) in (int,float) for v in durations.values()):
        result['first_load_through_final_cleanup_seconds']=(
            max(grants[a]['started_monotonic']+durations[a] for a in grants)
            -min(g['started_monotonic'] for g in grants.values()))
    else:
        result['first_load_through_final_cleanup_seconds']=None
    return result


def seconds_text(value):
    return 'unavailable' if value is None else f'{value:,.3f} seconds'


def report(root,public):
    root,public=Path(root).resolve(),Path(public)
    if any((public/name).exists() for name in ('result.json','README.md')):
        raise FileExistsError('Final public report destinations must be new')
    freeze=read_json(root/'evaluation-freeze.json')
    verified=read_json(public/'verification.json')
    if not verified['passed']:
        raise ValueError('Independent integrity verification must pass before publication')
    analysis=analyze(root)
    if (root/'analysis.json').exists():
        assert read_json(root/'analysis.json') == analysis
    else:
        save(root/'analysis.json',analysis)
    interpretation=public_interpretation(read_json(root/'interpretation.json'))
    stop_path=root/'stop-diagnostic.json'
    stop_summary=read_json(stop_path).get('public_summary') if stop_path.exists() else None
    if stop_summary is not None and not isinstance(stop_summary,str):
        raise ValueError('Public stop diagnosis must be sanitized text')
    arms={a:arm_summary(root,a,analysis['arm_accounting'][a]) for a in ('baseline','candidate')}
    metrics={}
    for mode in ('strict','loose'):
        pair=analysis['prompt_comparisons'][mode]
        primary=pair['comparison']
        metrics['prompt_'+mode]={'baseline':pair['baseline_projection']['official_metrics']['prompt_level_'+mode+'_accuracy'],
            'candidate':pair['candidate_projection']['official_metrics']['prompt_level_'+mode+'_accuracy'],
            'difference_percentage_points':100*primary['delta'],
            'ci95_percentage_points':[100*primary['confidence_interval']['low'],100*primary['confidence_interval']['high']]}
        item=analysis['instruction_comparisons'][mode]
        metrics['instruction_'+mode]={k:item[k] for k in ('baseline','candidate','difference_percentage_points')}
        metrics['instruction_'+mode]['ci95_percentage_points']=[item['confidence_interval']['low_percentage_points'],item['confidence_interval']['high_percentage_points']]
    complete=analysis['complete_paired_evaluation']
    primary=metrics['prompt_strict']
    delta=primary['difference_percentage_points']
    ci=primary['ci95_percentage_points']
    if not complete:
        conclusion='The held-out comparison is incomplete; no completed instruction-compliance gain is established.'
    elif delta>0:
        conclusion=('The candidate has higher observed strict prompt compliance on this subset. '
                    +('The paired 95% interval includes zero, so the direction of the gain remains uncertain.' if ci[0]<=0 else
                      'The paired 95% interval lies above zero for this preselected subset comparison.'))
    elif delta<0:
        conclusion=('The candidate has lower observed strict prompt compliance; this evaluation does not support a gain. '
                    +('The paired 95% interval includes zero.' if ci[1]>=0 else 'The paired 95% interval lies below zero.'))
    else:
        conclusion='Strict prompt compliance is unchanged; this evaluation does not show a held-out gain.'
    aggregate=aggregate_budget(root,arms)
    value={'schema_version':1,'kind':'final_ifeval_subset_report','author':'Arhaan2',
        'label':'128-example IFEval on-device subset','status':analysis['status'],
        'evaluation_freeze_sha256':canonical_hash(freeze),'evaluation_items_sha256':freeze['evaluation_items_sha256'],
        'original_data_manifest_sha256':freeze['original_data_manifest_sha256'],
        'original_data_manifest_file_sha256':freeze['original_data_manifest_file_sha256'],
        'source_revision':freeze['original_source']['revision'],'source_file_sha256':freeze['original_source']['sha256'],
        'paired_analysis_manifests':freeze['paired_analysis_manifests'],'execution_order':freeze['execution_order'],
        'settings':freeze['settings'],'quantization':freeze['quantization'],
        'tokenizer_sha256':freeze['tokenizer_sha256'],'template_sha256':freeze['template_sha256'],
        'runtime_sha256':freeze['runtime_sha256'],'model_library_sha256':freeze['model_library_sha256'],
        'grader':freeze['grader'],'policy':freeze['policy'],'arms':arms,'metrics':metrics,
        'paired_outcomes':analysis['paired_outcomes'],'aggregate_budget':aggregate,'limits':freeze['limits'],
        'analysis_specification':analysis['analysis_spec'],'exact_paired_significance_test':analysis['exact_paired_significance_test'],
        'conclusion':conclusion,'complete_paired_evaluation':complete,
        'semantic_grading_complete':analysis['semantic_grading_complete'],
        'metric_interpretation':('measured complete paired compliance' if complete else
            'all-scheduled zero-credit projections; incomplete comparison, not an identified completed semantic effect'),
        'grader_error_interpretation':analysis.get('grader_error_interpretation'),
        'resource_improvements_demonstrated':False,
        'resource_conclusion':('A single sequential comparison does not demonstrate a small speed or energy improvement. '
            +('The workloads are unequal: '+str(arms['baseline']['accounting']['generation_dispatches'])+' pristine scored requests versus '
              +str(arms['candidate']['accounting']['generation_dispatches'])+' candidate scored requests. Aggregate response times, tokens, and resources therefore cannot establish a speed or energy gain. '
              if arms['baseline']['accounting']['generation_dispatches'] != arms['candidate']['accounting']['generation_dispatches'] else '')
            +'Descriptor allocations and WASM address space are not physical VRAM.'),
        'stop_diagnosis':stop_summary,'stop_diagnostic_sha256':file_sha256(stop_path) if stop_path.exists() else None,
        'development_item_17_interpretation':DEVELOPMENT,'heldout_changed_outcomes':interpretation,
        'editor_isolation':'unverified','proposal_provenance':freeze['proposal_session_provenance']['statement'],
        'proposal_provenance_independently_observed':False,'subset_exposed_for_future_edits':True,
        'verification_sha256':file_sha256(public/'verification.json')}
    def score(metric,arm):
        m=metrics[metric][arm]
        return f"{m['numerator']}/{m['denominator']} ({100*m['value']:.2f}%)"
    missing_candidate=set(analysis['arm_accounting']['candidate']['unexecuted_item_ids'])
    missing_instructions=sum(len(i['target']['instruction_id_list']) for i in read_json(root/'evaluation-items.json')
                             if i['item_id'] in missing_candidate)
    lines=['# 128-example IFEval on-device subset','',f'Authorship: {AUTHOR}.','',conclusion,'',
        f"Evaluation status: **{analysis['status']}**. This is the final evaluation of the frozen selected candidate; no further proposal, development rerun, or test expansion was performed.",'',
        *([f"**Candidate values below include zero credit for {len(missing_candidate)} unexecuted prompts ({missing_instructions} instructions). They are incomplete-schedule accounting, not measured failures or a completed estimate of the candidate's effect.**",''] if missing_candidate else []),
        '| Compliance | Pristine | Candidate | Difference (pp) | Paired 95% interval (pp) |',
        '| --- | ---: | ---: | ---: | --- |']
    for key,label in [('prompt_strict','Strict prompt (primary)'),('prompt_loose','Loose prompt'),
                      ('instruction_strict','Strict instruction'),('instruction_loose','Loose instruction')]:
        m=metrics[key];lo,hi=m['ci95_percentage_points']
        lines.append(f"| {label} | {score(key,'baseline')} | {score(key,'candidate')} | {m['difference_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    if stop_summary:
        lines+=['',stop_summary]
    if not complete:
        lines+=['','This table retains the frozen zero-credit policy for missing or unresolved slots. The displayed differences and intervals describe that projection; even an interval excluding zero cannot establish an instruction-compliance gain or degradation from this incomplete comparison.']
    if not analysis['semantic_grading_complete']:
        lines+=['','Unresolved grader errors receive zero credit only for aggregate lower bounds. They are not demonstrated model failures, and the difference between arm bounds is not an identified semantic effect.']
    lines+=['', 'All 128 scheduled prompts and 173 scheduled instructions remain in the denominators. Instruction scores use counts, not an average of prompt percentages. Intervals use 10,000 paired bootstrap resamples, seed 0, with whole prompts as clusters; instruction ratios are recomputed within each resample. The existing analysis has no exact paired significance test. Significance was not a completion condition.','']
    for mode,changes in analysis['paired_outcomes'].items():
        lines.append(f"{mode.title()} prompt pairs: **{changes['improved']} improvements, {changes['regressed']} regressions, {changes['tied']} ties**, and {changes['unresolved']} unresolved pairs.")
    lines+=['','| Execution observation | Pristine | Candidate |','| --- | ---: | ---: |']
    for key,label in [('natural_completions','Natural completions'),('truncations','Truncations'),('unexecuted_slots','Unexecuted slots'),
        ('missing_outputs','Missing outputs (including unexecuted slots)'),
        ('attempted_model_failures','Attempted model failures'),('ambiguous_dispatch_outcomes','Ambiguous dispatch outcomes'),
        ('attempted_outputs_with_unknown_tokens','Attempted outputs with unknown token counts'),
        ('grader_errors','Grader errors'),('grader_timeouts','Grader timeouts'),
        ('known_generated_tokens','Scored sampled output tokens'),('known_response_seconds','Scored response seconds')]:
        vals=[arms[a]['accounting'][key] for a in arms]
        lines.append(f"| {label} | {vals[0]:,.3f} | {vals[1]:,.3f} |" if key.endswith('seconds') else f"| {label} | {vals[0]:,} | {vals[1]:,} |")
    lines+=['',f"Actual aggregate consumption: {aggregate['engine_load_attempts']}/2 engine loads, {aggregate['warmup_dispatches']}/2 warm-ups, {aggregate['scored_dispatches']}/256 scored requests, {aggregate['generation_dispatches']}/258 generation dispatches, and {aggregate['known_reported_generated_tokens']:,} known reported generated tokens against the 262,176-token limit. {aggregate['requests_with_unknown_generated_tokens']} dispatched requests have unknown token counts. Reservations total {aggregate['reserved_generated_tokens']:,} tokens; conservative charged usage is {aggregate['charged_generated_tokens']:,} tokens. First load through final cleanup: {seconds_text(aggregate['first_load_through_final_cleanup_seconds'])}. Automatic retries: {aggregate['automatic_retries']}; pending requests: {aggregate['pending_requests']}; recorded limit violations: {aggregate['limit_violations']}. Zero proposals, development reruns, or experimental frontier requests.",'',
        'Executed warm-ups used the frozen prompt and at most 16 tokens each, are included in the budget, and are excluded from all task scores. Four injected empty-thinking prefix IDs count toward max_tokens but not sampled completion usage, permitting at most 1,020 sampled tokens for a 1,024-token request and 12 for a 16-token warm-up.','']
    for arm,a in arms.items():
        lines.append(f"{arm.title()} load through cleanup: {seconds_text(a['budget']['first_load_through_cleanup_seconds'])}. Cleanup {'applicable' if a['cleanup_applicable'] else 'not applicable to an engine load'}; confirmations: {', '.join(k+'='+str(v) for k,v in a['cleanup'].items())}; remaining owned processes: {a['remaining_owned_processes']}.")
    lines+=['','Frozen order: pristine then candidate, one engine at a time; candidate loading required verified pristine unload. Both arms were bound to fresh conversations, thinking disabled, temperature 0, top-p 1, seed 42 reset per request, context 4,096, output cap 1,024, WebLLM 0.2.86 and q4f16_1. Request deadlines remained 120 seconds, per-model deadlines 90 minutes and aggregate deadline 180 minutes.','',
        '| Resource observation (sampled maximum) | Pristine | Candidate |','| --- | ---: | ---: |']
    for key,label in [('gpuBufferBytes','Worker GPUBuffer descriptor bytes'),('wasmBytes','Worker WASM address-space bytes'),('heapBytes','Worker JS heap bytes'),('eventLoopDelayMs','Worker timer delay ms')]:
        vals=[arms[a]['resources'].get('worker_sampled_maxima',{}).get(key) for a in arms]
        lines.append('| '+label+' | '+' | '.join('unavailable' if v is None else f'{v:,}' for v in vals)+' |')
    lines+=['',value['resource_conclusion']+' Owned-process memory, CPU time, disk bytes, kernel energy and host-wide numeric ranges are preserved separately in the JSON when available. CPU, disk and kernel-energy values are cumulative counters since each process was created, reported as maximum single observed process counters per role; they are not task-only deltas or sums across processes. Kernel energy is not battery energy. Shared allocations, unrelated host activity and sample spacing limit these observations. No controlled energy improvement was demonstrated.','',DEVELOPMENT,'',
        interpretation['summary'],'']
    for row in interpretation['changed_outcomes']:
        observation=' '.join(row['observation'].splitlines())
        lines.append(f"- Schedule item {row['schedule_position']}: strict prompt {row['strict_change']}; loose prompt {row['loose_change']}. {observation}")
    lines+=['',
        'The official scores are unchanged by this qualitative inspection. Instruction compliance and substantive answer correctness are separate; no post hoc grader was introduced.','',
        'Frozen identities (all four model identities were verified against actual existing bytes before generation):','']
    for arm in arms:
        lines += [f"- {arm.title()} source checkpoint: `{arms[arm]['source_checkpoint_sha256']}`.",f"- {arm.title()} deployed artifact: `{arms[arm]['artifact_sha256']}`."]
    lines += [f"- Evaluation freeze: `{value['evaluation_freeze_sha256']}`.",f"- Frozen 128-item content/order: `{value['evaluation_items_sha256']}`.",
        f"- Original complete data manifest: `{value['original_data_manifest_sha256']}`.",
        f"- Original Google source revision: `{value['source_revision']}`; source file: `{value['source_file_sha256']}`.",'',
        'The [machine-readable report](result.json) preserves configuration, tokenizer, template, runtime, model-library, grader, paired-analysis and resource identities. The [independent verifier](verification.json) checks evidence, schedule, accounting and preservation. Private prompts, outputs, session packets, authorization records and machine paths remain outside these public artifacts.','',
        'Limitations: this is a 128-example subset, not the full 541-example IFEval evaluation. Frozen eligibility and coverage selection do not establish representative full-benchmark performance or absence of pretraining exposure. The bootstrap describes one preselected comparison. This remains exploratory on-device work with editor isolation unverified. The supplied proposal-session provenance reports file-reading and JSON-parsing tools used on the development packet, with no external research, additional weight inspections, or target forward passes; that provenance was not independently observed. The selected weight columns are not a verified causal explanation of response behavior.','',
        'The evaluated subset is now exposed. Its feedback was not used for another weight edit, and it cannot later be called an untouched test set for further edits.','',
        'Upstream attribution remains with [Google Research IFEval (Zhou et al.)](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), Qwen, WebLLM/MLC, existing BrowserLLM work, Gufran’s contribution, and their preserved licenses.']
    public.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    markdown='\n'.join(lines)+'\n'
    for text in (encoded,markdown):
        if any(token in text for token in ('/Users/','/private/','/opt/homebrew/','authorization-source.txt','raw_output','task_text')):
            raise ValueError('Public report contains private material')
    for name,text in [('result.json',encoded),('README.md',markdown)]:
        with (public/name).open('x') as stream:
            stream.write(text)
    print(json.dumps({'status':analysis['status'],'conclusion':conclusion,'primary':primary,'budget':aggregate},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='artifacts/ifeval-browser-v1-heldout')
    parser.add_argument('--public',default='reports/ifeval-browser-v1-heldout')
    args=parser.parse_args()
    report(args.root,args.public)
