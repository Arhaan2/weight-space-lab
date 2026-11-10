#!/usr/bin/env python3
"""Execute the frozen matched held-out schedule once; never retry. Arhaan2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from weight_space_lab.ifeval_baseline import save
from weight_space_lab.ifeval_heldout_runner import execute, grade, load
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.native_runner import _lock
from weight_space_lab.protocol import canonical_hash


def run(root):
    root=Path(root).resolve()
    freeze=read_json(root/'evaluation-freeze.json')
    with _lock(root):
        if (root/'single-run-intent.json').exists():
            raise FileExistsError('This evaluation has already been started; automatic retries forbidden')
        for arm in freeze['execution_order']:
            load(root/arm)
        save(root/'single-run-intent.json',{'started_epoch_seconds':time.time(),
            'evaluation_freeze_sha256':canonical_hash(freeze), 'source_sha256':file_sha256(__file__),
            'execution_order':freeze['execution_order'],'automatic_retries':0})
    results,errors={},[]
    for arm in freeze['execution_order']:
        try:
            result=execute(root/arm)
            results[arm]={k:result[k] for k in ('status','error','engine_load_attempts',
                'generation_dispatches','scored_dispatches','warmup_dispatches','cleanup',
                'first_load_through_cleanup_seconds')}
            print(json.dumps({'arm':arm,'execution':results[arm]}),flush=True)
            if result['status']!='completed':
                break
        except BaseException as exc:
            errors.append({'arm':arm,'error':type(exc).__name__+': '+str(exc)})
            break
    # Grade only after generation ends; an unexecuted arm retains all slots.
    summaries={}
    for arm in freeze['execution_order']:
        try:
            summaries[arm]=grade(root/arm)
        except BaseException as exc:
            errors.append({'arm':arm,'grading_error':type(exc).__name__+': '+str(exc)})
    final={'status':'completed' if len(results)==2 and all(r['status']=='completed' for r in results.values())
           and len(summaries)==2 and all(s['technically_valid'] for s in summaries.values()) and not errors else 'incomplete',
           'execution':results,'grading':summaries,'errors':errors,'automatic_retries':0,
           'finished_epoch_seconds':time.time(),'evaluation_freeze_sha256':canonical_hash(freeze)}
    save(root/'run-result.json',final)
    print(json.dumps(final,indent=2),flush=True)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='artifacts/ifeval-browser-v1-heldout')
    args=parser.parse_args()
    run(args.root)
