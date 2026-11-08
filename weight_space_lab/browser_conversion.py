"""Audit and finalize local MLC artifacts without inference or proposal dispatch."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash


def _save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def replay_source(root, model, history):
    """Replay the previously verified smoke proposal in a new bundle and journal."""
    from .checkpoint import CheckpointEditor, verify_candidate
    root, model, history = Path(root), Path(model), Path(history)
    if (root/'source-provenance.json').exists() or (root/'source-candidate-replay').exists():
        raise FileExistsError('Source replay must have a new provenance destination')
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    response = read_json(history/'manual/response.json')
    original_response = history.parent/'response.json'
    expected = read_json('reports/editor-session/manual-result.json')
    if file_sha256(original_response) != expected['supplied_response_sha256'] or read_json(original_response) != response:
        raise ValueError('Saved historical response does not match verified evidence')
    old_bundle = history/'candidates'/response['request_id']
    proposal = read_json(old_bundle/'proposal.json')
    if proposal != response['action']['arguments']['proposal']:
        raise ValueError('Saved proposal differs from verified response')
    policy = read_json('configs/candidate_policy.json')
    if policy != read_json(old_bundle/'policy.json'):
        raise ValueError('Historical source edit policy differs')
    original = inventory_model(model)
    if canonical_hash(original['files']) != proposal['base_sha256']:
        raise ValueError('Pristine checkpoint does not match historical proposal base')
    editor = CheckpointEditor(model, policy, exposure_log=root/'source-replay-exposures.jsonl')
    audit = editor.propose(proposal, root/'source-candidate-replay')
    if audit['result_sha256'] != expected['checkpoint']['result_sha256']:
        raise ValueError('Replayed candidate differs from historical expected identity')
    if verify_candidate(model, root/'source-candidate-replay', policy) != audit:
        raise ValueError('Independent source audit failed')
    provenance = {'schema_version':1, 'kind':'new_replay_of_verified_existing_smoke_proposal',
        'not_recovered_historical_file':True, 'old_import_slot_reopened':False,
        'new_proposals':0, 'inference_sequences':0,
        'original_files_sha256':canonical_hash(original['files']),
        'proposal_sha256':canonical_hash(proposal),
        'historical_response_sha256':canonical_hash(response),
        'historical_supplied_response_file_sha256':file_sha256(original_response),
        'historical_candidate_expected_sha256':expected['checkpoint']['result_sha256'],
        'replayed_candidate_sha256':audit['result_sha256'],
        'source_model':str(model.resolve()),
        'candidate_model':str((root/'source-candidate-replay/model').resolve()),
        'audit':audit, 'preparation_seconds':time.time()-started}
    _save(root/'source-provenance.json',provenance)
    _save(root/'source-inventory.json',original)
    return {'verified':True,'source_sha256':provenance['original_files_sha256'],
            'replayed_sha256':audit['result_sha256'],'target_generations':0,'new_proposals':0}


def tensor_index(root):
    """Validate shard bounds/digests and return immutable tensor storage records."""
    root = Path(root)
    manifest = read_json(root/'tensor-cache.json')
    records = {}
    for shard in manifest['records']:
        path = root/shard['dataPath']
        if path.parent.resolve() != root.resolve() or path.is_symlink():
            raise ValueError('Nonlocal MLC shard')
        raw = path.read_bytes()
        if len(raw) != shard['nbytes'] or hashlib.md5(raw).hexdigest() != shard['md5sum']:
            raise ValueError('MLC shard digest or size mismatch')
        intervals = []
        for record in shard['records']:
            name = record['name']
            if name in records:
                raise ValueError('Duplicate MLC tensor')
            start, end = record['byteOffset'], record['byteOffset'] + record['nbytes']
            if not 0 <= start <= end <= len(raw):
                raise ValueError('MLC tensor out of shard bounds')
            intervals.append((start, end))
            records[name] = {**record, 'dataPath':shard['dataPath'],
                             'stored_sha256':hashlib.sha256(raw[start:end]).hexdigest()}
        ordered = sorted(intervals)
        if ordered and (ordered[0][0] != 0 or ordered[-1][1] != len(raw)
                        or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:]))):
            raise ValueError('MLC shard contains gaps or overlaps')
    if len(records) != manifest['metadata']['ParamSize']:
        raise ValueError('MLC parameter count mismatch')
    return records


def load_tensor(root, record):
    path = Path(root)/record['dataPath']
    with path.open('rb') as f:
        f.seek(record['byteOffset'])
        raw = f.read(record['nbytes'])
    if record['dtype'] == 'float32' and record['format'] == 'f32-to-bf16':
        result = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    else:
        result = np.frombuffer(raw, dtype=np.dtype(record['dtype']).newbyteorder('<'))
    return result.reshape(record['shape'])


def decode_q4(weight, scale):
    """MLC q4f16_1: eight little-nibble unsigned codes per word, offset7/group32."""
    codes = ((weight[..., None] >> (np.arange(8, dtype=np.uint32)*4)) & 15).reshape(weight.shape[0], -1)
    return ((codes.astype(np.float16)-np.float16(7))*np.repeat(scale, 32, axis=-1)).astype(np.float16)


def audit_pair(baseline, candidate, wasm_metadata):
    base, edited = tensor_index(baseline), tensor_index(candidate)
    contract = {p['name']:{'shape':p['shape'], 'dtype':p['dtype']} for p in read_json(wasm_metadata)['params']}
    for observed in (base, edited):
        if {n:{'shape':p['shape'], 'dtype':p['dtype']} for n,p in observed.items()} != contract:
            raise ValueError('Artifact parameters differ from compiled library')
    changed = [name for name in base if base[name]['stored_sha256'] != edited[name]['stored_sha256']]
    allowed = {'model.layers.15.mlp.down_proj.q_weight', 'model.layers.15.mlp.down_proj.q_scale'}
    if not set(changed) <= allowed:
        raise ValueError('Unexpected deployed edit outside the source smoke tensor')
    prefix = 'model.layers.15.mlp.down_proj.'
    bweight, cweight = (load_tensor(root, index[prefix+'q_weight']) for root,index in ((baseline,base),(candidate,edited)))
    bscale, cscale = (load_tensor(root, index[prefix+'q_scale']) for root,index in ((baseline,base),(candidate,edited)))
    bdecoded, cdecoded = decode_q4(bweight,bscale), decode_q4(cweight,cscale)
    mask = bdecoded != cdecoded
    delta = cdecoded.astype(np.float64)-bdecoded.astype(np.float64)
    codes_b = ((bweight[...,None] >> (np.arange(8,dtype=np.uint32)*4))&15).reshape(bdecoded.shape)
    codes_c = ((cweight[...,None] >> (np.arange(8,dtype=np.uint32)*4))&15).reshape(cdecoded.shape)
    return {'schema_version':1, 'kind':'effective_deployment_difference',
        'quantization':'q4f16_1', 'group_size':32, 'weight_storage':'uint32;8 little-endian nibbles/word',
        'decode_rule':'float16((uint4 - 7) * float16_scale); group32 along input columns',
        'parameter_contract_verified':True, 'compiled_parameter_count':len(contract),
        'changed_tensor_names':changed, 'unchanged_tensor_count':len(base)-len(changed),
        'quantized_weight_words_changed':int(np.count_nonzero(bweight!=cweight)),
        'quantized_nibbles_changed':int(np.count_nonzero(codes_b!=codes_c)),
        'quantization_scales_changed':int(np.count_nonzero(bscale!=cscale)),
        'scale_max_abs_change':float(np.max(np.abs(cscale.astype(float)-bscale.astype(float)))),
        'decoded_values_changed':int(mask.sum()), 'decoded_max_abs_change':float(np.max(np.abs(delta))),
        'decoded_l2_change':float(np.linalg.norm(delta)),
        'decoded_changed_outside_columns_0_31':int(mask[:,32:].sum()),
        'conversion_erases_edit':not bool(mask.any()),
        'decoded_baseline_sha256':hashlib.sha256(bdecoded.tobytes()).hexdigest(),
        'decoded_candidate_sha256':hashlib.sha256(cdecoded.tobytes()).hexdigest(),
        'tensor_storage_identities':{arm:{n:p['stored_sha256'] for n,p in idx.items()} for arm,idx in [('baseline',base),('candidate',edited)]}}


def finalize(root):
    root = Path(root)
    started = time.time()
    pair = audit_pair(root/'baseline-q4-final', root/'candidate-q4-final', root/'wasm-metadata.json')
    provenance = read_json(root/'source-provenance.json')
    source = Path(provenance['source_model'])
    if canonical_hash(inventory_model(source)['files']) != provenance['original_files_sha256']:
        raise ValueError('Pristine source changed after replay')
    config = read_json(root/'reference-mlc-chat-config.json')
    config['context_window_size'] = config['model_config']['context_window_size'] = 4096
    config['prefill_chunk_size'] = config['model_config']['prefill_chunk_size'] = 1024
    config['max_history_size'] = 1
    config['temperature'] = 0
    config['top_p'] = 1
    config['presence_penalty'] = config['frequency_penalty'] = 0
    config['repetition_penalty'] = 1
    config['conv_template']['name'] = 'weight-space-qwen3_5-webllm-v1'
    config['conv_template']['stop_token_ids'] = [248046,248044]
    config['eos_token_id'] = 248044
    config['bos_token_id'] = None
    config['pad_token_id'] = 248044
    # Plain assistant header: WebLLM0.2.86 adds the closed empty thinking block.
    assert config['conv_template']['roles']['assistant'] == '<|im_start|>assistant'
    manifest = {'schema_version':1,'kind':'verified_paired_browser_artifacts','authors':['Arhaan2'],
        'model_id':'Qwen/Qwen3.5-0.8B', 'source_revision':'2fc06364715b967f1860aea9cf38778875588b17',
        'quantization':'q4f16_1','context_window_size':4096,'prefill_chunk_size':1024,
        'webllm_version':'0.2.86','model_library_revision':'00fe7a40e57e5e8eb9fef87526a4bf28ad5846c0',
        'model_library_sha256':file_sha256(root/'model.wasm'),
        'converter_mlc_revision':'9fa644f54b04983adea4d0168f49fc6af4a893ba',
        'converter_tvm_revision':'837cb9de1127b48ce48e4cefe09e83215b9d4ba7',
        'converter_packages':{'mlc-llm-nightly-cpu':'0.26.dev6','mlc-ai-nightly-cpu':'0.26.dev246','apache-tvm-ffi':'0.1.13'},
        'converter_adapter_sha256':file_sha256('scripts/browser_convert.py'),
        'source_scope':'full pristine checkpoint; MLC loader deploys text backbone only, excludes vision and MTP',
        'source_identities_separate_from_deployment':True,'public_prebuilt_baseline_substituted':False,
        'new_proposals':0,'inference_sequences':0,'arms':{}}
    deployed = root/'deployed'
    deployed.mkdir(exist_ok=True)
    for arm, path, source_sha in [('baseline',root/'baseline-q4-final',provenance['original_files_sha256']),
                                  ('candidate',root/'candidate-q4-final',provenance['replayed_candidate_sha256'])]:
        for name in config['tokenizer_files']:
            shutil.copyfile(source/name,path/name)
        shutil.copyfile(source/'LICENSE',path/'LICENSE')
        shutil.copyfile(root/'model.wasm',path/'model.wasm')
        _save(path/'mlc-chat-config.json',config)
        files = {p.name:{'sha256':file_sha256(p),'size_bytes':p.stat().st_size} for p in sorted(path.iterdir()) if p.is_file()}
        identity = canonical_hash(files)
        target = deployed/f'{arm}-{identity[:16]}'
        path.rename(target)
        data = {'model_id':f'wsl-qwen35-{arm}-{identity}', 'artifact_sha256':identity,
                'source_checkpoint_sha256':source_sha,'path':str(target.resolve()),'files':files,
                'cache_namespace':f'wsl-qwen35-{arm}-{identity}', 'tensor_cache_sha256':files['tensor-cache.json']['sha256']}
        manifest['arms'][arm] = data
        _save(target/'artifact-manifest.json',data)
    manifest['effective_edit'] = {k:v for k,v in pair.items() if k!='tensor_storage_identities'}
    manifest['finalization_seconds'] = time.time()-started
    manifest['conversion_receipts'] = {arm:read_json(root/(arm+'-q4-final-receipt.json')) for arm in ['baseline','candidate']}
    _save(root/'deployment-difference.json',pair)
    _save(root/'artifacts.json',manifest)
    return manifest


def verify(root):
    root = Path(root)
    manifest = read_json(root/'artifacts.json')
    for arm in manifest['arms'].values():
        path = Path(arm['path'])
        files = {p.name:{'sha256':file_sha256(p),'size_bytes':p.stat().st_size} for p in sorted(path.iterdir()) if p.name!='artifact-manifest.json'}
        if files != arm['files'] or canonical_hash(files) != arm['artifact_sha256']:
            raise ValueError('Deployment artifact identity changed')
        tensor_index(path)
    pair = audit_pair(manifest['arms']['baseline']['path'],manifest['arms']['candidate']['path'],root/'wasm-metadata.json')
    if pair != read_json(root/'deployment-difference.json'):
        raise ValueError('Effective deployment audit changed')
    return {'verified':True,'baseline':manifest['arms']['baseline']['artifact_sha256'],'candidate':manifest['arms']['candidate']['artifact_sha256'],'conversion_erases_edit':pair['conversion_erases_edit']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['replay-source','finalize','verify'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--model',default='artifacts/on-device-recovery-v1/model')
    parser.add_argument('--history',default='artifacts/astra-session-development/attempt')
    args=parser.parse_args()
    if args.command=='replay-source':
        print(json.dumps(replay_source(args.root,args.model,args.history),indent=2))
        return
    result=globals()[args.command](args.root)
    print(json.dumps(result if args.command=='verify' else {'prepared':True,'arms':{k:v['path'] for k,v in result['arms'].items()},'edit':result['effective_edit']},indent=2))

if __name__=='__main__':main()
