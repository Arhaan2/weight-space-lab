"""Prepare a future validated pilot candidate's browser artifact without inference.

Separate from the frozen integration-fixture implementation. Conversion uses the
same pinned converter wrapper; packaging copies baseline deployment controls and
reports quantized/decoded differences for the existing allowlisted tensors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .browser_conversion import tensor_index, load_tensor, decode_q4
from .checkpoint import verify_candidate
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _files(root):
    root = Path(root)
    if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        raise ValueError('Deployment directory must contain only regular files')
    return {p.name:{'sha256':file_sha256(p),'size_bytes':p.stat().st_size}
            for p in sorted(root.iterdir()) if p.name!='artifact-manifest.json'}


def verify_baseline(manifest_path):
    manifest = read_json(manifest_path)
    if manifest['quantization'] != 'q4f16_1' or manifest['webllm_version'] != '0.2.86':
        raise ValueError('Unsupported baseline deployment contract')
    baseline = manifest['arms']['baseline']
    files = _files(baseline['path'])
    if files != baseline['files'] or canonical_hash(files) != baseline['artifact_sha256']:
        raise ValueError('Frozen baseline artifact changed')
    tensor_index(baseline['path'])
    return manifest


def source_identity(original, candidate, manifest, *, policy_path='configs/candidate_policy.json'):
    policy = read_json(policy_path)
    audit = verify_candidate(original, candidate, policy)
    if audit['base_sha256'] != manifest['arms']['baseline']['source_checkpoint_sha256']:
        raise ValueError('Candidate must start from this deployment baseline pristine source')
    proposal = read_json(Path(candidate)/'proposal.json')
    if proposal['provenance'] != 'editor_proposal':
        raise ValueError('Pilot candidate must be the original accepted editor proposal')
    return {'source_checkpoint_sha256':audit['result_sha256'],
            'source_original_sha256':audit['base_sha256'],
            'source_audit_sha256':canonical_hash(audit),
            'source_proposal_sha256':canonical_hash(proposal),
            'source_policy_sha256':canonical_hash(policy)}, audit, proposal


def convert(original, candidate, baseline_manifest, output, converter_python, *, policy_path='configs/candidate_policy.json'):
    """Validate, convert once locally, then bind stable source and result hashes."""
    started = time.monotonic()
    original, candidate, output = Path(original), Path(candidate), Path(output)
    manifest = verify_baseline(baseline_manifest)
    identity, _, _ = source_identity(original,candidate,manifest,policy_path=policy_path)
    script = Path('scripts/browser_convert.py').resolve()
    if file_sha256(script) != manifest['converter_adapter_sha256']:
        raise ValueError('Converter differs from baseline conversion adapter')
    if output.exists() or output.with_name(output.name+'-source-binding.json').exists():
        raise FileExistsError('Conversion destination and receipt must be new')
    output.parent.mkdir(parents=True,exist_ok=True)
    root = Path(baseline_manifest).parent
    expected_packages = manifest['converter_packages']
    installed = subprocess.run([str(Path(converter_python).resolve()),'-c',
        'import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in '+repr(list(expected_packages))+'}))'],
        check=True,capture_output=True,text=True,timeout=30)
    if json.loads(installed.stdout) != expected_packages:
        raise ValueError('Converter environment differs from baseline pinned packages')
    command = [str(Path(converter_python).resolve()),str(script),'--source',str((candidate/'model').resolve()),
        '--output',str(output.resolve()),'--wasm-metadata',str((root/'wasm-metadata.json').resolve())]
    env = dict(os.environ,TVM_BACKTRACE='1',MLC_LLM_HOME=str((root/'cache').resolve()),
        PYTHONPYCACHEPREFIX='/tmp/wsl-browser-convert',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TVM_NUM_THREADS='4')
    start_record = {'schema_version':1,'kind':'future_candidate_conversion_preparation',
        **identity,'baseline_manifest_sha256':file_sha256(baseline_manifest),
        'command':command,'converter_packages':expected_packages,'target_generations':0,'new_proposals':0}
    _save(output.with_name(output.name+'-conversion-start.json'),start_record)
    with output.with_name(output.name+'-convert.stdout.txt').open('x') as stdout, output.with_name(output.name+'-convert.stderr.txt').open('x') as stderr:
        result = subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=1800)
    if result.returncode:
        _save(output.with_name(output.name+'-conversion-failure.json'),
              {'returncode':result.returncode,'preparation_seconds':time.monotonic()-started,'target_generations':0})
        raise RuntimeError('Conversion failed; preserved logs are preparation evidence only')
    after, _, _ = source_identity(original,candidate,manifest,policy_path=policy_path)
    if after != identity:
        raise ValueError('Source candidate changed during conversion')
    files = _files(output)
    tensor_index(output)
    receipt = {**start_record,'converted_files':files,'converted_files_sha256':canonical_hash(files),
        'preparation_seconds':time.monotonic()-started,'status':'converted_source_bound'}
    _save(output.with_name(output.name+'-source-binding.json'),receipt)
    return {'converted':str(output.resolve()),'source_binding_receipt':str(output.with_name(output.name+'-source-binding.json').resolve()),
            'source_checkpoint_sha256':identity['source_checkpoint_sha256'],'target_generations':0,'new_proposals':0}


def audit_effective_pair(baseline, candidate, proposal, allowed_tensors):
    """Compare all storage; decode each affected group and disclose group spillover."""
    base, edited = tensor_index(baseline), tensor_index(candidate)
    shapes = lambda idx:{n:(p['shape'],p['dtype']) for n,p in idx.items()}
    if shapes(base) != shapes(edited):
        raise ValueError('Candidate tensor contract differs from baseline')
    selected = {}
    for op in proposal['operations']:
        source_name = op['tensor']
        if source_name not in allowed_tensors or not source_name.endswith('.mlp.down_proj.weight'):
            raise ValueError('Unsupported source-to-MLC tensor mapping')
        selected.setdefault(source_name,set()).update(op['columns'])
    mapping = {name:name.replace('model.language_model.','model.',1).removesuffix('.weight') for name in selected}
    allowed_deployed = {prefix+suffix for prefix in mapping.values() for suffix in ('.q_weight','.q_scale')}
    changed = [n for n in base if base[n]['stored_sha256'] != edited[n]['stored_sha256']]
    if not set(changed) <= allowed_deployed:
        raise ValueError('Deployment changed tensors not targeted by the validated source proposal')
    rows = {}
    for name,prefix in mapping.items():
        bw,cw = (load_tensor(root,idx[prefix+'.q_weight']) for root,idx in ((baseline,base),(candidate,edited)))
        bs,cs = (load_tensor(root,idx[prefix+'.q_scale']) for root,idx in ((baseline,base),(candidate,edited)))
        bd,cd = decode_q4(bw,bs),decode_q4(cw,cs)
        target = np.zeros(bd.shape[1],dtype=bool);target[list(selected[name])] = True
        groups = np.repeat(target.reshape(-1,32).any(axis=1),32)
        mask = bd!=cd
        if bool(mask[:,~groups].any()):
            raise ValueError('Decoded edit spills outside source-targeted quantization groups')
        nibble = lambda w:((w[...,None] >> (np.arange(8,dtype=np.uint32)*4))&15).reshape(bd.shape)
        delta=cd.astype(np.float64)-bd.astype(np.float64)
        rows[name]={'deployed_tensor_prefix':prefix,'target_columns':sorted(selected[name]),
            'source_targeted_quantization_groups':sorted({c//32 for c in selected[name]}),
            'quantized_weight_words_changed':int(np.count_nonzero(bw!=cw)),
            'quantized_nibbles_changed':int(np.count_nonzero(nibble(bw)!=nibble(cw))),
            'quantization_scales_changed':int(np.count_nonzero(bs!=cs)),
            'scale_max_abs_change':float(np.max(np.abs(cs.astype(float)-bs.astype(float)))),
            'decoded_values_changed':int(mask.sum()),
            'decoded_changed_outside_source_target_columns':int(mask[:,~target].sum()),
            'decoded_changed_outside_source_target_groups':int(mask[:,~groups].sum()),
            'decoded_max_abs_change':float(np.max(np.abs(delta))),
            'decoded_l2_change':float(np.linalg.norm(delta)),
            'decoded_baseline_sha256':hashlib.sha256(bd.tobytes()).hexdigest(),
            'decoded_candidate_sha256':hashlib.sha256(cd.tobytes()).hexdigest()}
    return {'schema_version':1,'kind':'validated_pilot_candidate_deployment_audit',
        'quantization':'q4f16_1','group_size':32,'parameter_contract_verified':True,
        'parameter_count':len(base),'changed_tensor_names':changed,
        'unchanged_tensor_count':len(base)-len(changed),'per_source_tensor':rows,
        'conversion_erases_edit':not any(row['decoded_values_changed'] for row in rows.values()),
        'tensor_storage_identities':{arm:{n:p['stored_sha256'] for n,p in idx.items()}
                                     for arm,idx in [('baseline',base),('candidate',edited)]}}


def package_candidate(original,candidate,baseline_manifest,converted,output, *, policy_path='configs/candidate_policy.json'):
    started=time.monotonic()
    original,candidate,converted,output=map(Path,(original,candidate,converted,output))
    if output.exists():raise FileExistsError('Candidate package output must be new')
    manifest=verify_baseline(baseline_manifest)
    identity,audit,proposal=source_identity(original,candidate,manifest,policy_path=policy_path)
    binding=read_json(converted.with_name(converted.name+'-source-binding.json'))
    if (binding.get('status')!='converted_source_bound' or
        binding['baseline_manifest_sha256']!=file_sha256(baseline_manifest) or
        any(binding.get(k)!=v for k,v in identity.items()) or
        binding['converted_files']!=_files(converted) or
        binding['converted_files_sha256']!=canonical_hash(_files(converted))):
        raise ValueError('Converted candidate is not bound to validated source and frozen baseline')
    policy=read_json(policy_path)
    effective=audit_effective_pair(manifest['arms']['baseline']['path'],converted,proposal,policy['allowed_tensors'])
    baseline=Path(manifest['arms']['baseline']['path'])
    output.mkdir(parents=True)
    stage=output/'candidate-staging'
    shutil.copytree(converted,stage)
    # Identical effective prompt, tokenizer, quantization config and compiled library.
    config=read_json(baseline/'mlc-chat-config.json')
    metadata=['mlc-chat-config.json','model.wasm','LICENSE',*config['tokenizer_files']]
    for name in metadata:
        if name not in manifest['arms']['baseline']['files'] or Path(name).name!=name:
            raise ValueError('Unsafe or unbound baseline metadata asset')
        shutil.copyfile(baseline/name,stage/name)
    files=_files(stage)
    artifact_sha=canonical_hash(files)
    destination=output/f'candidate-{artifact_sha[:16]}'
    stage.rename(destination)
    artifact={'schema_version':1,'model_id':f'wsl-qwen35-pilot-candidate-{artifact_sha}',
        'artifact_sha256':artifact_sha,'source_checkpoint_sha256':identity['source_checkpoint_sha256'],
        'path':str(destination.resolve()),'files':files,
        'cache_namespace':f'wsl-qwen35-pilot-candidate-{artifact_sha}',
        'tensor_cache_sha256':files['tensor-cache.json']['sha256'],
        'baseline_artifact_sha256':manifest['arms']['baseline']['artifact_sha256'],
        'source_identity':identity,'conversion_binding_sha256':canonical_hash(binding),
        'effective_deployment_audit_sha256':canonical_hash(effective),
        'shared_runtime_assets':{name:files[name]['sha256'] for name in metadata},
        'preparation_seconds':time.monotonic()-started,'new_proposals':0,'target_generations':0,
        'provenance':'original future manual proposal, already accepted and independently validated; not smoke candidate'}
    _save(destination/'artifact-manifest.json',artifact)
    _save(output/'candidate-artifact.json',artifact)
    _save(output/'source-audit.json',audit)
    _save(output/'deployment-difference.json',effective)
    return {'artifact_manifest':str((output/'candidate-artifact.json').resolve()),
        'artifact_path':artifact['path'],'artifact_sha256':artifact_sha,
        'conversion_erases_edit':effective['conversion_erases_edit'], 'target_generations':0,'new_proposals':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['convert','package-candidate'])
    parser.add_argument('--source-original',required=True)
    parser.add_argument('--source-candidate',required=True,help='Validated complete candidate bundle, parent of model/')
    parser.add_argument('--baseline-manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--converter-python')
    parser.add_argument('--converted')
    parser.add_argument('--policy',default='configs/candidate_policy.json',help='Frozen source candidate intervention policy')
    args=parser.parse_args()
    if args.command=='convert':
        if not args.converter_python:parser.error('--converter-python is required for convert')
        result=convert(args.source_original,args.source_candidate,args.baseline_manifest,args.output,args.converter_python,policy_path=args.policy)
    else:
        if not args.converted:parser.error('--converted is required for package-candidate')
        result=package_candidate(args.source_original,args.source_candidate,args.baseline_manifest,args.converted,args.output,policy_path=args.policy)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
