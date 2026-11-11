#!/usr/bin/env python3
"""Stage exact existing inputs away from Desktop; never run a model or change old data."""
from __future__ import annotations
import argparse, base64, csv, email, hashlib, io, json, os, shutil, subprocess, time, zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PINS = {'numpy':'2.3.5','absl-py':'2.5.1','langdetect':'1.0.9','nltk':'3.10.4',
    'immutabledict':'4.3.1','six':'1.17.0','defusedxml':'0.7.1','click':'8.5.1',
    'joblib':'1.6.0','regex':'2026.11.7','tqdm':'4.70.1','cloudpickle':'3.1.2'}

def sha(data): return hashlib.sha256(data).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def norm(s): return s.lower().replace('_','-').replace('.','-')

class Copier:
    def __init__(self, workspace):
        self.workspace=workspace; self.cache={}; self.rows=[]
        for name in ('resource-all-edits-v1-payload-recovery.json','resource-all-edits-v1-frozen-input-recovery.json'):
            r=json.loads((workspace/'artifacts'/name).read_text())
            blobs={x['sha256']:x['cache_file'] for x in r['unique_payloads']}
            for x in r['files']: self.cache[x['path']]=(x['sha256'],Path(blobs[x['sha256']]))
    def read(self,path,expected=None):
        path=Path(path); source=path; method='ordinary existing bytes'
        cached=self.cache.get(str(path))
        if cached and (not expected or expected==cached[0]):
            expected=cached[0];source=cached[1];method='previous exact-byte cache'
        if source.stat().st_flags&0x40000000:
            command='import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read())'
            data=subprocess.run(['/opt/homebrew/opt/python@3.13/bin/python3.13','-S','-c',command,str(source)],capture_output=True,check=True,timeout=25).stdout
        else: data=source.read_bytes()
        digest=sha(data)
        if expected and digest!=expected: raise ValueError('Input mismatch: '+str(path))
        return data,digest,method
    def copy(self,path,destination,expected=None):
        data,digest,method=self.read(path,expected)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        if sha(destination.read_bytes())!=digest: raise ValueError('Copy mismatch')
        try: destination.chmod(Path(path).stat().st_mode&0o777)
        except FileNotFoundError: pass
        self.rows.append({'original':str(path),'staged':str(destination),'sha256':digest,'size_bytes':len(data),'method':method})
        return digest

def wheel_info(path):
    try:
        with zipfile.ZipFile(path) as z:
            meta=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
            m=email.message_from_bytes(z.read(meta));name=norm(m['Name']);version=m['Version']
            wheel=z.read(meta.rsplit('/',1)[0]+'/WHEEL').decode()
            tags=[line[5:] for line in wheel.splitlines() if line.startswith('Tag: ')]
            compatible=any((tag.split('-')[0].startswith('py') or tag.split('-')[0]=='cp313') and
                (tag.endswith('-any') or tag.endswith('_arm64') or tag.endswith('_universal2')) for tag in tags)
            return name,version,compatible
    except (zipfile.BadZipFile,StopIteration,KeyError):return None

def stage_python(workspace,root):
    recovery=json.loads((workspace/'artifacts/ifeval-browser-v2-collaboration-continuation/grading-recovery/environment-restoration.json').read_text())
    found={norm(x['distribution']):Path(x['source_archive']) for x in recovery['restored_files'] if norm(x['distribution']) in PINS}
    for path in list((Path.home()/'Library/Caches/pip/http-v2').rglob('*.body'))+list((Path.home()/'Library/Caches/pip/wheels').rglob('*.whl')):
        if len(found)==len(PINS):break
        info=wheel_info(path)
        if info and info[0] not in found and PINS.get(info[0])==info[1] and info[2]:found[info[0]]=path
    if set(found)!=set(PINS):raise ValueError('Missing pinned cached wheels: '+str(set(PINS)-set(found)))
    target=root/'dependencies/python';target.mkdir(parents=True)
    rows=[];archives=[]
    for name,path in sorted(found.items()):
        info=wheel_info(path)
        if info!=(name,PINS[name],True):raise ValueError('Cached wheel identity/ABI changed: '+name)
        payload=path.read_bytes();archives.append({'distribution':name,'version':PINS[name],'source_archive':str(path),'sha256':sha(payload),'bytes':len(payload)})
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            record=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
            expected={row[0]:row[1] for row in csv.reader(io.StringIO(z.read(record).decode()))}
            for member in z.namelist():
                if member.endswith('/'):continue
                relative=Path(member)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe wheel path')
                data=z.read(member);digest=sha(data);entry=expected.get(member)
                if entry:
                    algorithm,encoded=entry.split('=',1)
                    if algorithm!='sha256' or base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')!=encoded:raise ValueError('Wheel RECORD mismatch')
                elif member!=record:raise ValueError('Unhashed wheel member')
                out=target/member;out.parent.mkdir(parents=True,exist_ok=True)
                if out.exists() and out.read_bytes()!=data:raise ValueError('Overlapping wheels differ')
                out.write_bytes(data)
                rows.append({'path':str(out),'sha256':digest,'size_bytes':len(data),'distribution':name,'wheel_record_verified':bool(entry)})
    subprocess.run(['/opt/homebrew/opt/python@3.13/bin/python3.13','-m','venv','--without-pip',str(root/'python')],check=True,timeout=30)
    # Trusted checkpoint workers use -I, so PYTHONPATH alone is insufficient.
    (root/'python/lib/python3.13/site-packages/frozen-dependencies.pth').write_text(str(target)+'\n')
    save(root/'python-dependencies.json',{'pins':PINS,'archives':archives,'files':rows,'network_requests':0,'installed_environment_modified':False})
    return rows

def stage(workspace, root):
    began=time.monotonic();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    save(root/'preparation-intent.json',{'started_epoch_seconds':time.time(),'target_calls':0,'local_seconds_limit':14400})
    copier=Copier(workspace)
    prior_path=workspace/'artifacts/resource-all-edits-v1/study.json'
    prior=json.loads(copier.read(prior_path)[0]);old_root=prior_path.parent
    prior_sha=copier.copy(prior_path,root/'inputs/previous-resource-study.json')
    copier.copy(PROJECT/'reports/resource-all-edits-v1/result.json',root/'inputs/historical-resource-result.json')
    old_base=Path(prior['baseline']['baseline_root']);baseline=root/'baseline';baseline.mkdir()
    for relative in ('baseline-config.json','manifest.json','data/development-items.json','weight-information/inspections.json','weight-information/policy.json','weight-information/exposures.jsonl','weight-information/host.json','handoff/request.json'):
        path=old_base/relative
        expected=prior['baseline'].get('files',{}).get(str(path))
        copier.copy(path,baseline/relative,expected)
    official=baseline/'official-ifeval';old_official=Path(prior['baseline']['official_ifeval_root'])
    for relative,digest in prior['baseline']['official_assets']['files'].items():copier.copy(old_official/relative,official/relative,digest)
    archive=old_official/'punkt_tab.zip';data,digest,_=copier.read(archive,prior['baseline']['official_assets']['nltk_archive_sha256'])
    (official/'punkt_tab.zip').write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            if member.endswith('/') or not member.startswith('punkt_tab/english/'):continue
            path=Path(member)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe tokenizer archive')
            dest=official/'nltk_data/tokenizers'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(member))
    benchmark=root/'benchmark';benchmark.mkdir()
    old_benchmark=old_root/'benchmark'
    for path,digest in prior['files'].items():
        path=Path(path)
        if path.is_relative_to(old_benchmark):copier.copy(path,benchmark/path.relative_to(old_benchmark),digest)
    modules=(old_benchmark/'node_modules').resolve()
    shutil.copytree(modules,benchmark/'node_modules',symlinks=True)
    # All npm dependency bytes are copied locally, never linked back to Desktop.
    for p in (benchmark/'node_modules').rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(benchmark/'node_modules'):raise ValueError('External npm symlink: '+str(p))
    artifacts={}
    for key,a in prior['artifacts'].items():
        target=root/'model-payloads'/key;target.mkdir(parents=True)
        for name,value in a['files'].items():copier.copy(Path(a['path'])/name,target/name,value['sha256'])
        artifacts[key]={**a,'path':str(target),'original_path':a['path']}
        # Baseline URLs are unchanged; map their exact old URL paths to staged bytes.
    config=json.loads((baseline/'baseline-config.json').read_text())
    from urllib.parse import urlparse
    for asset in config['assets']:
        relative=urlparse(asset['url']).path.lstrip('/');dest=benchmark/'public'/relative
        source=root/'model-payloads/pristine'/Path(relative).name
        if not source.exists():raise ValueError('Unknown baseline asset: '+relative)
        if sha(source.read_bytes())!=asset['sha256']:raise ValueError('Baseline asset mismatch')
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
    pyrows=stage_python(workspace,root)
    baseline_info={**prior['baseline'],'baseline_root':str(baseline),'official_ifeval_root':str(official)}
    baseline_info.pop('files',None)
    staged={'schema_version':1,'baseline':baseline_info,'artifacts':artifacts,'previous_resource_study_sha256':prior_sha,
        'original_inputs':copier.rows,'python_dependency_manifest_sha256':sha((root/'python-dependencies.json').read_bytes()),
        'target_calls':0,'network_requests':0,'charged_preparation_seconds':time.monotonic()-began}
    files={str(p):sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and not p.is_symlink() and
        (not p.is_relative_to(root/'python') or p.suffix=='.pth' or p.name=='pyvenv.cfg') and p.name not in ('preparation-intent.json',)}
    staged['files']=files
    staged['charged_preparation_seconds']=time.monotonic()-began
    save(root/'staged-inputs.json',staged)
    print(json.dumps({'status':'staged','root':str(root),'files':len(files),'bytes':sum(Path(p).stat().st_size for p in files),
        'seconds':staged['charged_preparation_seconds'],'target_calls':0}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();stage(args.workspace.resolve(),args.root.resolve())
