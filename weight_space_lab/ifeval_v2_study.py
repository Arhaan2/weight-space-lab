"""Frozen, manually operated four-round V2 study; preparation never executes models.

Arhaan2. Private state and raw evidence are append-only or atomically replaced.
V1 evidence is only read. External editor selection is procedural and unverified.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from urllib.parse import urlparse
from .inventory import file_sha256, read_json
from .protocol import canonical_hash
from .ifeval_baseline import SETTINGS, POLICY, save
from .ifeval_handoff import baseline_feedback, aggregate_feedback, validate_response, MAX_RESPONSE_BYTES, _identity_only
from .editor_contract import read_json_bounded
from .native_runner import _lock
from .grading import grader_identity, verify_ifeval_assets

LIMITS = {"automatic_retries":0,"browser_seconds":10800,"calendar_seconds":604800,
    "conversions":4,"dispatches":634,"editor_submissions":4,"engine_loads":10,
    "generated_tokens":408736,"imports":4,"local_seconds":14400,
    "minimum_free_bytes":8*1024**3,"packages":4,"retained_bytes":32*1024**3,
    "scored_requests":624,"task_requests":384,"resource_requests":240,"warmups":10}
STAGES = {"analysis_export":600,"browser_development":3600,"browser_final":5400,"browser_resource":1800,
    "browser_preparation":120,"conversions":1200,"converter_prechecks":120,"grading":720,"imports":600,"packaging":240}
STUDY_ID = "ifeval-browser-v2-narrowed"
BASE_SOURCE = "f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d"
BASE_ARTIFACT = "5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0"
DEV_SHA = "9aa732f4c0dbe06248a93c329299f3fb732a5f148447d3be5a057c93e5fae627"
TERMINAL = {"completed","failed","invalid","duplicate","skipped"}
CLEANUP = ("unload_confirmed","worker_terminated","chrome_exit_confirmed", "local_harness_exit_confirmed","controller_exit_verified","chrome_os_exit_verified")


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def retained_bytes(root):
    """Only this V2 tree, count each inode once; never traverse external links."""
    seen=set(); total=0
    for p in Path(root).rglob("*"):
        if p.is_symlink() or not p.is_file(): continue
        s=p.stat(); key=(s.st_dev,s.st_ino)
        if key not in seen: seen.add(key); total += s.st_size
    return total


def validate_config(config):
    schedule = config["schedule"]
    if (config["study_id"] != STUDY_ID or config["settings"] != SETTINGS
            or config["grading_policy"] != POLICY or config["limits"] != LIMITS
            or config["stage_seconds"] != STAGES
            or config.get("calendar_anchor") != "first_production_editor_packet_export_once"
            or config["editor"]["max_submissions"] != 4 or config["editor"]["max_response_bytes"] != MAX_RESPONSE_BYTES
            or schedule["rounds"] != 4 or schedule["development_order"] != [["astra"]]*4
            or schedule["development_arms_per_round"] != ["astra"]
            or schedule["final_arms"] != ["pristine_baseline","selected_astra"]
            or schedule["resource_order"] != ["pristine_baseline","selected_astra","selected_astra","pristine_baseline"]
            or schedule["resource_requests_per_session"] != 60
            or type(config["final_set"]["prompts"]) is not int or not 0 <= config["final_set"]["prompts"] <= 128):
        raise ValueError("Narrowed V2 frozen schedule or budget differs")


def calendar_deadline(state):
    anchor=state.get("first_packet_export_epoch_seconds")
    return None if anchor is None else anchor+LIMITS["calendar_seconds"]


def effective_capacity(study):
    """Final-data shortfalls expire unused capacity; maxima never transfer stages."""
    n=study["config"]["final_set"]["prompts"]
    loads=8+(2 if n else 0); task_requests=128+2*n; resource_requests=240
    return {"editor_submissions":4,"imports":4,"conversions":4,"packages":4,
        "engine_loads":loads,"warmups":loads,"task_requests":task_requests,
        "resource_requests":resource_requests,"scored_requests":task_requests+resource_requests,
        "dispatches":task_requests+resource_requests+loads,
        "generated_tokens":131136+15424+(2*n*1024+32 if n else 0),
        "browser_seconds":3600+1800+(5400 if n else 0),
        "local_seconds":min(14400,3600+1800+(5400 if n else 0)+3600),
        "calendar_seconds":604800,"retained_bytes":32*1024**3,"automatic_retries":0}


def begin_session_capacity(root, study, state, key, *, workload, requests, max_tokens):
    """Reserve one nonreplaceable session's entire capacity before its load ACK."""
    expected={"development":(4,32,1024),"final":(2,study["config"]["final_set"]["prompts"],1024),"resource":(4,60,64)}
    if workload not in expected or (requests,max_tokens) != expected[workload][1:] or requests <= 0:
        raise ValueError("Session differs from frozen workload capacity")
    rows=state.setdefault("generation_sessions",{})
    if key in rows: raise ValueError("Session load capacity already consumed; no retries")
    if sum(r["workload"]==workload for r in rows.values()) >= expected[workload][0]:
        raise ValueError("Workload session capacity exhausted")
    record={"workload":workload,"loads":1,"warmups":1,"requests":requests,
        "dispatches":requests+1,"reserved_tokens":requests*max_tokens+16}
    totals=[*rows.values(),record]; capacity=effective_capacity(study)
    if (sum(r["loads"] for r in totals)>capacity["engine_loads"]
            or sum(r["dispatches"] for r in totals)>capacity["dispatches"]
            or sum(r["reserved_tokens"] for r in totals)>capacity["generated_tokens"]):
        raise ValueError("Global generation capacity exhausted")
    rows[key]=record; atomic(root/"state.json",state)
    return record


def baseline_binding(root):
    root=Path(root).resolve(); manifest=read_json(root/"manifest.json")
    config=read_json(root/"baseline-config.json"); items=read_json(root/"data/development-items.json")
    envelope=read_json(root/"baseline/graded.json")
    if (manifest["settings"] != SETTINGS or config["settings"] != SETTINGS or manifest["policy"] != POLICY
            or manifest["grader"] != grader_identity("ifeval") or manifest["checkpoint"]["sha256"] != BASE_SOURCE
            or config["sourceCheckpointSha256"] != BASE_SOURCE or config["convertedArtifactSha256"] != BASE_ARTIFACT
            or canonical_hash(items) != DEV_SHA or envelope["summary"]["technically_valid"] is not True
            or envelope["manifest_sha256"] != canonical_hash(manifest)
            or verify_ifeval_assets(manifest["official_ifeval_root"]) != manifest["official_assets"]):
        raise ValueError("Saved development baseline model, data, runtime, or scorer differs")
    from .browser_candidate import verify_baseline
    from .inventory import inventory_model
    conversion=Path("artifacts/browser-integration-v1/conversion/artifacts.json").resolve()
    deployed=verify_baseline(conversion)
    host=read_json(root/"weight-information/host.json")
    source_files=inventory_model(host["model_path"])["files"]
    if (canonical_hash(source_files) != BASE_SOURCE or source_files != host["source_files"]
            or deployed["arms"]["baseline"]["artifact_sha256"] != BASE_ARTIFACT):
        raise ValueError("Actual pristine source or deployed model bytes differ")
    feedback=baseline_feedback(items,envelope["records"],envelope["grades"],POLICY)
    if sum(row["strict_prompt_compliant"] for row in feedback) != 16:
        raise ValueError("Saved baseline score changed")
    # Verify generation, extraction, grading and deployed runtime bytes, without
    # requiring unchanged orchestration or unrelated native implementations.
    required={"grading.py","grading_worker.py","browser_tasks.py","records.py","browser_conversion.py","configs/grading-requirements.txt"}
    files={str(conversion):file_sha256(conversion)}
    for name,digest in manifest["files"].items():
        p=Path(name)
        if "/benchmark/" in name or p.name in required or name.endswith("configs/grading-requirements.txt"):
            if file_sha256(p) != digest: raise ValueError("Frozen baseline runtime/grader bytes changed: "+p.name)
            files[name]=digest
    for relative in ("manifest.json","baseline-config.json","baseline/graded.json","grading-policy.json","data/development-items.json","weight-information/inspections.json","weight-information/policy.json","weight-information/exposures.jsonl","weight-information/host.json","handoff/request.json"):
        files[str(root/relative)]=file_sha256(root/relative)
    return {"files":files,"baseline_root":str(root),"source_sha256":BASE_SOURCE,"artifact_sha256":BASE_ARTIFACT,
        "development_sha256":DEV_SHA,"strict_passes":16,"scored_output_tokens":sum(r["token_counts"]["output"] for r in envelope["records"]),"grader":manifest["grader"],"official_ifeval_root":manifest["official_ifeval_root"],
        "official_assets":manifest["official_assets"]}


def prepare(root, baseline_root, evaluator_root, config_path, protocol_path, *, authorization=None):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (root/"study.json").exists(): raise FileExistsError("Study already frozen")
    config=read_json(config_path); validate_config(config)
    binding=baseline_binding(baseline_root)
    from .ifeval_v2_dataset import verify_frozen
    evaluator_root=Path(evaluator_root).resolve(); final=verify_frozen(evaluator_root)
    if (final["sha256"] != config["final_set"]["manifest_sha256"]
            or any(final["counts"][key]!=config["final_set"][key] for key in ("prompts","instructions"))
            or config["schedule"]["final_prompts_per_arm"]!=final["counts"]["prompts"]
            or config["schedule"]["final_instructions_per_arm"]!=final["counts"]["instructions"]):
        raise ValueError("Final-set manifest or actual counts differ from frozen config")
    files={str(Path(config_path).resolve()):file_sha256(config_path),str(Path(protocol_path).resolve()):file_sha256(protocol_path),
           str(evaluator_root/"manifest.json"):file_sha256(evaluator_root/"manifest.json"),**binding["files"]}
    files.update({str(p.resolve()):file_sha256(p) for p in Path(__file__).parent.glob("*.py")})
    for relative in ("scripts/browser_convert.py","scripts/run_ifeval_v2.py", "integrations/browser-resource-comparison.patch", "configs/browser_resource_comparison.json"):
        path=Path(relative).resolve()
        if path.exists(): files[str(path)]=file_sha256(path)
    historical_benchmark=Path("artifacts/ifeval-browser-v1-heldout-continuation/benchmark").resolve()
    benchmark=prepare_benchmark(root,{"benchmark_source":str(historical_benchmark),"baseline":binding})
    from .browser_resource import install_adapter,inspect_preset
    install_adapter(benchmark)
    subprocess.run(["node","scripts/build-worker.mjs"],cwd=benchmark,check=True,capture_output=True,timeout=120)
    resource_preset=inspect_preset(benchmark)
    for p in benchmark.rglob("*"):
        if p.is_file() and not p.is_symlink() and p.suffix not in (".pyc",".pyo") and not any(part in p.parts for part in ("node_modules","models",".git","__pycache__")):
            files[str(p)]=file_sha256(p)
    # Preserve exact current source independently of publication commits. Some
    # retained helper modules contain pre-existing local work outside this change.
    snapshot=root/"source-snapshot"; snapshot.mkdir(exist_ok=True,mode=0o700)
    source_index={}
    repository=Path(__file__).resolve().parents[1]
    for name,digest in list(files.items()):
        path=Path(name)
        try: relative=path.relative_to(repository)
        except ValueError: continue
        if relative.parts[0] not in ("weight_space_lab","scripts","configs","docs","integrations"): continue
        target=snapshot/relative; target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        if target.exists():
            if file_sha256(target)!=digest: raise ValueError("Existing immutable source snapshot differs")
        else: shutil.copyfile(path,target); target.chmod(0o400)
        if file_sha256(target)!=digest: raise ValueError("Source changed while snapshotting")
        source_index[str(relative)]={"sha256":digest,"snapshot":str(target)}
        files[str(target)]=digest
    snapshot_index=snapshot/"index.json"
    if snapshot_index.exists():
        if read_json(snapshot_index)!=source_index: raise ValueError("Existing source index differs")
    else: save(snapshot_index,source_index)
    files[str(snapshot_index)]=file_sha256(snapshot_index)
    study={"source_snapshot_sha256":canonical_hash(source_index),
        "source_publication_note":"Exact local runtime sources are privately snapshotted; pre-existing unrelated source changes remain outside workflow commits.",
        "benchmark_source":str(benchmark),"historical_benchmark_source":str(historical_benchmark),"resource_preset":resource_preset,"schema_version":2,"kind":"frozen_ifeval_browser_quality_resource","config":config,"protocol_sha256":file_sha256(protocol_path),
        "baseline":binding,"evaluator_root":str(evaluator_root),"files":files,
        "custody":"procedural separation; technical isolation unverified","execution_authorized_by_preparation":False}
    save(root/"study.json",study); save(root/"ready.json",{"study_sha256":canonical_hash(study)})
    atomic(root/"state.json",{"schema_version":1,"study_sha256":canonical_hash(study),"submissions":{},"slots":{
        f"r{r}-{method}":{"round":r,"method":method,"status":"pending"} for r in range(1,5) for method in ("astra",)},
        "stages":[],"active":None,"halted":False,"selection":None,"final":{},"resource":{},
        "generation_sessions":{},"first_packet_export_epoch_seconds":None})
    if authorization is not None: authorize(root,authorization,canonical_hash(study))
    return {"prepared":True,"study_sha256":canonical_hash(study),"production_stages":0,"dispatches":0}


def load(root, *, authorized=False, allow_expired=False):
    root=Path(root).resolve(); study=read_json(root/"study.json"); state=read_json(root/"state.json")
    if read_json(root/"ready.json") != {"study_sha256":canonical_hash(study)} or state["study_sha256"] != canonical_hash(study):
        raise ValueError("Study state binding changed")
    validate_config(study["config"])
    if (root/"calendar.json").exists():
        calendar=read_json(root/"calendar.json")
        if (calendar.get("study_sha256")!=canonical_hash(study)
                or calendar.get("anchor_epoch_seconds")!=state.get("first_packet_export_epoch_seconds")
                or calendar.get("deadline_epoch_seconds")!=calendar_deadline(state)):
            raise ValueError("Immutable first-packet calendar binding changed")
    if authorized:
        auth=read_json(root/"execution-authorization.json") if (root/"execution-authorization.json").exists() else {}
        if auth.get("study_sha256") != canonical_hash(study) or auth.get("execution_authorized") is not True:
            raise PermissionError("Bind the existing study authorization before execution; no new generic approval is required")
        if file_sha256(root/"authorization-source.txt") != auth["authorization_source_sha256"]:
            raise ValueError("Execution authorization changed")
        deadline=calendar_deadline(state)
        if deadline is None: raise ValueError("First production packet has not anchored the calendar")
        if not allow_expired and time.time() >= deadline: raise TimeoutError("Seven-day study window expired")
        if state["halted"]: raise RuntimeError("Study stopped: unresolved cleanup or active-budget exhaustion")
    clock=None
    # Integrity checks during production are local processing, not free work.
    # Expired imports must still be recorded as terminal invalid submissions.
    charge=authorized and not (allow_expired and calendar_deadline(state) is not None and time.time() >= calendar_deadline(state))
    if charge:
        clock,seconds=budget_start(root,study,state,"verification-"+uuid.uuid4().hex,"analysis_export")
    try:
        if charge:
            with bounded_python(seconds):
                for name,digest in study["files"].items():
                    if file_sha256(name) != digest: raise ValueError("Frozen study evidence changed: "+Path(name).name)
        else:
            for name,digest in study["files"].items():
                if file_sha256(name) != digest: raise ValueError("Frozen study evidence changed: "+Path(name).name)
    except BaseException:
        if clock is not None: budget_finish(root,state,clock,status="failed")
        raise
    if clock is not None: budget_finish(root,state,clock)
    return root,study,state


def authorize(root, source, approved_sha):
    root,study,state=load(root)
    if canonical_hash(study) != approved_sha or not Path(source).read_bytes().strip(): raise ValueError("Explicit authorization must bind this study")
    if (root/"execution-authorization.json").exists(): raise FileExistsError("Authorization already saved")
    shutil.copyfile(source,root/"authorization-source.txt"); (root/"authorization-source.txt").chmod(0o600)
    save(root/"execution-authorization.json",{"study_sha256":approved_sha,"execution_authorized":True,
        "authorization_source_sha256":file_sha256(root/"authorization-source.txt")})
    return {"authorized":True,"study_sha256":approved_sha}


def budget_start(root, study, state, key, stage, *, reserve_bytes=0):
    if state["halted"]: raise RuntimeError("Study halted; no additional stage")
    if state["active"] is not None: raise RuntimeError("Interrupted/active stage requires recovery; replay forbidden")
    if any(r["key"]==key for r in state["stages"]): raise ValueError("Stage capacity already consumed; no retries")
    totals={s:sum(r["charged_seconds"] for r in state["stages"] if r["stage"]==s) for s in STAGES}
    operation_limits={"imports":4,"converter_prechecks":4,"conversions":4,"packaging":4,
        "browser_development":4,"browser_final":2,"browser_resource":4}
    if stage in operation_limits and sum(r["stage"]==stage for r in state["stages"]) >= operation_limits[stage]:
        raise ValueError("Stage attempt capacity exhausted; failed attempts are not replaced")
    deadline=calendar_deadline(state)
    if deadline is None: raise ValueError("Export the first production packet before active processing")
    left=min(STAGES[stage]-totals[stage],14400-sum(totals.values()),deadline-time.time(),
        study["config"]["per_attempt_seconds"][stage])
    if stage.startswith("browser_") and stage != "browser_preparation":
        browser_used=sum(v for name,v in totals.items() if name in ("browser_development","browser_final","browser_resource"))
        left=min(left,LIMITS["browser_seconds"]-browser_used)
    if left <= 0: raise TimeoutError("Stage or local/calendar budget exhausted")
    if retained_bytes(root)+reserve_bytes > LIMITS["retained_bytes"] or shutil.disk_usage(root).free-reserve_bytes < LIMITS["minimum_free_bytes"]:
        raise RuntimeError("V2 retention/free-space budget exhausted; protected evidence is never deleted")
    active={"key":key,"stage":stage,"reserved_seconds":left,"started_epoch_seconds":time.time(),"pid":os.getpid()}
    state["active"]=active; atomic(root/"state.json",state)
    return time.monotonic(),left


def budget_finish(root,state,started,*,status="completed",cleanup=True):
    active=state["active"]
    if active is None: raise ValueError("No active stage")
    elapsed=max(0,time.monotonic()-started)
    state["stages"].append({**active,"charged_seconds":elapsed,"status":status,"cleanup_confirmed":cleanup})
    state["active"]=None
    if not cleanup or elapsed > active["reserved_seconds"] or retained_bytes(root)>LIMITS["retained_bytes"]: state["halted"]=True
    atomic(root/"state.json",state)


@contextmanager
def bounded_python(seconds):
    """Bound CPU/file-only packaging; subprocess stages own their timeout/cleanup."""
    def expired(*_): raise TimeoutError("Local stage deadline")
    old=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


@contextmanager
def local_stage(root,study,state,key,stage="analysis_export"):
    started,seconds=budget_start(root,study,state,key,stage)
    try:
        with bounded_python(seconds): yield
    except BaseException:
        budget_finish(root,state,started,status="failed")
        raise
    else: budget_finish(root,state,started)


def round_ready(state,round_number):
    if type(round_number) is not int or not 1 <= round_number <= 4: raise ValueError("Exactly four proposal slots")
    if state["selection"] is not None: raise ValueError("Final selection already frozen; no more editor packets")
    for r in range(1,round_number):
        if any(state["slots"][f"r{r}-{m}"]["status"] not in TERMINAL for m in ("astra",)):
            raise ValueError("Earlier round must close before feedback or next proposal")


def export_packet(root,round_number):
    root,study,state=load(root,authorized=round_number > 1); round_ready(state,round_number)
    packet_root=root/"editor"/f"round-{round_number}"
    if (packet_root/"request.json").exists():
        request=read_json(packet_root/"request.json")
        if (not (packet_root/"ready.json").exists() or not (packet_root/"editor-packet.txt").exists()
                or read_json(packet_root/"ready.json") != {"request_sha256":canonical_hash(request),"packet_sha256":file_sha256(packet_root/"editor-packet.txt")}):
            raise RuntimeError("Incomplete or changed editor packet; no turn may be submitted")
        return {"packet":str(packet_root/"editor-packet.txt"),"request_sha256":canonical_hash(request),"reused":True}
    if str(round_number) in state["submissions"]: raise ValueError("Submission already consumed")
    if state.get("first_packet_export_epoch_seconds") is None:
        if round_number != 1: raise ValueError("Only the first packet anchors the study calendar")
        state["first_packet_export_epoch_seconds"]=time.time()
        atomic(root/"state.json",state)
        save(root/"calendar.json",{"study_sha256":canonical_hash(study),
            "anchor_epoch_seconds":state["first_packet_export_epoch_seconds"],"deadline_epoch_seconds":calendar_deadline(state)})
    context=local_stage(root,study,state,f"r{round_number}:export") if round_number > 1 else nullcontext()
    with context:
        base=Path(study["baseline"]["baseline_root"])
        old=read_json(base/"handoff/request.json")
        # Explicit allowlist. No recursively copied final, control, authorization,
        # host path, or tool-command records can enter an editor packet.
        keys=("checkpoint","policy","approved_weight_information","scoring_policy","primary_metric","secondary_metrics",
            "aggregate_metrics","development","resource_observations","warmup_observations","proposal_contract","response_contract","validation","restrictions")
        fresh_feedback=baseline_feedback(read_json(base/"data/development-items.json"),read_json(base/"baseline/graded.json")["records"],read_json(base/"baseline/graded.json")["grades"],POLICY)
        if (old["development"] != fresh_feedback or old["approved_weight_information"] != read_json(base/"weight-information/inspections.json")
                or old["scoring_policy"] != POLICY or old["policy"] != read_json(base/"weight-information/policy.json")):
            raise ValueError("Saved packet differs from verified development baseline")
        request={k:_identity_only(old[k]) for k in keys if k in old}
        request.update(schema_version=1,kind="ifeval_browser_v2_manual_request",study_id=STUDY_ID,
            request_id=str(uuid.uuid4()),round=round_number,study_sha256=canonical_hash(study),protocol_sha256=study["protocol_sha256"],
            created_at_epoch_seconds=state["first_packet_export_epoch_seconds"],
            submission_expires_at_epoch_seconds=calendar_deadline(state),
            editor=study["config"]["editor"],editor_isolation="unverified",settings=SETTINGS,
            edit_family=study["config"]["edit_family"],
            objective="Improve instruction compliance while preserving or reducing measured on-device browser resource use. Qualification requires strictly higher strict prompt compliance and no more scored output tokens than pristine on the same 32 development prompts. Preserve the frozen deployment and memory limits.",
            selection_rule=study["config"]["selection"],resource_interpretation="Timing is noisy; shorter outputs are not per-token speed gains. GPUBuffer allocation descriptors and WASM address space are not physical VRAM. Unknown signals are unavailable, not zero.",
            previous_astra_rounds=[],prior_astra_feedback=[],
            submission_rule="One submission for this round. Invalid, oversized, expired or repeated submissions consume the slot; no replacements.")
        # V2 deliberately permits scale_columns only, although the existing generic
        # envelope schema and bounded checkpoint validator remain unchanged.
        request["proposal_contract"]["operations"]="Exactly the two scale_columns operations in edit_family; sequential composition"
        request["response_contract"]["operation_forms"]={"scale_columns":request["response_contract"]["operation_forms"]["scale_columns"]}
        for r in range(1,round_number):
            slot=state["slots"][f"r{r}-astra"]
            row={k:slot.get(k) for k in ("round","status","artifact_sha256")}
            code=str(slot.get("error","")).split(":",1)[0]
            row["diagnostic"]={"code":code if code in {"ValueError","RuntimeError","TimeoutError","FileExistsError","PermissionError","OSError"} else "invalid_or_failed_submission",
                "stage":next((entry["stage"] for entry in reversed(state["stages"]) if entry["key"].startswith(f"r{r}-astra:") and entry.get("status") != "completed"),None)} if slot.get("error") else None
            proposal_path=root/"slots"/f"r{r}-astra"/"proposal.json"
            if proposal_path.exists():
                proposal=read_json(proposal_path)
                if canonical_hash(proposal) != slot.get("proposal_sha256") or proposal.get("provenance") != "editor_proposal":
                    raise ValueError("Prior Astra proposal changed")
                row["proposal"]=proposal
            row["strict_passes"]=slot.get("strict_passes")
            request["previous_astra_rounds"].append(row)
            feedback=root/"slots"/f"r{r}-astra"/"feedback.json"
            if slot["status"] in {"completed","duplicate"} and not feedback.exists():
                raise ValueError("Completed Astra round lacks required frozen development feedback")
            if feedback.exists():
                value=read_json(feedback)
                if (file_sha256(feedback) != slot.get("feedback_file_sha256")
                        or not {"round","aggregate_metrics","development"}.issubset(value) or value["round"] != r):
                    raise ValueError("Frozen Astra feedback changed")
                request["prior_astra_feedback"].append(value)
        packet_root.mkdir(parents=True,mode=0o700)
        save(packet_root/"request.json",request)
        text=("Manual authenticated Astra / Ultra exploratory proposal. Operator-selected identity; isolation unverified.\n"
              "Use only this development packet, without tools, external access, extra inspection or target inference.\n"
              "Return only the existing JSON response envelope. Invalid submission consumes this round.\n"
              f"request_id={request['request_id']}\nrequest_sha256={canonical_hash(request)}\n\n"+json.dumps(request,indent=2,sort_keys=True)+"\n")
        (packet_root/"editor-packet.txt").write_text(text); (packet_root/"editor-packet.txt").chmod(0o600)
        save(packet_root/"ready.json",{"request_sha256":canonical_hash(request),"packet_sha256":file_sha256(packet_root/"editor-packet.txt")})
    return {"packet":str(packet_root/"editor-packet.txt"),"request_sha256":canonical_hash(request),"dispatches":0}


def import_response(root,round_number,response_path):
    from .ifeval_v2_controls import validate_v2_proposal
    root,study,state=load(root,authorized=True,allow_expired=True); round_ready(state,round_number)
    if not Path(response_path).is_file(): raise ValueError("Missing response; no submission consumed")
    if state["submissions"].get(str(round_number),{}).get("status") != "requested": raise ValueError("Record begin-turn before submitting; slot is consumed before manual dispatch")
    packet=root/"editor"/f"round-{round_number}"; request=read_json(packet/"request.json")
    ready=read_json(packet/"ready.json")
    if ready != {"request_sha256":canonical_hash(request),"packet_sha256":file_sha256(packet/"editor-packet.txt")}:
        raise ValueError("Frozen editor packet changed")
    state["submissions"][str(round_number)]["status"]="consumed"
    for method in ("astra",): state["slots"][f"r{round_number}-{method}"]["status"]="invalid"
    atomic(root/"state.json",state)  # Consume before any parsing, even malformed JSON.
    with Path(response_path).open("rb") as stream: raw=stream.read(MAX_RESPONSE_BYTES+1)
    original=packet/"original-response.json"; original.write_bytes(raw); original.chmod(0o600)
    try:
        if calendar_deadline(state) is not None and time.time() >= calendar_deadline(state): raise ValueError("Expired response; submission remains consumed")
        with local_stage(root,study,state,f"r{round_number}:response"):
            if len(raw)>MAX_RESPONSE_BYTES: raise ValueError("Response exceeds frozen byte limit; bounded original prefix retained")
            response=read_json_bounded(original,max_bytes=MAX_RESPONSE_BYTES)
            proposal=validate_v2_proposal(validate_response(response,request),BASE_SOURCE)
            for method,value in (("astra",proposal),):
                target=root/"slots"/f"r{round_number}-{method}"; target.mkdir(parents=True,mode=0o700)
                save(target/"proposal.json",value)
                state["slots"][f"r{round_number}-{method}"].update(status="accepted",proposal_sha256=canonical_hash(value))
            state["submissions"][str(round_number)]["status"]="accepted"
    except Exception as exc:
        state["submissions"][str(round_number)].update(status="invalid",error=type(exc).__name__+": "+str(exc))
        state["slots"][f"r{round_number}-astra"]["error"]=str(exc)
        atomic(root/"state.json",state)
        return {"accepted":False,"slot_consumed":True,"error":str(exc)}
    atomic(root/"state.json",state)
    return {"accepted":True,"round":round_number,"automatic_frontier_requests":0}


def _slot(state,round_number,method):
    if method != "astra": raise ValueError("Only Astra candidates belong to the narrowed workflow")
    key=f"r{round_number}-{method}"; round_ready(state,round_number)
    return key,state["slots"][key]


def build(root,round_number,method,converter_python):
    from .editor_tools import run_checkpoint_action
    from .ifeval_v2_candidate import convert,package_candidate
    root,study,state=load(root,authorized=True); key,slot=_slot(state,round_number,method)
    if slot["status"] != "accepted": raise ValueError("Build requires accepted fresh proposal")
    target=root/"slots"/key; base=Path(study["baseline"]["baseline_root"])
    policy=base/"weight-information/policy.json"; host=read_json(base/"weight-information/host.json")
    model=host["model_path"]; proposal=read_json(target/"proposal.json")
    if canonical_hash(proposal) != slot["proposal_sha256"]: raise ValueError("Frozen proposal changed before import")
    from .ifeval_v2_controls import validate_v2_proposal
    validate_v2_proposal(proposal,BASE_SOURCE)
    conversion_manifest=Path("artifacts/browser-integration-v1/conversion/artifacts.json").resolve()
    provenance="editor_proposal"
    control=None
    started=None
    try:
        started,seconds=budget_start(root,study,state,key+":import","imports",reserve_bytes=2*1024**3)
        audit=run_checkpoint_action(model=model,policy=read_json(policy),exposure_log=target/"exposures.jsonl",output=target/"source",tool="propose",
            arguments={"proposal":proposal},timeout_seconds=max(.001,seconds-2),check_active=lambda:None,expected_original_sha256=BASE_SOURCE)
        save(target/"source-audit.json",audit); budget_finish(root,state,started); started=None
        started,seconds=budget_start(root,study,state,key+":precheck","converter_prechecks",reserve_bytes=1024**3)
        def begin_conversion():
            nonlocal started
            budget_finish(root,state,started)
            started,remaining=budget_start(root,study,state,key+":conversion","conversions",reserve_bytes=1024**3)
            signal.setitimer(signal.ITIMER_REAL,remaining)
            return remaining
        with bounded_python(seconds):
            convert(model,target/"source",conversion_manifest,target/"converted",converter_python,policy_path=policy,
            expected_provenance=provenance,control_audit=control,precheck_timeout=seconds,on_conversion=begin_conversion)
        budget_finish(root,state,started); started=None
        started,seconds=budget_start(root,study,state,key+":package","packaging",reserve_bytes=1024**3)
        with bounded_python(seconds):
            artifact=package_candidate(model,target/"source",conversion_manifest,target/"converted",target/"deployed",policy_path=policy,
                expected_provenance=provenance,control_audit=control)
        budget_finish(root,state,started); started=None
        if artifact["conversion_erases_edit"]: raise ValueError("Quantization erased edit; slot spent")
        slot.update(status="built",artifact_sha256=artifact["artifact_sha256"],artifact_path=artifact["artifact_manifest"],artifact_file_sha256=file_sha256(artifact["artifact_manifest"]))
    except BaseException as exc:
        if started is not None and state["active"] is not None: budget_finish(root,state,started,status="failed")
        slot.update(status="failed",error=type(exc).__name__+": "+str(exc))
    atomic(root/"state.json",state)
    return {"slot":key,"status":slot["status"],"error":slot.get("error")}


def begin_turn(root,round_number,*,recording_supplied_response=False):
    """Record the manual proposal slot BEFORE pasting its packet to Astra."""
    root,study,state=load(root,authorized=True,allow_expired=recording_supplied_response); round_ready(state,round_number)
    if str(round_number) in state["submissions"]: raise ValueError("Editor submission slot already consumed")
    if not (root/"editor"/f"round-{round_number}"/"request.json").exists(): raise ValueError("Export this round packet first")
    state["submissions"][str(round_number)]={"status":"requested","started_epoch_seconds":time.time(),
        "frontier_usage":None,"immutable_model_version":None,"observed_effort":None,"marginal_billing":None,
        "accounting_event":"original_response_supplied" if recording_supplied_response else "operator_manual_turn_begin"}
    for method in ("astra",): state["slots"][f"r{round_number}-{method}"]["status"]="awaiting_response"
    atomic(root/"state.json",state)
    return {"round":round_number,"submission_consumed":True,"automatic_frontier_requests":0}


def close_turn(root,round_number,reason):
    root,study,state=load(root)
    submission=state["submissions"].get(str(round_number))
    if not submission or submission["status"] != "requested": raise ValueError("Only an unresolved requested turn may close")
    submission.update(status="invalid",error=reason)
    for method in ("astra",): state["slots"][f"r{round_number}-{method}"].update(status="invalid",error=reason)
    atomic(root/"state.json",state)
    return {"round":round_number,"status":"invalid","replacement_permitted":False}


def prepare_benchmark(root,study):
    """Copy the existing continuation harness; only private V2 paths are mutable.

    Hard links are used only for frozen, read-only model data. The driver checks
    real paths stay within this V2 public tree. No browser or model is started.
    """
    destination=root/"benchmark"
    if destination.exists(): return destination
    source=Path(study["benchmark_source"])
    shutil.copytree(source,destination,symlinks=True,ignore=shutil.ignore_patterns(".git","node_modules","models","results",".DS_Store","__pycache__","*.pyc","*.pyo"))
    (destination/"node_modules").symlink_to((source/"node_modules").resolve(),target_is_directory=True)
    (destination/"public/models").mkdir(parents=True,exist_ok=True)
    old=read_json(Path(study["baseline"]["baseline_root"])/"baseline-config.json")
    base_public=Path(study["baseline"]["baseline_root"])/"benchmark/public"
    for asset in old["assets"]:
        relative=urlparse(asset["url"]).path.lstrip("/")
        target=destination/"public"/relative; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): os.link(base_public/relative,target)
    return destination


def prepare_session(root,study,key,*,artifact=None,final=False):
    from . import ifeval_v2_runtime as runtime
    from .browser_candidate import _files
    session=root/"sessions"/key
    if session.exists(): raise ValueError("Session already prepared; repeat dispatch forbidden")
    benchmark=prepare_benchmark(root,study); baseline=Path(study["baseline"]["baseline_root"])
    config=copy.deepcopy(read_json(baseline/"baseline-config.json"))
    items_path=Path(study["evaluator_root"])/"final-items.json" if final else baseline/"data/development-items.json"
    items=read_json(items_path)
    if final:
        from .ifeval_v2_dataset import verify_frozen
        verify_frozen(Path(study["evaluator_root"]))
        if not 1<=len(items)<=128 or len(items)!=study["config"]["final_set"]["prompts"]:
            raise ValueError("Final task session requires the actual nonempty frozen prompt count")
    session.mkdir(parents=True,mode=0o700); (session/"benchmark").symlink_to(benchmark,target_is_directory=True)
    config.update(runId=STUDY_ID+"-"+key,arm="baseline" if artifact is None else "candidate")
    config.pop("configurationSha256",None)
    if final:
        config["evaluationStage"]="heldout"
        config["narrowedFinalPromptCount"]=len(items)
        # The canonical evaluator items retain the exact rendered count used at freeze.
        config["items"]=[{"item_id":i["item_id"],"prompt":i["task_text"],
            "rendered_input_tokens":i["metadata"]["rendered_input_tokens"]} for i in items]
    if artifact is not None:
        if _files(artifact["path"]) != artifact["files"] or canonical_hash(artifact["files"]) != artifact["artifact_sha256"]:
            raise ValueError("Packaged candidate bytes changed")
        relative="models/v2-"+artifact["artifact_sha256"]+"/resolve/"+artifact["artifact_sha256"]+"/"
        target=benchmark/"public"/relative
        if not target.exists(): shutil.copytree(artifact["path"],target,copy_function=os.link)
        origin=urlparse(config["model"]["modelUrl"]); base_url=origin.scheme+"://"+origin.netloc+"/"+relative
        config["model"].update(id=artifact["model_id"],modelUrl=base_url)
        config.update(sourceCheckpointSha256=artifact["source_checkpoint_sha256"],convertedArtifactSha256=artifact["artifact_sha256"])
        names=["mlc-chat-config.json","tensor-cache.json","tokenizer.json",*[r["dataPath"] for r in read_json(target/"tensor-cache.json")["records"]]]
        config["assets"]=[{"url":base_url+n,"sha256":artifact["files"][n]["sha256"]} for n in names]+[
            {"url":config["model"]["modelLib"],"sha256":artifact["files"]["model.wasm"]["sha256"]}]
    # Freeze the actual rebuilt client/worker closure, including hashed chunks.
    from .browser_resource import served_build
    origin=urlparse(config["model"]["modelUrl"])
    config["servedBuild"]=served_build(benchmark,origin.scheme+"://"+origin.netloc)
    config["configurationSha256"]=canonical_hash(config)
    save(session/"baseline-config.json",config)
    files={str(session/"baseline-config.json"):file_sha256(session/"baseline-config.json"),str(items_path):file_sha256(items_path)}
    files.update({str(p):file_sha256(p) for p in benchmark.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix not in (".pyc",".pyo") and "models" not in p.parts and "node_modules" not in p.parts and "__pycache__" not in p.parts})
    manifest={"schema_version":1,"kind":"ifeval_v2_session","arm":config["arm"],"run_id":config["runId"],"slot":key,
        "study_manifest_sha256":canonical_hash(study),"configuration_sha256":config["configurationSha256"],
        "artifact_sha256":config["convertedArtifactSha256"],"checkpoint":{**read_json(baseline/"manifest.json")["checkpoint"],"sha256":config["sourceCheckpointSha256"]},
        "scheduled_prompts":len(items),"scheduled_instructions":sum(len(i["target"]["instruction_id_list"]) for i in items),
        "items_path":str(items_path),"items_sha256":canonical_hash(items),"settings":SETTINGS,"policy":POLICY,
        "workload_identity":"ifeval-task","browser_seconds":5400 if final else 3600,
        "grader":study["baseline"]["grader"],"official_ifeval_root":study["baseline"]["official_ifeval_root"],
        "official_assets":study["baseline"]["official_assets"],"files":files,
        "legacy_transport_discriminator":"study=ifeval-browser-v1 reuses unchanged client schema; run_id and evidence are V2"}
    save(session/"manifest.json",manifest); save(session/"ready.json",{"manifest_sha256":canonical_hash(manifest)})
    with runtime.ledger(session,manifest,create=True) as book: save(session/"accounting-before.json",book.summary())
    return session


def _evaluate_session(root,study,state,key,artifact,*,final=False):
    from . import ifeval_v2_runtime as runtime
    workload="final" if final else "development"
    started,seconds=budget_start(root,study,state,key+":browser","browser_"+workload,reserve_bytes=64*1024**2)
    session=None
    try:
        with bounded_python(min(seconds,300)):
            session=prepare_session(root,study,key,artifact=artifact,final=final)
        def on_load():
            begin_session_capacity(root,study,state,key,workload=workload,
                requests=study["config"]["final_set"]["prompts"] if final else 32,max_tokens=1024)
            return seconds-(time.monotonic()-started)
        remaining=max(.001,seconds-(time.monotonic()-started))
        with bounded_python(remaining):
            execution=runtime.execute(session,on_load=on_load,preparation_seconds=min(300,remaining))
        clean=all(execution["cleanup"].get(k) is True for k in CLEANUP) and execution["cleanup"].get("retained_owned_pids_alive")==[]
        if key in state.get("generation_sessions",{}):
            state["generation_sessions"][key]["observed"]={k:execution.get(k) for k in (
                "engine_load_attempts","warmup_dispatches","scored_dispatches","generation_dispatches","generation_budget","status")}
        budget_finish(root,state,started,status=execution["status"],cleanup=clean); started=None
        started,seconds=budget_start(root,study,state,key+":grading","grading")
        deadline=time.monotonic()+seconds
        summary=runtime.grade(session,before_grade=lambda:_require_grading_time(deadline))
        budget_finish(root,state,started); started=None
        return session,summary
    except BaseException:
        if started is not None and state["active"] is not None:
            clean=session is None or not (session/"execution").exists()
            budget_finish(root,state,started,status="failed",cleanup=clean)
        raise


def _require_grading_time(deadline):
    if deadline-time.monotonic() < 20: raise TimeoutError("Insufficient grading stage allowance for one bounded checker")


def evaluate(root,round_number,method):
    root,study,state=load(root,authorized=True); key,slot=_slot(state,round_number,method)
    if slot["status"] != "built": raise ValueError("Evaluation requires a newly built candidate")
    # Reuse complete grades for the same artifact; savings never transfer.
    duplicate=next((s for s in state["slots"].values() if s.get("artifact_sha256")==slot["artifact_sha256"] and s["status"] in {"completed","duplicate"}),None)
    if duplicate:
        for k in ("strict_passes","scored_output_tokens","deployment_limits_satisfied","memory_limits_satisfied","completed","technically_valid","independently_graded","scheduled_prompts","graded_prompts","session"):
            slot[k]=duplicate[k]
        slot.update(status="duplicate",reused_from=f"r{duplicate['round']}-{duplicate['method']}")
    elif slot["artifact_sha256"]==BASE_ARTIFACT:
        slot.update(status="duplicate",strict_passes=16,scored_output_tokens=study["baseline"]["scored_output_tokens"],deployment_limits_satisfied=True,memory_limits_satisfied=True,completed=True,technically_valid=True,independently_graded=True,
            scheduled_prompts=32,graded_prompts=32,session=study["baseline"]["baseline_root"],reused_from="pristine")
    else:
        try:
            artifact=read_json(slot["artifact_path"])
            if artifact["artifact_sha256"] != slot["artifact_sha256"] or file_sha256(slot["artifact_path"]) != slot["artifact_file_sha256"]:
                raise ValueError("Candidate artifact differs from frozen built identity")
            session,summary=_evaluate_session(root,study,state,key,artifact)
            slot.update(status="completed" if summary["technically_valid"] else "failed",strict_passes=summary["metrics"]["prompt_level_strict_accuracy"]["passed"],
                scored_output_tokens=summary["known_generated_tokens"],deployment_limits_satisfied=summary["deployment_limits_satisfied"],memory_limits_satisfied=summary["memory_limits_satisfied"],
                completed=summary["execution_complete"],technically_valid=summary["technically_valid"],independently_graded=summary["semantic_grading_complete"],
                scheduled_prompts=32,graded_prompts=32-summary["unresolved_grader_errors"],session=str(session))
        except BaseException as exc: slot.update(status="failed",error=type(exc).__name__+": "+str(exc))
    atomic(root/"state.json",state)
    if method=="astra" and slot.get("session") and slot["status"] in {"completed","duplicate","failed"}:
        with local_stage(root,study,state,key+":feedback"):
            _feedback(root,study,key,slot)
            slot["feedback_file_sha256"]=file_sha256(root/"slots"/key/"feedback.json")
            atomic(root/"state.json",state)
    return {"slot":key,"status":slot["status"],"strict_passes":slot.get("strict_passes"),"error":slot.get("error")}


def resource_feedback(session,envelope):
    """Keep scopes and missing counters separate; share the historical worker summary."""
    from .ifeval_handoff import resource_summary
    result=resource_summary(session,envelope)
    arm=read_json(Path(session)/"manifest.json").get("arm","baseline")
    if arm=="baseline": return result
    rows=[]; path=Path(session)/"execution"/(arm+"-1")/"resources.jsonl"
    if path.exists(): rows=[json.loads(line) for line in path.read_text().splitlines()]
    host={}; roles={}
    for row in rows:
        for name,value in row.get("host",{}).items():
            if type(value) in (int,float):
                entry=host.setdefault(name,{"minimum":value,"maximum":value,"samples":0})
                entry.update(minimum=min(entry["minimum"],value),maximum=max(entry["maximum"],value),samples=entry["samples"]+1)
        for process in row.get("owned_processes",[]):
            role=roles.setdefault(process["role"],{})
            for name in ("footprintBytes","rssBytes","wiredBytes"):
                value=process.get(name)
                if type(value) in (int,float): role[name]=max(role.get(name,0),value)
    result.update(host_sampled_numeric_ranges=host,host_sample_count=len(rows),owned_role_sampled_maxima=roles)
    return result


def _feedback(root,study,key,slot):
    session=Path(slot["session"])
    manifest=read_json(session/"manifest.json"); arm=manifest.get("arm","baseline")
    envelope=read_json(session/arm/"graded.json")
    base=Path(study["baseline"]["baseline_root"])
    items=read_json(base/"data/development-items.json")
    complete=slot.get("technically_valid") is True
    if complete:
        feedback=baseline_feedback(items,envelope["records"],envelope["grades"],POLICY)
        aggregate=aggregate_feedback(feedback)
    else:
        feedback=[]
        for item,record,grade in zip(items,envelope["records"],envelope["grades"],strict=True):
            metrics=grade.get("metrics",{}); resolved=grade.get("status")=="graded"
            feedback.append({"item_id":item["item_id"],"prompt":item["task_text"],"raw_response":record.get("raw_output"),
                "status":record["status"],"grading_status":grade["status"],"finish_reason":record.get("finish_reason"),
                "token_counts":record.get("token_counts"),"duration_seconds":record.get("duration_seconds"),
                "constraints":[{"instruction_id":name,"kwargs":args,
                    "strict":metrics.get("instruction_strict",[])[i] if resolved else None,
                    "loose":metrics.get("instruction_loose",[])[i] if resolved else None}
                    for i,(name,args) in enumerate(zip(item["target"]["instruction_id_list"],item["target"]["kwargs"],strict=True))]})
        aggregate={"complete":False,"official_summary":envelope["summary"],"interpretation":"Partial failed development evidence; unresolved entries are not successes and this candidate cannot qualify."}
    original=read_json(base/"baseline/graded.json")
    original_feedback=baseline_feedback(items,original["records"],original["grades"],POLICY)
    paired={}
    for mode in ("strict","loose"):
        before=[all(c[mode] for c in row["constraints"]) for row in original_feedback]
        after=[None if any(c[mode] is None for c in row["constraints"]) else all(c[mode] for c in row["constraints"]) for row in feedback]
        resolved=[(a,b) for a,b in zip(before,after) if b is not None]
        paired[mode]={"improvements":sum(not a and b for a,b in resolved),
            "regressions":sum(a and not b for a,b in resolved),"ties":sum(a==b for a,b in resolved),"unresolved":sum(b is None for b in after)}
    resources=resource_feedback(session,envelope); prior_resources=resource_feedback(base,original)
    generation_seconds=sum(r["duration_seconds"] or 0 for r in envelope["records"])
    baseline_seconds=sum(r["duration_seconds"] for r in original["records"])
    memory_flags=[]
    for role,values in resources["owned_role_sampled_maxima"].items():
        for name,value in values.items():
            baseline=prior_resources["owned_role_sampled_maxima"].get(role,{}).get(name)
            if baseline is not None and value>baseline: memory_flags.append({"role":role,"counter":name,"candidate":value,"baseline":baseline})
    audit={}
    for name in ("source-audit.json","converted-source-binding.json","deployed/source-audit.json","deployed/deployment-difference.json"):
        path=root/"slots"/key/name
        if path.exists(): audit[name]=file_sha256(path)
    save(root/"slots"/key/"feedback.json",{"round":slot["round"],"aggregate_metrics":aggregate,
        "development":feedback,"paired_outcomes":paired,"scored_output_tokens":slot["scored_output_tokens"],
        "baseline_scored_output_tokens":study["baseline"]["scored_output_tokens"],
        "generation_seconds":generation_seconds,"baseline_generation_seconds":baseline_seconds,
        "resource_observations":resources,"resource_regression_flags":{"generation_seconds_increased":generation_seconds>baseline_seconds,
            "observed_process_memory_increases":memory_flags,"interpretation":"Unpaired historical development observations are noisy; no permissible regression margin or speed/equality claim. Final matched resource sessions supply separate evidence."},
        "applied_edit":read_json(root/"slots"/key/"proposal.json"),
        "audit_identities":audit,"artifact_sha256":slot["artifact_sha256"],
        "source_checkpoint_sha256":manifest["checkpoint"]["sha256"]})


def select(root):
    from .ifeval_v2_controls import select_candidate,final_arms
    root,study,state=load(root,authorized=True)
    if state["selection"] is not None: return state["selection"]
    if any(s["status"] not in TERMINAL for s in state["slots"].values()): raise ValueError("All four proposal rounds must close before final identities freeze")
    with local_stage(root,study,state,"final:selection"):
        selected={"astra":select_candidate(list(state["slots"].values()),BASE_ARTIFACT,baseline_tokens=study["baseline"]["scored_output_tokens"])}
        plan=final_arms(BASE_ARTIFACT,selected["astra"],final_prompts=study["config"]["final_set"]["prompts"])
        plan.update(study_sha256=canonical_hash(study),selected=selected,final_manifest_sha256=study["config"]["final_set"]["manifest_sha256"],
            frozen_before_final_results=True,created_epoch_seconds=read_json(root/"selection-intent.json")["created_epoch_seconds"] if (root/"selection-intent.json").exists() else time.time())
        plan["resource_artifacts"]={}
        for name,identity in plan["named_arms"].items():
            path=None if identity==BASE_ARTIFACT else next(s["artifact_path"] for s in state["slots"].values() if s.get("artifact_sha256")==identity)
            plan["resource_artifacts"][name]={"artifact_sha256":identity,"artifact_path":path,
                "artifact_file_sha256":None if path is None else file_sha256(path)}
        for row in plan["distinct_evaluations"]:
            row["artifact_path"]=None if row["artifact_sha256"]==BASE_ARTIFACT else next(s["artifact_path"] for s in state["slots"].values() if s.get("artifact_sha256")==row["artifact_sha256"])
        for row in plan["distinct_evaluations"]:
            row["artifact_file_sha256"]=None if row["artifact_path"] is None else file_sha256(row["artifact_path"])
            if row["artifact_path"] and read_json(row["artifact_path"])["artifact_sha256"] != row["artifact_sha256"]:
                raise ValueError("Selected package differs from selected development identity")
        from .browser_resource import freeze_comparison
        plan["resource_preset_sha256"]=freeze_comparison(root,study,plan)
        if (root/"selection-intent.json").exists():
            if read_json(root/"selection-intent.json") != plan: raise ValueError("Existing selection intent differs")
        else: save(root/"selection-intent.json",plan)
        from .ifeval_v2_analysis import freeze_analysis_plan
        freeze_analysis_plan(root,study,plan)
        if (root/"final-selection.json").exists():
            if read_json(root/"final-selection.json") != plan: raise ValueError("Frozen selected identity changed")
        else: save(root/"final-selection.json",plan)
        state["selection"]=plan; atomic(root/"state.json",state)
    return plan


def evaluate_final(root,arm):
    root,study,state=load(root,authorized=True)
    plan=state["selection"]
    if plan is None or read_json(root/"final-selection.json") != plan: raise ValueError("Final identities must freeze before any final outputs")
    rows=plan["distinct_evaluations"]; row=next((r for r in rows if r["dispatch_arm"]==arm),None)
    if row is None: raise ValueError("Not a distinct frozen final arm; duplicate arms reuse their recorded output")
    if arm in state["final"]: raise ValueError("Final arm already consumed; no retries")
    for previous in rows[:rows.index(row)]:
        if state["final"].get(previous["dispatch_arm"],{}).get("status") != "completed": raise ValueError("Final order requires complete earlier arm")
    state["final"][arm]={"status":"started","artifact_sha256":row["artifact_sha256"]}; atomic(root/"state.json",state)
    try:
        artifact=read_json(row["artifact_path"]) if row["artifact_path"] else None
        if artifact is not None and (artifact["artifact_sha256"] != row["artifact_sha256"]
                or file_sha256(row["artifact_path"]) != row["artifact_file_sha256"]):
            raise ValueError("Final artifact differs from frozen selected identity")
        session,summary=_evaluate_session(root,study,state,"final-"+arm,artifact,final=True)
        state["final"][arm].update(status="completed" if summary["technically_valid"] else "failed",session=str(session),summary=summary)
    except BaseException as exc: state["final"][arm].update(status="failed",error=type(exc).__name__+": "+str(exc))
    atomic(root/"state.json",state)
    return state["final"][arm]


def evaluate_resource(root,index):
    """One prescheduled standard-workload session; ABBA is never a retry policy."""
    from . import browser_resource
    root,study,state=load(root,authorized=True)
    plan=state["selection"]
    if plan is None or read_json(root/"final-selection.json")!=plan:
        raise ValueError("Selection must freeze before resource comparison")
    order=plan["resource_order"]
    if type(index) is not int or not 1<=index<=len(order): raise ValueError("No scheduled resource session")
    for row in plan["distinct_evaluations"]:
        if row["dispatch_arm"] not in state["final"]: raise ValueError("Final task arms must be attempted before resource comparison")
    records=state.setdefault("resource",{})
    if str(index) in records: raise ValueError("Resource session already consumed; no retries")
    if any(str(previous) not in records for previous in range(1,index)):
        raise ValueError("Frozen ABBA resource order changed")
    name=order[index-1]; row=plan["resource_artifacts"][name]
    artifact=read_json(row["artifact_path"]) if row["artifact_path"] else None
    if artifact is not None and (artifact["artifact_sha256"]!=row["artifact_sha256"]
            or file_sha256(row["artifact_path"])!=row["artifact_file_sha256"]): raise ValueError("Frozen resource artifact changed")
    key=f"resource-{index}-{name}"
    records[str(index)]={"status":"started","arm":name,"artifact_sha256":row["artifact_sha256"]}
    atomic(root/"state.json",state)
    started=None; session=None
    try:
        started,seconds=budget_start(root,study,state,key+":browser","browser_resource",reserve_bytes=64*1024**2)
        with bounded_python(min(300,seconds)):
            session=browser_resource.prepare_session(root,study,key,artifact=artifact)
        def on_load():
            begin_session_capacity(root,study,state,key,workload="resource",requests=60,max_tokens=64)
            return seconds-(time.monotonic()-started)
        remaining=max(.001,seconds-(time.monotonic()-started))
        with bounded_python(remaining):
            result=browser_resource.execute(session,on_load=on_load,preparation_seconds=min(300,remaining))
        clean=all(result["cleanup"].get(k) is True for k in CLEANUP) and result["cleanup"].get("retained_owned_pids_alive")==[]
        if key in state.get("generation_sessions",{}):
            state["generation_sessions"][key]["observed"]={k:result.get(k) for k in ("engine_load_attempts","warmup_dispatches","scored_dispatches","generation_dispatches","generation_budget","status")}
        budget_finish(root,state,started,status=result["status"],cleanup=clean); started=None
        records[str(index)].update(status=result["status"],session=str(session),workload_identity="browser-llm-bench-standard")
    except BaseException as exc:
        if started is not None and state["active"] is not None:
            budget_finish(root,state,started,status="failed",cleanup=session is None or not (session/"execution").exists())
        records[str(index)].update(status="failed",error=type(exc).__name__+": "+str(exc))
    atomic(root/"state.json",state)
    return records[str(index)]


def advance(root,*,response=None,converter_python="artifacts/browser-integration-v1/conversion/venv/bin/python"):
    """Continue authorized stages until one genuine manual response is needed."""
    root,study,state=load(root)
    if state["active"] is not None: raise RuntimeError("Interrupted stage requires recovery; replay forbidden")
    for number in range(1,5):
        state=read_json(root/"state.json"); slot=state["slots"][f"r{number}-astra"]
        if slot["status"] in TERMINAL: continue
        if slot["status"]=="pending":
            if response is None:
                packet=export_packet(root,number)
                return {"status":"awaiting_manual_astra_response","round":number,**packet,"submission_consumed":False}
            # The supplied original response is itself the one manual submission.
            # Consume before parsing; this records no automated editor dispatch.
            begin_turn(root,number,recording_supplied_response=True)
            slot=read_json(root/"state.json")["slots"][f"r{number}-astra"]
        if slot["status"]=="awaiting_response":
            if response is None:
                return {"status":"awaiting_manual_astra_response","round":number,"submission_consumed":True,
                    "packet":str(root/"editor"/f"round-{number}"/"editor-packet.txt")}
            result=import_response(root,number,response); response=None
            if not result["accepted"]:
                if time.time()>=calendar_deadline(read_json(root/"state.json")):
                    return {"status":"calendar_expired_response_consumed","round":number,"result":result}
                continue
            slot=read_json(root/"state.json")["slots"][f"r{number}-astra"]
        if slot["status"]=="accepted":
            result=build(root,number,"astra",converter_python)
            if result["status"]!="built": continue
            slot=read_json(root/"state.json")["slots"][f"r{number}-astra"]
        if slot["status"]=="built": evaluate(root,number,"astra")
    plan=select(root)
    for row in plan["distinct_evaluations"]:
        state=read_json(root/"state.json")
        if row["dispatch_arm"] not in state["final"]: evaluate_final(root,row["dispatch_arm"])
    for index in range(1,len(plan["resource_order"])+1):
        state=read_json(root/"state.json")
        if str(index) not in state.get("resource",{}): evaluate_resource(root,index)
    return {"status":"authorized_schedule_finished","selection":plan,"analysis":analyze_all(root),"state":read_json(root/"state.json")}


def analyze_all(root):
    from .ifeval_v2_analysis import analyze
    from .browser_resource import analyze as analyze_resource
    root,study,state=load(root,authorized=True)
    if (root/"study-result.json").exists(): return read_json(root/"study-result.json")
    clock,seconds=budget_start(root,study,state,"final:analysis","analysis_export")
    try:
        with bounded_python(seconds):
            task=read_json(root/"final-analysis.json") if (root/"final-analysis.json").exists() else analyze(root)
            resource=analyze_resource(root)
            retained=state["selection"]["selected"]["astra"]["selection"]=="pristine_fallback"
            result={"kind":"ifeval_browser_quality_resource_result","study_sha256":canonical_hash(study),
                "selection_sha256":canonical_hash(state["selection"]),"task":task,"resource":resource,
                "joint_goal_status":"no_qualifying_edit" if retained else "inconclusive",
                "interpretation":"Pristine retained; no redundant comparison was run." if retained else
                    "Report task and resource observations separately. Two resource sessions per artifact form a small pilot; missing untouched final task data or uncertain resource differences cannot establish the joint goal.",
                "observed_generation_sessions":state.get("generation_sessions",{}),"final_results_returned_to_editor":False}
            save(root/"study-result.json",result)
    except BaseException:
        budget_finish(root,state,clock,status="failed"); raise
    budget_finish(root,state,clock)
    return result


def recover(root):
    """Never resume an interrupted dispatch; close spent stage after process audit.

    Missing terminal cleanup is a hard stop, not permission to reuse capacity.
    Completed earlier rounds can continue using their durable state.
    """
    from .browser_v2_runner import alive
    root,study,state=load(root)
    active=state["active"]
    if active is None: return {"recovered":False,"active":None,"halted":state["halted"]}
    if alive(active["pid"]): raise RuntimeError("Recorded stage process still exists; cannot recover")
    state["stages"].append({**active,"charged_seconds":active["reserved_seconds"],"status":"interrupted","cleanup_confirmed":False})
    state["active"]=None; state["halted"]=True
    atomic(root/"state.json",state)
    return {"recovered":True,"halted":True,"reason":"Interrupted stage charged full reservation; closure unverified; no retries"}


def status(root):
    root,study,state=load(root)
    cap=effective_capacity(study); sessions=list(state.get("generation_sessions",{}).values())
    stages=state["stages"]
    consumed={"editor_submissions":len(state["submissions"]),
        "imports":sum(r["stage"]=="imports" for r in stages),
        "conversions":sum(r["stage"]=="conversions" for r in stages),
        "packages":sum(r["stage"]=="packaging" for r in stages),
        "engine_loads":sum(r["loads"] for r in sessions),"warmups":sum(r["warmups"] for r in sessions),
        "task_requests":sum(r["requests"] for r in sessions if r["workload"]!="resource"),
        "resource_requests":sum(r["requests"] for r in sessions if r["workload"]=="resource"),
        "scored_requests":sum(r["requests"] for r in sessions),"dispatches":sum(r["dispatches"] for r in sessions),
        "generated_tokens":sum(r["reserved_tokens"] for r in sessions),
        "local_seconds":sum(r["charged_seconds"] for r in stages),
        "browser_seconds":sum(r["charged_seconds"] for r in stages if r["stage"] in ("browser_development","browser_final","browser_resource")),
        "retained_bytes":retained_bytes(root)}
    return {"study_sha256":canonical_hash(study),"submissions":state["submissions"],"slots":state["slots"],
        "stages":stages,"active":state["active"],"halted":state["halted"],"selection":state["selection"],
        "authorized_maximum":LIMITS,"effective_capacity":cap,"capacity_consumed_or_reserved":consumed,
        "remaining_capacity":{k:max(0,v-consumed[k]) for k,v in cap.items() if k in consumed},
        "capacity_note":"Session reservations are ceilings, not observed generation counts. Failed/unused reserved capacity is nontransferable; observed outcomes are reported separately.",
        "generation_sessions":state.get("generation_sessions",{}),
        "calendar_deadline_epoch_seconds":calendar_deadline(state),"resource":state.get("resource",{})}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","export","authorize","begin-turn","close-turn","import","build","evaluate","select","final","resource","advance","analyze","status","recover"])
    parser.add_argument("--root",default="artifacts/ifeval-browser-v2-narrowed"); parser.add_argument("--round",type=int); parser.add_argument("--method",choices=["astra"],default="astra")
    parser.add_argument("--baseline-root",default="artifacts/ifeval-browser-v1"); parser.add_argument("--evaluator-root",default="artifacts/ifeval-browser-v2-narrowed/evaluator")
    parser.add_argument("--config",default="configs/ifeval_browser_quality_resource.json"); parser.add_argument("--protocol",default="docs/ifeval-browser-quality-resource-protocol.md")
    parser.add_argument("--response"); parser.add_argument("--reason"); parser.add_argument("--authorization"); parser.add_argument("--approved-study-sha256")
    parser.add_argument("--converter-python",default="artifacts/browser-integration-v1/conversion/venv/bin/python"); parser.add_argument("--arm"); parser.add_argument("--session",type=int)
    args=parser.parse_args(); root=Path(args.root); root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with _lock(root):
        if args.command=="prepare": result=prepare(root,args.baseline_root,args.evaluator_root,args.config,args.protocol,authorization=args.authorization)
        elif args.command=="export": result=export_packet(root,args.round)
        elif args.command=="authorize": result=authorize(root,args.authorization,args.approved_study_sha256)
        elif args.command=="begin-turn": result=begin_turn(root,args.round)
        elif args.command=="close-turn": result=close_turn(root,args.round,args.reason)
        elif args.command=="import": result=import_response(root,args.round,args.response)
        elif args.command=="build": result=build(root,args.round,args.method,args.converter_python)
        elif args.command=="evaluate": result=evaluate(root,args.round,args.method)
        elif args.command=="select": result=select(root)
        elif args.command=="final": result=evaluate_final(root,args.arm)
        elif args.command=="resource": result=evaluate_resource(root,args.session)
        elif args.command=="advance": result=advance(root,response=args.response,converter_python=args.converter_python)
        elif args.command=="analyze": result=analyze_all(root)
        elif args.command=="recover": result=recover(root)
        else: result=status(root)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__": main()
