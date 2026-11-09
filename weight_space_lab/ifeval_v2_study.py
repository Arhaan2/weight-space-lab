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
    "conversions":8,"dispatches":651,"editor_submissions":4,"engine_loads":11,
    "generated_tokens":655536,"imports":8,"local_seconds":14400,
    "minimum_free_bytes":8*1024**3,"packages":8,"retained_bytes":32*1024**3,
    "scored_requests":640,"warmups":11}
STAGES = {"analysis_export":600,"browser":10800,"browser_preparation":120,
    "conversions":1200,"converter_prechecks":120,"grading":720,"imports":600,"packaging":240}
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
    if (config["study_id"] != "ifeval-browser-v2" or config["settings"] != SETTINGS
            or config["grading_policy"] != POLICY or config["limits"] != LIMITS
            or config["stage_seconds"] != STAGES
            or config["submission_deadline_epoch_ms"] != config["freeze_epoch_ms"]+604800000
            or config["editor"]["max_submissions"] != 4 or config["editor"]["max_response_bytes"] != MAX_RESPONSE_BYTES
            or config["schedule"]["rounds"] != 4
            or config["schedule"]["development_order"] != [["astra","control"],["control","astra"],["astra","control"],["control","astra"]]
            or config["schedule"]["final_arms"] != ["pristine_baseline","selected_astra","selected_random_control"]):
        raise ValueError("V2 frozen schedule or budget differs")


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
        "development_sha256":DEV_SHA,"strict_passes":16,"grader":manifest["grader"],"official_ifeval_root":manifest["official_ifeval_root"],
        "official_assets":manifest["official_assets"]}


def prepare(root, baseline_root, evaluator_root, config_path, protocol_path):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (root/"study.json").exists(): raise FileExistsError("Study already frozen")
    config=read_json(config_path); validate_config(config)
    binding=baseline_binding(baseline_root)
    from .ifeval_v2_dataset import verify_frozen
    evaluator_root=Path(evaluator_root).resolve(); final=verify_frozen(evaluator_root)
    if final["sha256"] != config["final_set"]["manifest_sha256"]:
        raise ValueError("Final-set manifest differs from frozen config")
    files={str(Path(config_path).resolve()):file_sha256(config_path),str(Path(protocol_path).resolve()):file_sha256(protocol_path),
           str(evaluator_root/"manifest.json"):file_sha256(evaluator_root/"manifest.json"),**binding["files"]}
    files.update({str(p.resolve()):file_sha256(p) for p in Path(__file__).parent.glob("*.py")})
    converter_script=Path("scripts/browser_convert.py").resolve()
    files[str(converter_script)]=file_sha256(converter_script)
    benchmark=Path("artifacts/ifeval-browser-v1-heldout-continuation/benchmark").resolve()
    for p in benchmark.rglob("*"):
        if p.is_file() and not p.is_symlink() and not any(part in p.parts for part in ("node_modules","models",".git")):
            files[str(p)]=file_sha256(p)
    study={"benchmark_source":str(benchmark),"schema_version":1,"kind":"frozen_ifeval_browser_v2","config":config,"protocol_sha256":file_sha256(protocol_path),
        "baseline":binding,"evaluator_root":str(evaluator_root),"files":files,
        "custody":"procedural separation; technical isolation unverified","execution_authorized_by_preparation":False}
    save(root/"study.json",study); save(root/"ready.json",{"study_sha256":canonical_hash(study)})
    atomic(root/"state.json",{"schema_version":1,"study_sha256":canonical_hash(study),"submissions":{},"slots":{
        f"r{r}-{method}":{"round":r,"method":method,"status":"pending"} for r in range(1,5) for method in ("astra","control")},
        "stages":[],"active":None,"halted":False,"selection":None,"final":{}})
    return {"prepared":True,"study_sha256":canonical_hash(study),"production_stages":0,"dispatches":0}


def load(root, *, authorized=False, allow_expired=False):
    root=Path(root).resolve(); study=read_json(root/"study.json"); state=read_json(root/"state.json")
    if read_json(root/"ready.json") != {"study_sha256":canonical_hash(study)} or state["study_sha256"] != canonical_hash(study):
        raise ValueError("Study state binding changed")
    validate_config(study["config"])
    if authorized:
        auth=read_json(root/"execution-authorization.json") if (root/"execution-authorization.json").exists() else {}
        if auth.get("study_sha256") != canonical_hash(study) or auth.get("execution_authorized") is not True:
            raise PermissionError("Later explicit execution authorization is required; preparation does not authorize inference or edits")
        if file_sha256(root/"authorization-source.txt") != auth["authorization_source_sha256"]:
            raise ValueError("Execution authorization changed")
        if not allow_expired and time.time()*1000 >= study["config"]["submission_deadline_epoch_ms"]: raise TimeoutError("Seven-day study window expired")
        if state["halted"]: raise RuntimeError("Study stopped: unresolved cleanup or active-budget exhaustion")
    clock=None
    # Integrity checks during production are local processing, not free work.
    # Expired imports must still be recorded as terminal invalid submissions.
    charge=authorized and not (allow_expired and time.time()*1000 >= study["config"]["submission_deadline_epoch_ms"])
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
    left=min(STAGES[stage]-totals[stage],14400-sum(totals.values()),
        (study["config"]["submission_deadline_epoch_ms"]-time.time()*1000)/1000,
        study["config"]["per_attempt_seconds"][stage])
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
        if any(state["slots"][f"r{r}-{m}"]["status"] not in TERMINAL for m in ("astra","control")):
            raise ValueError("Earlier paired round must close before feedback or next proposal")


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
    context=local_stage(root,study,state,f"r{round_number}:export") if round_number > 1 else nullcontext()
    with context:
        base=Path(study["baseline"]["baseline_root"])
        old=read_json(base/"handoff/request.json")
        # Explicit allowlist. No recursively copied final, control, authorization,
        # host path, or tool-command records can enter an editor packet.
        keys=("checkpoint","policy","approved_weight_information","scoring_policy","primary_metric","secondary_metrics",
            "aggregate_metrics","development","proposal_contract","response_contract","validation","restrictions")
        fresh_feedback=baseline_feedback(read_json(base/"data/development-items.json"),read_json(base/"baseline/graded.json")["records"],read_json(base/"baseline/graded.json")["grades"],POLICY)
        if (old["development"] != fresh_feedback or old["approved_weight_information"] != read_json(base/"weight-information/inspections.json")
                or old["scoring_policy"] != POLICY or old["policy"] != read_json(base/"weight-information/policy.json")):
            raise ValueError("Saved packet differs from verified development baseline")
        request={k:_identity_only(old[k]) for k in keys}
        request.update(schema_version=1,kind="ifeval_browser_v2_manual_request",study_id="ifeval-browser-v2",
            request_id=str(uuid.uuid4()),round=round_number,study_sha256=canonical_hash(study),protocol_sha256=study["protocol_sha256"],
            created_at_epoch_seconds=study["config"]["freeze_epoch_ms"]/1000,
            submission_expires_at_epoch_seconds=study["config"]["submission_deadline_epoch_ms"]/1000,
            editor=study["config"]["editor"],editor_isolation="unverified",settings=SETTINGS,
            edit_family=study["config"]["edit_family"],final_set={k:study["config"]["final_set"][k] for k in ("prompts","instructions","supported_types","families","manifest_sha256")},
            objective="Improve strict prompt compliance on the fixed development set within the frozen browser budget.",
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
            if slot.get("feedback_withheld"):
                row["feedback_withheld"]="Matching artifact evaluated only in the control arm; control outcomes are withheld"
            else: row["strict_passes"]=slot.get("strict_passes")
            request["previous_astra_rounds"].append(row)
            feedback=root/"slots"/f"r{r}-astra"/"feedback.json"
            if slot["status"] in {"completed","duplicate"} and not slot.get("feedback_withheld") and not feedback.exists():
                raise ValueError("Completed Astra round lacks required frozen development feedback")
            if feedback.exists() and not slot.get("feedback_withheld"):
                value=read_json(feedback)
                if (file_sha256(feedback) != slot.get("feedback_file_sha256")
                        or set(value) != {"round","aggregate_metrics","development"} or value["round"] != r):
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
    from .ifeval_v2_controls import validate_v2_proposal,generate_control
    root,study,state=load(root,authorized=True,allow_expired=True); round_ready(state,round_number)
    if not Path(response_path).is_file(): raise ValueError("Missing response; no submission consumed")
    if state["submissions"].get(str(round_number),{}).get("status") != "requested": raise ValueError("Record begin-turn before submitting; slot is consumed before manual dispatch")
    packet=root/"editor"/f"round-{round_number}"; request=read_json(packet/"request.json")
    ready=read_json(packet/"ready.json")
    if ready != {"request_sha256":canonical_hash(request),"packet_sha256":file_sha256(packet/"editor-packet.txt")}:
        raise ValueError("Frozen editor packet changed")
    state["submissions"][str(round_number)]["status"]="consumed"
    for method in ("astra","control"): state["slots"][f"r{round_number}-{method}"]["status"]="invalid"
    atomic(root/"state.json",state)  # Consume before any parsing, even malformed JSON.
    with Path(response_path).open("rb") as stream: raw=stream.read(MAX_RESPONSE_BYTES+1)
    original=packet/"original-response.json"; original.write_bytes(raw); original.chmod(0o600)
    try:
        if time.time()*1000 >= study["config"]["submission_deadline_epoch_ms"]: raise ValueError("Expired response; submission remains consumed")
        with local_stage(root,study,state,f"r{round_number}:response"):
            if len(raw)>MAX_RESPONSE_BYTES: raise ValueError("Response exceeds frozen byte limit; bounded original prefix retained")
            response=read_json_bounded(original,max_bytes=MAX_RESPONSE_BYTES)
            proposal=validate_v2_proposal(validate_response(response,request),BASE_SOURCE)
            control,audit=generate_control(proposal,round_number)
            for method,value in (("astra",proposal),("control",control)):
                target=root/"slots"/f"r{round_number}-{method}"; target.mkdir(parents=True,mode=0o700)
                save(target/"proposal.json",value)
                if method=="control": save(target/"control-audit.json",audit)
                state["slots"][f"r{round_number}-{method}"].update(status="accepted",proposal_sha256=canonical_hash(value))
            state["submissions"][str(round_number)]["status"]="accepted"
    except Exception as exc:
        state["submissions"][str(round_number)].update(status="invalid",error=type(exc).__name__+": "+str(exc))
        state["slots"][f"r{round_number}-astra"]["error"]=str(exc)
        atomic(root/"state.json",state)
        return {"accepted":False,"slot_consumed":True,"error":str(exc)}
    atomic(root/"state.json",state)
    return {"accepted":True,"round":round_number,"control_frozen_before_outcomes":True}


def _slot(state,round_number,method):
    if method not in ("astra","control"): raise ValueError("Unknown method")
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
    from .ifeval_v2_controls import validate_v2_proposal, verify_control
    validate_v2_proposal(proposal,BASE_SOURCE,provenance="editor_proposal" if method=="astra" else "engineering_edit")
    if method=="control": verify_control(proposal,read_json(target/"control-audit.json"))
    conversion_manifest=Path("artifacts/browser-integration-v1/conversion/artifacts.json").resolve()
    provenance="editor_proposal" if method=="astra" else "engineering_edit"
    control=None if method=="astra" else read_json(target/"control-audit.json")
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


def begin_turn(root,round_number):
    """Record the manual proposal slot BEFORE pasting its packet to Astra."""
    root,study,state=load(root,authorized=True); round_ready(state,round_number)
    if str(round_number) in state["submissions"]: raise ValueError("Editor submission slot already consumed")
    if not (root/"editor"/f"round-{round_number}"/"request.json").exists(): raise ValueError("Export this round packet first")
    state["submissions"][str(round_number)]={"status":"requested","started_epoch_seconds":time.time(),
        "frontier_usage":None,"immutable_model_version":None,"observed_effort":None,"marginal_billing":None}
    for method in ("astra","control"): state["slots"][f"r{round_number}-{method}"]["status"]="awaiting_response"
    atomic(root/"state.json",state)
    return {"round":round_number,"submission_consumed":True,"automatic_frontier_requests":0}


def close_turn(root,round_number,reason):
    root,study,state=load(root)
    submission=state["submissions"].get(str(round_number))
    if not submission or submission["status"] != "requested": raise ValueError("Only an unresolved requested turn may close")
    submission.update(status="invalid",error=reason)
    for method in ("astra","control"): state["slots"][f"r{round_number}-{method}"].update(status="invalid",error=reason)
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
    shutil.copytree(source,destination,symlinks=True,ignore=shutil.ignore_patterns(".git","node_modules","models","results",".DS_Store"))
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
    session.mkdir(parents=True,mode=0o700); (session/"benchmark").symlink_to(benchmark,target_is_directory=True)
    items_path=Path(study["evaluator_root"])/"final-items.json" if final else baseline/"data/development-items.json"
    items=read_json(items_path)
    if final:
        from .ifeval_v2_dataset import verify_frozen
        verify_frozen(Path(study["evaluator_root"]))
    config.update(runId="ifeval-browser-v2-"+key,arm="baseline" if artifact is None else "candidate")
    config.pop("configurationSha256",None)
    if final:
        config["evaluationStage"]="heldout"
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
    # Existing served client includes the frozen 32/128 schedule support.
    for row in config["servedBuild"]:
        row["sha256"]=file_sha256(benchmark/"public"/urlparse(row["url"]).path.lstrip("/"))
    config["configurationSha256"]=canonical_hash(config)
    save(session/"baseline-config.json",config)
    files={str(session/"baseline-config.json"):file_sha256(session/"baseline-config.json"),str(items_path):file_sha256(items_path)}
    files.update({str(p):file_sha256(p) for p in benchmark.rglob("*") if p.is_file() and not p.is_symlink() and "models" not in p.parts and "node_modules" not in p.parts})
    manifest={"schema_version":1,"kind":"ifeval_v2_session","arm":config["arm"],"run_id":config["runId"],"slot":key,
        "study_manifest_sha256":canonical_hash(study),"configuration_sha256":config["configurationSha256"],
        "artifact_sha256":config["convertedArtifactSha256"],"checkpoint":{**read_json(baseline/"manifest.json")["checkpoint"],"sha256":config["sourceCheckpointSha256"]},
        "scheduled_prompts":len(items),"scheduled_instructions":sum(len(i["target"]["instruction_id_list"]) for i in items),
        "items_path":str(items_path),"items_sha256":canonical_hash(items),"settings":SETTINGS,"policy":POLICY,
        "grader":study["baseline"]["grader"],"official_ifeval_root":study["baseline"]["official_ifeval_root"],
        "official_assets":study["baseline"]["official_assets"],"files":files,
        "legacy_transport_discriminator":"study=ifeval-browser-v1 reuses unchanged client schema; run_id and evidence are V2"}
    save(session/"manifest.json",manifest); save(session/"ready.json",{"manifest_sha256":canonical_hash(manifest)})
    with runtime.ledger(session,manifest,create=True) as book: save(session/"accounting-before.json",book.summary())
    return session


def _evaluate_session(root,study,state,key,artifact,*,final=False):
    from . import ifeval_v2_runtime as runtime
    started,seconds=budget_start(root,study,state,key+":preparation","browser_preparation",reserve_bytes=64*1024**2)
    session=None
    try:
        with bounded_python(seconds):
            session=prepare_session(root,study,key,artifact=artifact,final=final)
        def on_load():
            nonlocal started
            budget_finish(root,state,started)
            started,remaining=budget_start(root,study,state,key+":browser","browser")
            signal.setitimer(signal.ITIMER_REAL,remaining)
            return remaining
        used=time.monotonic()-started
        with bounded_python(max(.001,seconds-used)):
            execution=runtime.execute(session,on_load=on_load,preparation_seconds=max(.001,seconds-used))
        clean=all(execution["cleanup"].get(k) is True for k in CLEANUP) and execution["cleanup"].get("retained_owned_pids_alive")==[]
        budget_finish(root,state,started,status=execution["status"],cleanup=clean); started=None
        started,seconds=budget_start(root,study,state,key+":grading","grading")
        deadline=time.monotonic()+seconds
        summary=runtime.grade(session,before_grade=lambda:_require_grading_time(deadline))
        budget_finish(root,state,started); started=None
        return session,summary
    except BaseException:
        if started is not None and state["active"] is not None:
            # A browser runner that failed before returning cannot establish
            # process closure. Subsequent arms stay blocked until audited.
            clean=session is None or not (session/"execution").exists()
            budget_finish(root,state,started,status="failed",cleanup=clean)
        raise


def _require_grading_time(deadline):
    if deadline-time.monotonic() < 20: raise TimeoutError("Insufficient grading stage allowance for one bounded checker")


def evaluate(root,round_number,method):
    root,study,state=load(root,authorized=True); key,slot=_slot(state,round_number,method)
    if slot["status"] != "built": raise ValueError("Evaluation requires a newly built candidate")
    order=study["config"]["schedule"]["development_order"][round_number-1]
    if method==order[1] and state["slots"][f"r{round_number}-{order[0]}"]["status"] not in TERMINAL:
        raise ValueError("Frozen within-round browser order changed")
    # Reuse complete grades for the same artifact; savings never transfer.
    duplicate=next((s for s in state["slots"].values() if s.get("artifact_sha256")==slot["artifact_sha256"] and s["status"] in {"completed","duplicate"}),None)
    if duplicate:
        for k in ("strict_passes","completed","technically_valid","independently_graded","scheduled_prompts","graded_prompts","session"):
            slot[k]=duplicate[k]
        slot.update(status="duplicate",reused_from=f"r{duplicate['round']}-{duplicate['method']}",feedback_withheld=method=="astra" and (duplicate["method"]=="control" or duplicate.get("feedback_withheld",False)))
    elif slot["artifact_sha256"]==BASE_ARTIFACT:
        slot.update(status="duplicate",strict_passes=16,completed=True,technically_valid=True,independently_graded=True,
            scheduled_prompts=32,graded_prompts=32,session=study["baseline"]["baseline_root"],reused_from="pristine")
    else:
        try:
            artifact=read_json(slot["artifact_path"])
            if artifact["artifact_sha256"] != slot["artifact_sha256"] or file_sha256(slot["artifact_path"]) != slot["artifact_file_sha256"]:
                raise ValueError("Candidate artifact differs from frozen built identity")
            session,summary=_evaluate_session(root,study,state,key,artifact)
            slot.update(status="completed" if summary["technically_valid"] else "failed",strict_passes=summary["metrics"]["prompt_level_strict_accuracy"]["passed"],
                completed=summary["execution_complete"],technically_valid=summary["technically_valid"],independently_graded=summary["semantic_grading_complete"],
                scheduled_prompts=32,graded_prompts=32-summary["unresolved_grader_errors"],session=str(session))
        except BaseException as exc: slot.update(status="failed",error=type(exc).__name__+": "+str(exc))
    atomic(root/"state.json",state)
    if method=="astra" and not slot.get("feedback_withheld") and slot["status"] in {"completed","duplicate"}:
        with local_stage(root,study,state,key+":feedback"):
            _feedback(root,study,key,slot)
            slot["feedback_file_sha256"]=file_sha256(root/"slots"/key/"feedback.json")
            atomic(root/"state.json",state)
    return {"slot":key,"status":slot["status"],"strict_passes":slot.get("strict_passes"),"error":slot.get("error")}


def _feedback(root,study,key,slot):
    session=Path(slot["session"])
    manifest=read_json(session/"manifest.json"); arm=manifest["arm"]
    envelope=read_json(session/arm/"graded.json")
    items=read_json(Path(study["baseline"]["baseline_root"])/"data/development-items.json")
    feedback=baseline_feedback(items,envelope["records"],envelope["grades"],POLICY)
    save(root/"slots"/key/"feedback.json",{"round":slot["round"],"aggregate_metrics":aggregate_feedback(feedback),"development":feedback})


def select(root):
    from .ifeval_v2_controls import select_candidate,final_arms
    root,study,state=load(root,authorized=True)
    if state["selection"] is not None: return state["selection"]
    if any(s["status"] not in TERMINAL for s in state["slots"].values()): raise ValueError("All four paired rounds must close before final identities freeze")
    with local_stage(root,study,state,"final:selection"):
        selected={m:select_candidate([s for s in state["slots"].values() if s["method"]==m],BASE_ARTIFACT) for m in ("astra","control")}
        plan=final_arms(BASE_ARTIFACT,selected["astra"],selected["control"])
        plan.update(study_sha256=canonical_hash(study),selected=selected,final_manifest_sha256=study["config"]["final_set"]["manifest_sha256"],
            frozen_before_final_results=True,created_epoch_seconds=read_json(root/"selection-intent.json")["created_epoch_seconds"] if (root/"selection-intent.json").exists() else time.time())
        for row in plan["distinct_evaluations"]:
            row["artifact_path"]=None if row["artifact_sha256"]==BASE_ARTIFACT else next(s["artifact_path"] for s in state["slots"].values() if s.get("artifact_sha256")==row["artifact_sha256"])
        for row in plan["distinct_evaluations"]:
            row["artifact_file_sha256"]=None if row["artifact_path"] is None else file_sha256(row["artifact_path"])
            if row["artifact_path"] and read_json(row["artifact_path"])["artifact_sha256"] != row["artifact_sha256"]:
                raise ValueError("Selected package differs from selected development identity")
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","export","authorize","begin-turn","close-turn","import","build","evaluate","select","final","analyze","status","recover"])
    parser.add_argument("--root",required=True); parser.add_argument("--round",type=int); parser.add_argument("--method",choices=["astra","control"])
    parser.add_argument("--baseline-root",default="artifacts/ifeval-browser-v1"); parser.add_argument("--evaluator-root",default="artifacts/ifeval-browser-v2/evaluator")
    parser.add_argument("--config",default="configs/ifeval_browser_v2.json"); parser.add_argument("--protocol",default="docs/ifeval-browser-v2-protocol.md")
    parser.add_argument("--response"); parser.add_argument("--reason"); parser.add_argument("--authorization"); parser.add_argument("--approved-study-sha256")
    parser.add_argument("--converter-python",default="artifacts/browser-integration-v1/conversion/venv/bin/python"); parser.add_argument("--arm")
    args=parser.parse_args(); root=Path(args.root); root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with _lock(root):
        if args.command=="prepare": result=prepare(root,args.baseline_root,args.evaluator_root,args.config,args.protocol)
        elif args.command=="export": result=export_packet(root,args.round)
        elif args.command=="authorize": result=authorize(root,args.authorization,args.approved_study_sha256)
        elif args.command=="begin-turn": result=begin_turn(root,args.round)
        elif args.command=="close-turn": result=close_turn(root,args.round,args.reason)
        elif args.command=="import": result=import_response(root,args.round,args.response)
        elif args.command=="build": result=build(root,args.round,args.method,args.converter_python)
        elif args.command=="evaluate": result=evaluate(root,args.round,args.method)
        elif args.command=="select": result=select(root)
        elif args.command=="final": result=evaluate_final(root,args.arm)
        elif args.command=="analyze":
            from .ifeval_v2_analysis import analyze
            root,study,state=load(root,authorized=True)
            clock,seconds=budget_start(root,study,state,"final:analysis","analysis_export")
            try:
                with bounded_python(seconds): result=analyze(root)
            finally: budget_finish(root,state,clock)
        elif args.command=="recover": result=recover(root)
        else:
            _,study,state=load(root); result={"study_sha256":canonical_hash(study),"submissions":state["submissions"],"slots":state["slots"],"stages":state["stages"],"active":state["active"],"halted":state["halted"],"selection":state["selection"]}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__": main()
