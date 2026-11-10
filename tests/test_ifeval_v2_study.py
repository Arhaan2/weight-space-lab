"""Synthetic V2 workflow only: no editor calls, model loads, weight edits or conversion."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from weight_space_lab import ifeval_v2_study as study
from weight_space_lab import ifeval_v2_runtime as runtime
from weight_space_lab.ifeval_v2_controls import TENSORS
from weight_space_lab.inventory import read_json
from weight_space_lab.protocol import canonical_hash


class V2StudyTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)/"v2"; self.root.mkdir()
        self.base=Path(tmp.name)/"baseline"; self.base.mkdir()
        cfg=read_json("configs/ifeval_browser_quality_resource.json")
        cfg["final_set"]["prompts"]=128; cfg["final_set"]["instructions"]=160
        cfg["final_set"]["manifest_sha256"]="f"*64
        self.feedback=[{"item_id":f"dev-{i}","prompt":f"Synthetic development prompt {i}","raw_response":"SYNTHETIC_DEV", "strict_prompt_compliant":i<16,
            "status":"ok","constraints":[{"strict":i<16,"loose":True}]} for i in range(32)]
        self.write(self.base/"data/development-items.json",[{"item_id":f"dev-{i}"} for i in range(32)])
        self.write(self.base/"baseline/graded.json",{"records":[],"grades":[]})
        self.write(self.base/"weight-information/policy.json",{})
        self.write(self.base/"weight-information/inspections.json",{})
        self.write(self.base/"weight-information/host.json",{"model_path":"synthetic-pristine"})
        old={k:{} for k in ("checkpoint","policy","approved_weight_information","proposal_contract","validation")}
        old.update(scoring_policy=study.POLICY,primary_metric="prompt_level_strict_accuracy",secondary_metrics=[],
            aggregate_metrics={},development=self.feedback,restrictions=[],
            response_contract={"operation_forms":{"scale_columns":{"required_keys":["op","tensor","columns","factor"]}}},
            final={"prompt":"SECRET_FINAL"},controls={"output":"SECRET_CONTROL"},authorization="SECRET_AUTHORIZATION")
        self.write(self.base/"handoff/request.json",old)
        self.manifest={"config":cfg,"files":{},"protocol_sha256":"c"*64,"baseline":{"baseline_root":str(self.base),"artifact_sha256":study.BASE_ARTIFACT,"scored_output_tokens":13941},"evaluator_root":str(self.root/"evaluator")}
        self.state={"study_sha256":canonical_hash(self.manifest),"submissions":{},"slots":{f"r{r}-{m}":{"round":r,"method":m,"status":"pending"} for r in range(1,5) for m in ("astra",)},"stages":[],"active":None,"halted":False,"selection":None,"final":{},"resource":{},"generation_sessions":{},"first_packet_export_epoch_seconds":time.time()}
        self.write(self.root/"study.json",self.manifest); self.write(self.root/"ready.json",{"study_sha256":canonical_hash(self.manifest)})
        self.write(self.root/"state.json",self.state)
        self.auth=self.root/"later-authorization.txt"; self.auth.write_text("Synthetic explicit future execution authorization")
        study.authorize(self.root,self.auth,canonical_hash(self.manifest))
        self.feedback_patch=patch.object(study,"baseline_feedback",return_value=self.feedback); self.feedback_patch.start(); self.addCleanup(self.feedback_patch.stop)
        self.space_patch=patch.object(study.shutil,"disk_usage",return_value=type("Disk",(),{"free":100*1024**3})()); self.space_patch.start(); self.addCleanup(self.space_patch.stop)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))

    def submission(self,r,*,start=None):
        study.export_packet(self.root,r); study.begin_turn(self.root,r)
        request=read_json(self.root/"editor"/f"round-{r}"/"request.json")
        proposal={"schema_version":1,"base_sha256":study.BASE_SOURCE,"provenance":"editor_proposal","composition":"sequential",
            "operations":[{"op":"scale_columns","tensor":t,"columns":list(range(r if start is None else start,(r if start is None else start)+60)),"factor":.75} for t in TENSORS]}
        response={"schema_version":1,"request_id":request["request_id"],"request_sha256":canonical_hash(request),"proposal":proposal}
        path=self.root/f"response-{r}.json"; self.write(path,response)
        return study.import_response(self.root,r,path)

    def fake_package(self,original,candidate,baseline,converted,output,**kwargs):
        key=Path(candidate).parent.name; identity=canonical_hash({"synthetic":key})
        path=Path(output)/"candidate-artifact.json"; self.write(path,{"artifact_sha256":identity,"synthetic":True})
        return {"artifact_manifest":str(path),"artifact_sha256":identity,"conversion_erases_edit":False}

    def fake_session(self,root,manifest,state,key,artifact,*,final=False):
        score={"r1-astra":18,"r2-astra":19,"r3-astra":19,"r4-astra":17,"r1-control":17,"r2-control":16,"r3-control":19,"r4-control":20}.get(key,64)
        session=root/"sessions"/key; session.mkdir(parents=True)
        return session,{"technically_valid":True,"execution_complete":True,"semantic_grading_complete":True,
            "unresolved_grader_errors":0,"known_generated_tokens":13000,"deployment_limits_satisfied":True,"memory_limits_satisfied":True,"metrics":{"prompt_level_strict_accuracy":{"passed":score}}}

    def fake_feedback(self,root,manifest,key,slot):
        self.write(root/"slots"/key/"feedback.json",{"round":slot["round"],"aggregate_metrics":{},"development":[{"raw_response":"SYNTHETIC_ASTRA_ONLY"}]})

    def test_four_round_production_transitions_with_mocked_heavy_backends(self):
        def convert(*args,**kwargs): kwargs["on_conversion"](); return {}
        with patch("weight_space_lab.editor_tools.run_checkpoint_action",return_value={"result_sha256":"a"*64}) as importer, \
             patch("weight_space_lab.ifeval_v2_candidate.convert",side_effect=convert) as converter, \
             patch("weight_space_lab.ifeval_v2_candidate.package_candidate",side_effect=self.fake_package) as package, \
             patch.object(study,"_evaluate_session",side_effect=self.fake_session) as evaluate, \
             patch.object(study,"_feedback",side_effect=self.fake_feedback), \
             patch("weight_space_lab.ifeval_v2_analysis.freeze_analysis_plan",return_value={}), \
             patch("weight_space_lab.browser_resource.freeze_comparison",return_value="d"*64):
            for r in range(1,5):
                self.assertTrue(self.submission(r)["accepted"])
                for method in ("astra",):
                    self.assertEqual(study.build(self.root,r,method,"synthetic-converter")["status"],"built")
                for method in self.manifest["config"]["schedule"]["development_order"][r-1]:
                    self.assertEqual(study.evaluate(self.root,r,method)["status"],"completed")
                # Re-open the durable files, as a fresh process would.
                self.assertEqual(len(study.load(self.root)[2]["submissions"]),r)
            selection=study.select(self.root)
            self.assertEqual(selection["selected"]["astra"]["round"],2)
            for row in selection["distinct_evaluations"]:
                self.assertEqual(study.evaluate_final(self.root,row["dispatch_arm"])["status"],"completed")
            self.assertEqual((importer.call_count,converter.call_count,package.call_count,evaluate.call_count),(4,4,4,6))
            with self.assertRaises(ValueError): study.evaluate_final(self.root,"pristine_baseline")
            with self.assertRaises(ValueError): study.export_packet(self.root,4)
            for path in (self.root/"editor").rglob("editor-packet.txt"):
                packet=path.read_text()
                for forbidden in ("SECRET_FINAL","SECRET_CONTROL","SECRET_AUTHORIZATION",str(self.root),str(self.base)):
                    self.assertNotIn(forbidden,packet)
                self.assertIn("SYNTHETIC_DEV",packet)
            packet=(self.root/"editor/round-4/editor-packet.txt").read_text()
            self.assertIn("SYNTHETIC_ASTRA_ONLY",packet)
            request=read_json(self.root/"editor/round-4/request.json")
            self.assertEqual(request["previous_astra_rounds"][0]["proposal"]["operations"][0]["columns"],list(range(1,61)))
            self.assertEqual(request["previous_astra_rounds"][0]["proposal"]["operations"][0]["factor"],.75)

    def test_invalid_turns_are_consumed_and_final_falls_back_once(self):
        study.export_packet(self.root,1)
        request=read_json(self.root/"editor/round-1/request.json")
        invalid=self.root/"invalid.json"; invalid.write_text("not JSON")
        with self.assertRaisesRegex(ValueError,"begin-turn"): study.import_response(self.root,1,invalid)
        study.begin_turn(self.root,1)
        self.assertFalse(study.import_response(self.root,1,invalid)["accepted"])
        with self.assertRaises(ValueError): study.begin_turn(self.root,1)
        with self.assertRaises(ValueError): study.import_response(self.root,1,invalid)
        for r in range(2,5):
            study.export_packet(self.root,r); study.begin_turn(self.root,r); study.close_turn(self.root,r,"abandoned synthetic manual turn")
        with patch("weight_space_lab.ifeval_v2_analysis.freeze_analysis_plan",return_value={}), patch.object(study,"_evaluate_session",side_effect=self.fake_session) as execute:
            selection=study.select(self.root)
            self.assertEqual(len(selection["distinct_evaluations"]),0)
            with self.assertRaises(ValueError): study.evaluate_final(self.root,"pristine_baseline")
            with self.assertRaises(ValueError): study.evaluate_final(self.root,"selected_astra")
            self.assertEqual(execute.call_count,0)
        self.assertEqual(request["submission_expires_at_epoch_seconds"]-request["created_at_epoch_seconds"],604800)

    def test_sequence_and_duplicate_dispatch_guards(self):
        with self.assertRaisesRegex(ValueError,"Earlier round"): study.export_packet(self.root,2)
        self.assertTrue(self.submission(1)["accepted"])
        state=read_json(self.root/"state.json")
        with self.assertRaisesRegex(ValueError,"Only Astra"):
            study.evaluate(self.root,1,"control")
        started,_=study.budget_start(self.root,self.manifest,state,"fixture:once","imports")
        with self.assertRaisesRegex(RuntimeError,"Interrupted/active"): study.budget_start(self.root,self.manifest,state,"fixture:twice","imports")
        study.budget_finish(self.root,state,started)
        with self.assertRaisesRegex(ValueError,"already consumed"): study.budget_start(self.root,self.manifest,state,"fixture:once","imports")

    def test_duplicate_reuses_completed_development_without_another_dispatch(self):
        state=read_json(self.root/"state.json")
        for slot in state["slots"].values(): slot["status"]="invalid"
        state["slots"]["r1-astra"].update(status="completed",artifact_sha256="a"*64,strict_passes=20,completed=True,
            scored_output_tokens=13000,deployment_limits_satisfied=True,memory_limits_satisfied=True,
            technically_valid=True,independently_graded=True,scheduled_prompts=32,graded_prompts=32,session="synthetic")
        state["slots"]["r2-astra"].update(status="built",artifact_sha256="a"*64)
        study.atomic(self.root/"state.json",state)
        with patch.object(study,"_feedback",side_effect=self.fake_feedback),patch.object(study,"_evaluate_session",side_effect=AssertionError("Duplicate dispatch")):
            self.assertEqual(study.evaluate(self.root,2,"astra")["status"],"duplicate")
        self.assertEqual(read_json(self.root/"state.json")["slots"]["r2-astra"]["scored_output_tokens"],13000)

    def test_storage_exhaustion_preserves_protected_evidence(self):
        protected=self.base/"must-survive"; protected.write_text("protected V1")
        with patch.object(study,"retained_bytes",return_value=32*1024**3):
            with self.assertRaisesRegex(RuntimeError,"retention/free-space"):
                study.budget_start(self.root,self.manifest,self.state,"fixture:storage","imports",reserve_bytes=1)
        self.assertEqual(protected.read_text(),"protected V1")
        self.assertIsNone(read_json(self.root/"state.json")["active"])

    def test_budget_exhaustion_and_unconfirmed_cleanup_stop_later_stages(self):
        state=read_json(self.root/"state.json")
        state["stages"]=[{"key":"past","stage":"imports","charged_seconds":600}]
        with self.assertRaises(TimeoutError): study.budget_start(self.root,self.manifest,state,"next","imports")
        state["stages"]=[]
        started,_=study.budget_start(self.root,self.manifest,state,"browser-one","browser_development")
        study.budget_finish(self.root,state,started,status="failed",cleanup=False)
        with self.assertRaisesRegex(RuntimeError,"halted"): study.budget_start(self.root,self.manifest,state,"browser-two","browser_development")

    def test_interrupted_stage_charges_reservation_and_cannot_restart_dispatch(self):
        state=read_json(self.root/"state.json")
        _,reserved=study.budget_start(self.root,self.manifest,state,"r1-astra:browser","browser_development")
        with patch("weight_space_lab.browser_v2_runner.alive",return_value=False): result=study.recover(self.root)
        self.assertTrue(result["halted"])
        after=read_json(self.root/"state.json")
        self.assertEqual(after["stages"][-1]["charged_seconds"],reserved)
        with self.assertRaises(RuntimeError): study.begin_turn(self.root,1)

    def test_expired_response_still_closes_consumed_turn(self):
        study.export_packet(self.root,1); study.begin_turn(self.root,1)
        path=self.root/"late.json"; path.write_text("{}")
        with patch.object(study.time,"time",return_value=study.calendar_deadline(read_json(self.root/"state.json"))+1):
            result=study.import_response(self.root,1,path)
        self.assertFalse(result["accepted"])
        self.assertEqual(read_json(self.root/"state.json")["submissions"]["1"]["status"],"invalid")

    def test_runtime_grade_uses_full_32_and_128_denominators_without_legacy_plan(self):
        for n in (32,128):
            session=self.root/f"grade-{n}"; session.mkdir()
            items=[{"item_id":str(i),"split":"development" if n==32 else "test",
                "target":{"instruction_id_list":["synthetic"]*(2 if n==128 and i<32 else 1)}} for i in range(n)]
            item_path=session/"items.json"; self.write(item_path,items)
            manifest={"items_path":str(item_path),"arm":"candidate","checkpoint":{},"policy":study.POLICY,
                "official_ifeval_root":"synthetic-official","study_manifest_sha256":"a"*64}
            self.write(session/"baseline-config.json",{"runId":"synthetic-v2","configurationSha256":"b"*64})
            self.write(session/"execution/candidate-1/supervisor-result.json",{"status":"completed","outputs":[{"item_id":str(i)} for i in range(n)]})
            def adapt(item,output,**kwargs):
                return {"status":"ok","token_counts":{"output":1,"input":1},"duration_seconds":.01},{}
            def grade(record,item,policy,**kwargs):
                count=len(item["target"]["instruction_id_list"])
                return {"status":"graded","correct":True,"diagnostics":{"code":"synthetic"},"metrics":{
                    "prompt_level_strict_accuracy":True,"prompt_level_loose_accuracy":True,
                    "instruction_strict":[True]*count,"instruction_loose":[True]*count}}
            checked=[]
            with patch.object(runtime,"load",return_value=(session,manifest)),patch.object(runtime,"development_adapter",side_effect=adapt), \
                 patch.object(runtime,"adapt_browser_output",side_effect=adapt),patch.object(runtime,"grade_record",side_effect=grade):
                summary=runtime.grade(session,before_grade=lambda:checked.append(True))
            self.assertEqual(len(checked),n)
            self.assertEqual(summary["scheduled_instructions"],32 if n==32 else 160)
            self.assertEqual(summary["metrics"]["prompt_level_strict_accuracy"]["passed"],n)
            self.assertTrue((session/"candidate/graded.json").exists())

    def test_shortened_final_prepare_session_binds_actual_browser_schedule(self):
        from weight_space_lab.grading import grader_identity
        benchmark=self.root/"benchmark"
        for relative in ("public/task-evaluation.html","public/task-evaluation.js","public/benchmark-worker/benchmark.worker.js"):
            path=benchmark/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("synthetic runtime fixture")
        self.write(self.base/"manifest.json",{"checkpoint":{"sha256":study.BASE_SOURCE,"revision":"synthetic"}})
        self.write(self.base/"baseline-config.json",{"settings":study.SETTINGS,"study":"ifeval-browser-v1",
            "model":{"modelUrl":"http://127.0.0.1:4198/models/pristine/"},"sourceCheckpointSha256":study.BASE_SOURCE,
            "convertedArtifactSha256":study.BASE_ARTIFACT,"items":[],"servedBuild":[]})
        self.manifest["baseline"].update(grader=grader_identity("ifeval"),official_ifeval_root="synthetic-official",official_assets={})
        self.manifest["config"]["final_set"]["prompts"]=3
        items=[{"item_id":f"final-{i}","task_text":f"synthetic private final {i}","split":"test",
            "target":{"instruction_id_list":["synthetic"]},"metadata":{"rendered_input_tokens":10}} for i in range(3)]
        self.write(Path(self.manifest["evaluator_root"])/"final-items.json",items)
        with patch("weight_space_lab.ifeval_v2_dataset.verify_frozen",return_value={}), \
             patch.object(runtime,"verify_ifeval_assets",return_value={}):
            session=study.prepare_session(self.root,self.manifest,"short-final",final=True)
            loaded,manifest=runtime.load(session)
        config=read_json(session/"baseline-config.json")
        self.assertEqual(config["narrowedFinalPromptCount"],3)
        self.assertEqual(config["evaluationStage"],"heldout")
        self.assertEqual([r["item_id"] for r in config["items"]],[r["item_id"] for r in items])
        self.assertEqual(config["configurationSha256"],canonical_hash({k:v for k,v in config.items() if k!="configurationSha256"}))
        self.assertEqual(manifest["scheduled_prompts"],3)
        with runtime.ledger(session,manifest) as ledger:
            self.assertEqual(ledger.summary()["limits"]["generated_tokens"],3*1024+16)
        self.write(Path(self.manifest["evaluator_root"])/"final-items.json",[])
        self.manifest["config"]["final_set"]["prompts"]=0
        with patch("weight_space_lab.ifeval_v2_dataset.verify_frozen",return_value={}):
            with self.assertRaisesRegex(ValueError,"nonempty frozen prompt count"):
                study.prepare_session(self.root,self.manifest,"zero-final",final=True)
        self.assertFalse((self.root/"sessions/zero-final").exists())

    def test_prepare_snapshots_exact_current_source_and_built_runtime(self):
        config=read_json("configs/ifeval_browser_quality_resource.json")
        root=self.root/"prepared"; root.mkdir()
        evaluator=root/"evaluator"; self.write(evaluator/"manifest.json",{"fixture":True})
        benchmark=root/"benchmark"; benchmark.mkdir(); (benchmark/"worker.js").write_text("synthetic built runtime")
        binding={"files":{},"baseline_root":str(self.base),"scored_output_tokens":13941}
        frozen={"sha256":config["final_set"]["manifest_sha256"],"counts":config["final_set"]}
        with patch.object(study,"baseline_binding",return_value=binding), \
             patch("weight_space_lab.ifeval_v2_dataset.verify_frozen",return_value=frozen), \
             patch.object(study,"prepare_benchmark",return_value=benchmark), \
             patch("weight_space_lab.browser_resource.install_adapter"), \
             patch("weight_space_lab.browser_resource.inspect_preset",return_value={"fixture":True}), \
             patch.object(study.subprocess,"run"):
            study.prepare(root,self.base,evaluator,"configs/ifeval_browser_quality_resource.json",
                "docs/ifeval-browser-quality-resource-protocol.md")
        manifest=read_json(root/"study.json")
        self.assertIn(str(benchmark/"worker.js"),manifest["files"])
        snapshot=root/"source-snapshot/weight_space_lab/ifeval_v2_study.py"
        self.assertEqual(snapshot.read_bytes(),Path(study.__file__).read_bytes())
        self.assertIn(str(snapshot.resolve()),manifest["files"])
        self.assertEqual(manifest["resource_preset"],{"fixture":True})
        (benchmark/"worker.js").write_text("changed after freeze")
        with self.assertRaisesRegex(ValueError,"Frozen study evidence changed"): study.load(root)

    def test_first_export_anchors_once_and_never_exposes_final_inventory(self):
        state=read_json(self.root/"state.json"); state["first_packet_export_epoch_seconds"]=None
        study.atomic(self.root/"state.json",state)
        first=study.export_packet(self.root,1)
        anchor=read_json(self.root/"state.json")["first_packet_export_epoch_seconds"]
        with patch.object(study.time,"time",return_value=anchor+3600):
            again=study.export_packet(self.root,1)
        self.assertEqual(first["request_sha256"],again["request_sha256"])
        self.assertEqual(study.calendar_deadline(read_json(self.root/"state.json")),anchor+604800)
        request=read_json(self.root/"editor/round-1/request.json")
        self.assertNotIn("final_set",request)
        self.assertEqual(len(read_json(self.root/"state.json")["submissions"]),0)

    def test_pending_expired_original_response_consumes_one_slot_without_dispatch(self):
        study.export_packet(self.root,1)
        response=self.root/"expired.json"; response.write_text("{}")
        deadline=study.calendar_deadline(read_json(self.root/"state.json"))
        with patch.object(study.time,"time",return_value=deadline+1):
            result=study.advance(self.root,response=response)
        self.assertEqual(result["status"],"calendar_expired_response_consumed")
        self.assertEqual(study.status(self.root)["capacity_consumed_or_reserved"]["editor_submissions"],1)
        self.assertEqual(study.status(self.root)["capacity_consumed_or_reserved"]["engine_loads"],0)

    def test_zero_final_data_expires_capacity_and_stage_reservations_cannot_repeat(self):
        self.manifest["config"]["final_set"]["prompts"]=0
        cap=study.effective_capacity(self.manifest)
        self.assertEqual((cap["engine_loads"],cap["dispatches"],cap["generated_tokens"],cap["browser_seconds"]),(8,376,146560,5400))
        state=read_json(self.root/"state.json")
        for r in range(4):
            study.begin_session_capacity(self.root,self.manifest,state,f"dev-{r}",workload="development",requests=32,max_tokens=1024)
        with self.assertRaisesRegex(ValueError,"capacity exhausted"):
            study.begin_session_capacity(self.root,self.manifest,state,"dev-5",workload="development",requests=32,max_tokens=1024)
        with self.assertRaisesRegex(ValueError,"already consumed"):
            study.begin_session_capacity(self.root,self.manifest,state,"dev-0",workload="development",requests=32,max_tokens=1024)
        with self.assertRaises(ValueError):
            study.begin_session_capacity(self.root,self.manifest,state,"final",workload="final",requests=0,max_tokens=1024)
        for r in range(4):
            study.begin_session_capacity(self.root,self.manifest,state,f"resource-{r}",workload="resource",requests=60,max_tokens=64)
        self.assertEqual(sum(row["reserved_tokens"] for row in state["generation_sessions"].values()),146560)

    def test_mutated_frozen_runtime_rejected_before_next_stage(self):
        from weight_space_lab.inventory import file_sha256
        runtime_file=self.root/"runtime.js"; runtime_file.write_text("frozen")
        self.manifest["files"][str(runtime_file)]=file_sha256(runtime_file)
        self.write(self.root/"study.json",self.manifest)
        self.write(self.root/"ready.json",{"study_sha256":canonical_hash(self.manifest)})
        state=read_json(self.root/"state.json"); state["study_sha256"]=canonical_hash(self.manifest); study.atomic(self.root/"state.json",state)
        runtime_file.write_text("changed")
        with self.assertRaisesRegex(ValueError,"Frozen study evidence changed"): study.load(self.root)

    def test_real_virtual_environment_module_cli_status(self):
        import subprocess
        interpreter=Path("artifacts/ifeval-browser-v1/evaluator-env/bin/python").absolute()
        if not interpreter.exists(): self.skipTest("private evaluator environment not installed")
        result=subprocess.run([str(interpreter),"-m","weight_space_lab.ifeval_v2_study","status","--root",str(self.root)],
            check=True,capture_output=True,text=True,timeout=30)
        value=json.loads(result.stdout)
        self.assertEqual(value["effective_capacity"]["dispatches"],634)
        self.assertEqual(value["capacity_consumed_or_reserved"]["engine_loads"],0)

    def test_runtime_intent_uses_exact_dynamic_schedule_and_identity(self):
        for n in (1,27,32,128):
            config={"items":[{"item_id":str(i)} for i in range(n)],"arm":"candidate","configurationSha256":"a"*64,"convertedArtifactSha256":"b"*64,"runId":"synthetic-v2"}
            expected={"kind":"generation","arm":"candidate","request_id":f"generation-{n+1}","attempt_id":"candidate-1",
                "code_version_sha256":"c"*64,"configuration_sha256":"a"*64,"artifact_sha256":"b"*64,"run_id":"synthetic-v2",
                "warmup":False,"max_tokens":1024,"sequence_index":n,"item_id":str(n-1)}
            self.assertEqual(runtime.validate_intent(expected,config,"c"*64,n+1),expected)
            with self.assertRaises(ValueError): runtime.validate_intent(expected,config,"c"*64,n+2)
            expected["max_tokens"]=1025
            with self.assertRaises(ValueError): runtime.validate_intent(expected,config,"c"*64,n+1)


if __name__=="__main__": unittest.main()
