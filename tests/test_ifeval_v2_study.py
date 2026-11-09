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
        cfg=read_json("configs/ifeval_browser_v2.json")
        cfg["freeze_epoch_ms"]=time.time()*1000; cfg["submission_deadline_epoch_ms"]=cfg["freeze_epoch_ms"]+604800000
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
        self.manifest={"config":cfg,"files":{},"protocol_sha256":"c"*64,"baseline":{"baseline_root":str(self.base),"artifact_sha256":study.BASE_ARTIFACT},"evaluator_root":str(self.root/"evaluator")}
        self.state={"study_sha256":canonical_hash(self.manifest),"submissions":{},"slots":{f"r{r}-{m}":{"round":r,"method":m,"status":"pending"} for r in range(1,5) for m in ("astra","control")},"stages":[],"active":None,"halted":False,"selection":None,"final":{}}
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
            "unresolved_grader_errors":0,"metrics":{"prompt_level_strict_accuracy":{"passed":score}}}

    def fake_feedback(self,root,manifest,key,slot):
        self.write(root/"slots"/key/"feedback.json",{"round":slot["round"],"aggregate_metrics":{},"development":[{"raw_response":"SYNTHETIC_ASTRA_ONLY"}]})

    def test_four_round_production_transitions_with_mocked_heavy_backends(self):
        def convert(*args,**kwargs): kwargs["on_conversion"](); return {}
        with patch("weight_space_lab.editor_tools.run_checkpoint_action",return_value={"result_sha256":"a"*64}) as importer, \
             patch("weight_space_lab.ifeval_v2_candidate.convert",side_effect=convert) as converter, \
             patch("weight_space_lab.ifeval_v2_candidate.package_candidate",side_effect=self.fake_package) as package, \
             patch.object(study,"_evaluate_session",side_effect=self.fake_session) as evaluate, \
             patch.object(study,"_feedback",side_effect=self.fake_feedback), \
             patch("weight_space_lab.ifeval_v2_analysis.freeze_analysis_plan",return_value={}):
            for r in range(1,5):
                self.assertTrue(self.submission(r)["accepted"])
                for method in ("astra","control"):
                    self.assertEqual(study.build(self.root,r,method,"synthetic-converter")["status"],"built")
                for method in self.manifest["config"]["schedule"]["development_order"][r-1]:
                    self.assertEqual(study.evaluate(self.root,r,method)["status"],"completed")
                # Re-open the durable files, as a fresh process would.
                self.assertEqual(len(study.load(self.root)[2]["submissions"]),r)
            selection=study.select(self.root)
            self.assertEqual(selection["selected"]["astra"]["round"],2)
            self.assertEqual(selection["selected"]["control"]["round"],4)
            for row in selection["distinct_evaluations"]:
                self.assertEqual(study.evaluate_final(self.root,row["dispatch_arm"])["status"],"completed")
            self.assertEqual((importer.call_count,converter.call_count,package.call_count,evaluate.call_count),(8,8,8,11))
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
            self.assertEqual(len(selection["distinct_evaluations"]),1)
            self.assertEqual(study.evaluate_final(self.root,"pristine_baseline")["status"],"completed")
            with self.assertRaises(ValueError): study.evaluate_final(self.root,"selected_astra")
            self.assertEqual(execute.call_count,1)
        self.assertEqual(request["submission_expires_at_epoch_seconds"]-request["created_at_epoch_seconds"],604800)

    def test_sequence_and_duplicate_dispatch_guards(self):
        with self.assertRaisesRegex(ValueError,"Earlier paired round"): study.export_packet(self.root,2)
        self.assertTrue(self.submission(1)["accepted"])
        state=read_json(self.root/"state.json"); state["slots"]["r1-control"].update(status="built",artifact_sha256="d"*64)
        study.atomic(self.root/"state.json",state)
        with self.assertRaisesRegex(ValueError,"within-round"): study.evaluate(self.root,1,"control")
        started,_=study.budget_start(self.root,self.manifest,state,"fixture:once","imports")
        with self.assertRaisesRegex(RuntimeError,"Interrupted/active"): study.budget_start(self.root,self.manifest,state,"fixture:twice","imports")
        study.budget_finish(self.root,state,started)
        with self.assertRaisesRegex(ValueError,"already consumed"): study.budget_start(self.root,self.manifest,state,"fixture:once","imports")

    def test_control_duplicate_reuse_never_enters_astra_feedback(self):
        state=read_json(self.root/"state.json")
        for slot in state["slots"].values(): slot["status"]="invalid"
        state["slots"]["r1-control"].update(status="completed",artifact_sha256="a"*64,strict_passes=31,completed=True,
            technically_valid=True,independently_graded=True,scheduled_prompts=32,graded_prompts=32,session="SECRET_CONTROL_SESSION")
        state["slots"]["r2-astra"].update(status="built",artifact_sha256="a"*64)
        study.atomic(self.root/"state.json",state)
        with patch.object(study,"_feedback",side_effect=AssertionError("Control leak")),patch.object(study,"_evaluate_session",side_effect=AssertionError("Duplicate dispatch")):
            self.assertEqual(study.evaluate(self.root,2,"astra")["status"],"duplicate")
        study.export_packet(self.root,3)
        request=read_json(self.root/"editor/round-3/request.json")
        self.assertNotIn("strict_passes",request["previous_astra_rounds"][1])
        self.assertNotIn("SECRET_CONTROL_SESSION",json.dumps(request))

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
        started,_=study.budget_start(self.root,self.manifest,state,"browser-one","browser")
        study.budget_finish(self.root,state,started,status="failed",cleanup=False)
        with self.assertRaisesRegex(RuntimeError,"halted"): study.budget_start(self.root,self.manifest,state,"browser-two","browser")

    def test_interrupted_stage_charges_reservation_and_cannot_restart_dispatch(self):
        state=read_json(self.root/"state.json")
        _,reserved=study.budget_start(self.root,self.manifest,state,"r1-astra:browser","browser")
        with patch("weight_space_lab.browser_v2_runner.alive",return_value=False): result=study.recover(self.root)
        self.assertTrue(result["halted"])
        after=read_json(self.root/"state.json")
        self.assertEqual(after["stages"][-1]["charged_seconds"],reserved)
        with self.assertRaises(RuntimeError): study.begin_turn(self.root,1)

    def test_expired_response_still_closes_consumed_turn(self):
        study.export_packet(self.root,1); study.begin_turn(self.root,1)
        path=self.root/"late.json"; path.write_text("{}")
        with patch.object(study.time,"time",return_value=self.manifest["config"]["submission_deadline_epoch_ms"]/1000+1):
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

    def test_runtime_intent_uses_exact_dynamic_schedule_and_identity(self):
        for n in (32,128):
            config={"items":[{"item_id":str(i)} for i in range(n)],"arm":"candidate","configurationSha256":"a"*64,"convertedArtifactSha256":"b"*64,"runId":"synthetic-v2"}
            expected={"kind":"generation","arm":"candidate","request_id":f"generation-{n+1}","attempt_id":"candidate-1",
                "code_version_sha256":"c"*64,"configuration_sha256":"a"*64,"artifact_sha256":"b"*64,"run_id":"synthetic-v2",
                "warmup":False,"max_tokens":1024,"sequence_index":n,"item_id":str(n-1)}
            self.assertEqual(runtime.validate_intent(expected,config,"c"*64,n+1),expected)
            with self.assertRaises(ValueError): runtime.validate_intent(expected,config,"c"*64,n+2)
            expected["max_tokens"]=1025
            with self.assertRaises(ValueError): runtime.validate_intent(expected,config,"c"*64,n+1)


if __name__=="__main__": unittest.main()
