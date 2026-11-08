"""Future candidate packaging audits, using tiny MLC fixtures and no inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from weight_space_lab.browser_candidate import audit_effective_pair, package_candidate, source_identity, _files
from weight_space_lab.protocol import canonical_hash


class BrowserCandidateTests(unittest.TestCase):
    source15='model.language_model.layers.15.mlp.down_proj.weight'
    source19='model.language_model.layers.19.mlp.down_proj.weight'

    def cache(self,root,*,change_layer=None,change_group=0):
        root=Path(root);root.mkdir()
        raw=b'';records=[]
        for layer in (15,19):
            weight=np.full((1,8),0x88888888,dtype='<u4')
            scale=np.ones((1,2),dtype='<f2')
            if change_layer==layer:scale[0,change_group]=2
            for suffix,array in [('q_weight',weight),('q_scale',scale)]:
                payload=array.tobytes()
                records.append({'name':f'model.layers.{layer}.mlp.down_proj.{suffix}',
                    'shape':list(array.shape),'dtype':array.dtype.name,'format':'raw','byteOffset':len(raw),'nbytes':len(payload)})
                raw+=payload
        (root/'params.bin').write_bytes(raw)
        (root/'tensor-cache.json').write_text(json.dumps({'metadata':{'ParamSize':4},'records':[
            {'dataPath':'params.bin','nbytes':len(raw),'md5sum':hashlib.md5(raw).hexdigest(),'records':records}]}))
        return root

    def test_scale_changes_disclose_quantization_group_spillover(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline=self.cache(Path(temp)/'b')
            candidate=self.cache(Path(temp)/'c',change_layer=15)
            result=audit_effective_pair(baseline,candidate,{'operations':[{'tensor':self.source15,'columns':[0]}]},[self.source15,self.source19])
            row=result['per_source_tensor'][self.source15]
            self.assertEqual(row['decoded_values_changed'],32)
            self.assertEqual(row['decoded_changed_outside_source_target_columns'],31)
            self.assertEqual(row['decoded_changed_outside_source_target_groups'],0)
            self.assertFalse(result['conversion_erases_edit'])

    def test_untargeted_source_tensor_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline=self.cache(Path(temp)/'b')
            candidate=self.cache(Path(temp)/'c',change_layer=19)
            with self.assertRaisesRegex(ValueError,'not targeted'):
                audit_effective_pair(baseline,candidate,{'operations':[{'tensor':self.source15,'columns':[0]}]},[self.source15,self.source19])

    def test_unrelated_quantization_group_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline=self.cache(Path(temp)/'b')
            candidate=self.cache(Path(temp)/'c',change_layer=15,change_group=1)
            with self.assertRaisesRegex(ValueError,'outside source-targeted'):
                audit_effective_pair(baseline,candidate,{'operations':[{'tensor':self.source15,'columns':[0]}]},[self.source15,self.source19])

    def test_identity_or_erased_edit_is_reported_without_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline=self.cache(Path(temp)/'b')
            candidate=self.cache(Path(temp)/'c')
            result=audit_effective_pair(baseline,candidate,{'operations':[{'tensor':self.source19,'columns':[1]}]},[self.source15,self.source19])
            self.assertTrue(result['conversion_erases_edit'])
            self.assertEqual(result['unchanged_tensor_count'],4)
            self.assertEqual(result['changed_tensor_names'],[])


    def test_package_binds_source_and_copies_exact_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            baseline=self.cache(temp/'baseline')
            converted=self.cache(temp/'converted',change_layer=15)
            config={'tokenizer_files':['tokenizer.json'],'context_window_size':4096}
            for name,data in [('mlc-chat-config.json',json.dumps(config)),('tokenizer.json','{}'),('model.wasm','wasm-fixture'),('LICENSE','license-fixture')]:
                (baseline/name).write_text(data)
            files=_files(baseline)
            manifest={'quantization':'q4f16_1','webllm_version':'0.2.86','arms':{'baseline':{
                'path':str(baseline),'files':files,'artifact_sha256':canonical_hash(files),'source_checkpoint_sha256':'original'}}}
            manifest_path=temp/'baseline-manifest.json';manifest_path.write_text(json.dumps(manifest))
            identity={'source_checkpoint_sha256':'candidate','source_original_sha256':'original','source_audit_sha256':'audit','source_proposal_sha256':'proposal','source_policy_sha256':'policy'}
            from weight_space_lab.inventory import file_sha256
            binding={**identity,'status':'converted_source_bound','baseline_manifest_sha256':file_sha256(manifest_path),
                'converted_files':_files(converted),'converted_files_sha256':canonical_hash(_files(converted))}
            binding_path=temp/'converted-source-binding.json';binding_path.write_text(json.dumps(binding))
            proposal={'operations':[{'tensor':self.source15,'columns':[0]}]}
            with patch('weight_space_lab.browser_candidate.source_identity',return_value=(identity,{'complete':True},proposal)):
                result=package_candidate('unused','unused',manifest_path,converted,temp/'packaged')
            artifact=json.loads(Path(result['artifact_manifest']).read_text())
            for name in config['tokenizer_files']+['mlc-chat-config.json','model.wasm','LICENSE']:
                self.assertEqual(Path(artifact['path'],name).read_bytes(),(baseline/name).read_bytes())
            self.assertEqual(artifact['source_checkpoint_sha256'],'candidate')
            self.assertEqual(artifact['artifact_sha256'],canonical_hash(artifact['files']))
            self.assertNotEqual(artifact['artifact_sha256'],manifest['arms']['baseline']['artifact_sha256'])
            self.assertEqual(_files(baseline),files)
            binding['source_checkpoint_sha256']='unrelated'
            binding_path.write_text(json.dumps(binding))
            with patch('weight_space_lab.browser_candidate.source_identity',return_value=(identity,{},proposal)):
                with self.assertRaisesRegex(ValueError,'not bound'):
                    package_candidate('unused','unused',manifest_path,converted,temp/'must-not-publish')
            self.assertFalse((temp/'must-not-publish').exists())


    def test_source_verification_uses_the_frozen_narrowed_pilot_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            policy={'schema_version':1,'inspection':{'max_requests':2,'max_total_values':12}}
            policy_path=temp/'pilot-policy.json';policy_path.write_text(json.dumps(policy))
            proposal={'provenance':'editor_proposal','operations':[]}
            (temp/'proposal.json').write_text(json.dumps(proposal))
            audit={'base_sha256':'original','result_sha256':'candidate'}
            manifest={'arms':{'baseline':{'source_checkpoint_sha256':'original'}}}
            with patch('weight_space_lab.browser_candidate.verify_candidate',return_value=audit) as verifier:
                identity,_,_=source_identity('original-source',temp,manifest,policy_path=policy_path)
            verifier.assert_called_once_with('original-source',temp,policy)
            self.assertEqual(identity['source_policy_sha256'],canonical_hash(policy))

if __name__=='__main__':unittest.main()
