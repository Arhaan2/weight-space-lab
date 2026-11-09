"""Offline contract tests; synthetic content never represents the private final set."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from weight_space_lab.ifeval_data import MAPPINGS
from weight_space_lab.ifeval_v2_dataset import validate_allocation, validate_fixtures, deduplicate, verify_frozen, public_summary
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.inventory import file_sha256


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


def item(types, text='Synthetic evaluator-only prompt.'):
    return {'benchmark':'ifeval','item_id':'fixture','dataset_revision':'a'*64,'split':'test','task_text':text,
            'target':{'instruction_id_list':types,'kwargs':[{} for _ in types]},
            'source':{'dataset_id':'synthetic','revision':'a'*64,'split':'test','row_id':'fixture'},
            'metadata':{'underlying_task':'Synthetic evaluator-only task.'}}


class FinalDatasetTests(unittest.TestCase):
    def test_exact_normalized_and_underlying_exposure_rejected(self):
        sample=item(['punctuation:no_comma'])
        for text in [sample['task_text'],sample['task_text'].upper().replace(' ','  '),sample['metadata']['underlying_task']]:
            inventory={'texts':[{'text':text,'source':'fixture','id':'1'}],'files':{}}
            with self.assertRaisesRegex(ValueError,'overlaps'):
                deduplicate([sample],inventory)

    def test_negative_must_isolate_one_constraint(self):
        sample=item(['punctuation:no_comma','startend:quotation'])
        fixtures=[{'item_id':'fixture','constructive_response':'both pass','negative_fixtures':[
            {'failed_instruction_index':0,'response':'first fails'},
            {'failed_instruction_index':1,'response':'both fail'}]}]
        def evaluate(_,text):
            bits={'both pass':[True,True],'first fails':[False,True],'both fail':[False,False]}[text]
            return {'instruction_strict':bits,'instruction_loose':bits}
        with self.assertRaisesRegex(ValueError,'isolate'):
            validate_fixtures([sample],fixtures,Tokenizer(),evaluate)
        fixtures[0]['negative_fixtures'][1]['response']='second fails'
        def correct(_,text):
            bits={'both pass':[True,True],'first fails':[False,True],'second fails':[True,False]}[text]
            return {'instruction_strict':bits,'instruction_loose':bits}
        result=validate_fixtures([sample],fixtures,Tokenizer(),correct)
        self.assertEqual(result['negative_cases'],2)
        self.assertEqual(result['combined_constraint_cases'],1)

    def test_each_constraint_requires_a_negative(self):
        sample=item(['punctuation:no_comma'])
        fixtures=[{'item_id':'fixture','constructive_response':'valid answer','negative_fixtures':[]}]
        with self.assertRaisesRegex(ValueError,'per instruction'):
            validate_fixtures([sample],fixtures,Tokenizer(),lambda *_:{'instruction_strict':[True],'instruction_loose':[True]})

    def test_constructive_answer_must_fit_sampled_allowance(self):
        sample=item(['punctuation:no_comma'])
        fixtures=[{'item_id':'fixture','constructive_response':'word '*1021,'negative_fixtures':[]}]
        with self.assertRaisesRegex(ValueError,'sampled output'):
            validate_fixtures([sample],fixtures,Tokenizer(),lambda *_:None)

    def test_frozen_component_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'final-items.json').write_text('[]')
            manifest={'files':{'final-items.json':file_sha256(root/'final-items.json')},'items_sha256':canonical_hash([])}
            manifest['sha256']=canonical_hash(manifest)
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_frozen(root)['sha256'],manifest['sha256'])
            (root/'final-items.json').write_text('[{}]')
            with self.assertRaisesRegex(ValueError,'component changed'):
                verify_frozen(root)

    def test_allocation_hash_is_bound(self):
        allocation={'instruction_count':160};allocation['sha256']=canonical_hash(allocation)
        allocation['instruction_count']=159
        with self.assertRaisesRegex(ValueError,'allocation hash'):
            validate_allocation(allocation,[])

    def test_public_summary_uses_allowlist(self):
        sentinel='PRIVATE FINAL SENTINEL'
        manifest={k:'a'*64 for k in ['sha256','items_sha256','checker_configuration_sha256','allocation_sha256','tokenizer_sha256','template_sha256','runtime_sha256']}
        manifest.update(counts={'prompts':128},custody='procedural',secret=sentinel)
        validation={k:1 for k in ['all_passed','constructive_cases','negative_cases','combined_constraint_cases','max_rendered_input_tokens','max_constructive_response_tokens']}
        validation['cases']=[{'prompt':sentinel}]
        summary=public_summary(manifest,validation,{'inventory_rows':1,'exact_or_normalized_matches':0,'private':sentinel},{'template_reuse_summary':'reused templates','secret':sentinel})
        self.assertNotIn(sentinel,json.dumps(summary))


if __name__=='__main__':
    unittest.main()
