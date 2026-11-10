"""Offline contract tests; synthetic content never represents the private final set."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from weight_space_lab.ifeval_data import MAPPINGS
from weight_space_lab.ifeval_v2_dataset import validate_allocation, validate_fixtures, deduplicate, verify_frozen, public_summary, select_official
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
    def test_official_unresolved_exposure_is_not_silently_clean(self):
        samples = [item(['punctuation:no_comma'])]
        selected, audit = select_official(samples, {'items': {}}, Tokenizer())
        self.assertEqual(selected, [])
        self.assertEqual(audit['source_prompts'], 1)
        self.assertEqual(audit['eligible_unexposed_prompts'], 0)
        self.assertEqual(audit['cases'][0]['exclusion_reasons'], ['exposure_unresolved'])

    def test_official_smaller_set_preserves_counts_and_type_sharing(self):
        samples = [item(['punctuation:no_comma'], text) for text in [
            'Explain how an electric motor works without commas.',
            'Name three ingredients for a winter vegetable soup without commas.']]
        for index, sample in enumerate(samples): sample['item_id'] = str(index)
        exposure = {'items': {sample['item_id']: {'status': 'unexposed', 'evidence': ['synthetic custody record'],
            'assessment_complete': True} for sample in samples}}
        selected, audit = select_official(samples, exposure, Tokenizer())
        self.assertEqual(len(selected), 2)
        self.assertEqual(audit['selected_prompts'], 2)
        self.assertEqual({i['item_id'] for i in selected}, {'0', '1'})
        self.assertEqual(select_official(list(reversed(samples)), exposure, Tokenizer())[0], selected)

    def test_official_duplicate_instances_and_exposed_matches_are_excluded(self):
        samples = [item(['punctuation:no_comma'], text) for text in [
            'Write a detailed winter vegetable soup recipe without any commas.',
            'Write a detailed winter vegetable soup recipe without any commas!',
            'Explain electric motor operation.', 'Describe oak leaves.']]
        for index, sample in enumerate(samples): sample['item_id'] = str(index)
        exposure = {'items': {sample['item_id']: {'status': 'unexposed', 'evidence': ['fixture'],
            'assessment_complete': True} for sample in samples},
            'known_exposed_prompts': [{'id': 'old', 'text': samples[2]['task_text']}]}
        selected, audit = select_official(samples, exposure, Tokenizer())
        self.assertEqual([i['item_id'] for i in selected], ['3'])
        self.assertIn('source_exact_or_near_duplicate', audit['cases'][0]['exclusion_reasons'])
        self.assertIn('exposed_exact_or_near_duplicate', audit['cases'][2]['exclusion_reasons'])

    def test_official_unexposed_requires_completed_evidence(self):
        with self.assertRaisesRegex(ValueError, 'completed evidence'):
            select_official([item(['punctuation:no_comma'])],
                {'items': {'fixture': {'status': 'unexposed'}}}, Tokenizer())

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
