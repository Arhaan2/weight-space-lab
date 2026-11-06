import copy
import unittest

import numpy as np

from weight_space_lab.patches import apply_patch, state_sha256


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.state = {'w': np.array([[0, 2], [3, 4]], dtype=np.float32),
                      'bias': np.array([1, 2], dtype=np.float32)}
        self.op = {'op': 'scale_columns', 'tensor': 'w', 'columns': [0], 'factor': 2}
        self.patch = {'schema_version': 1, 'base_sha256': state_sha256(self.state),
                      'operations': [self.op]}

    def run_patch(self, patch=None, budget=1, allowed=('w',)):
        return apply_patch(self.state, self.patch if patch is None else patch,
                           allowed_tensors=allowed, max_changed_fraction=budget)

    def test_hash_is_layout_and_mapping_order_independent(self):
        reordered = {'bias': self.state['bias'].copy(),
                     'w': np.asfortranarray(self.state['w'])}
        self.assertEqual(state_sha256(self.state), state_sha256(reordered))
        reordered['w'] = reordered['w'].astype(np.float64)
        self.assertNotEqual(state_sha256(self.state), state_sha256(reordered))

    def test_independent_copy_actual_count_and_whole_state_budget(self):
        before = copy.deepcopy(self.state)
        edited, audit = self.run_patch(budget=1 / 6)
        self.assertEqual(audit['changed_scalars'], 1)
        self.assertEqual(audit['changed_fraction'], 1 / 6)
        self.assertEqual(audit['l2_norm'], 3)
        self.assertEqual(audit['result_sha256'], state_sha256(edited))
        for name in self.state:
            np.testing.assert_array_equal(self.state[name], before[name])
            self.assertFalse(np.shares_memory(edited[name], self.state[name]))
            self.assertEqual(edited[name].dtype, self.state[name].dtype)
            self.assertEqual(edited[name].shape, self.state[name].shape)
        with self.assertRaises(ValueError):
            self.run_patch(budget=0)
        np.testing.assert_array_equal(self.state['w'], before['w'])

    def test_projection_and_restored_values(self):
        self.patch['operations'] = [{'op': 'remove_direction_from_columns',
            'tensor': 'w', 'columns': [1], 'direction': [2, 0], 'strength': 1}]
        edited, _ = self.run_patch()
        np.testing.assert_array_equal(edited['w'], [[0, 0], [3, 4]])
        self.patch['operations'] = [self.op, dict(self.op, factor=0.5)]
        _, audit = self.run_patch(budget=0)
        self.assertEqual(audit['changed_scalars'], 0)
        self.assertEqual(audit['base_sha256'], audit['result_sha256'])

    def test_invalid_patch_is_rejected_without_mutation(self):
        original_hash = state_sha256(self.state)
        invalid = [dict(self.patch, base_sha256='0' * 64), dict(self.patch, extra=True)]
        for change in ({'columns': [0, 0]}, {'columns': [True]}, {'columns': [-1]},
                       {'columns': [2]}, {'factor': True}, {'factor': float('nan')},
                       {'factor': float('inf')}, {'tensor': 'absent'}, {'op': 'unknown'},
                       {'extra': 1}):
            invalid.append(dict(self.patch, operations=[dict(self.op, **change)]))
        for patch in invalid:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.run_patch(patch)
        with self.assertRaises(ValueError):
            self.run_patch(allowed=())
        self.assertEqual(state_sha256(self.state), original_hash)

    def test_zero_direction_and_cast_overflow_are_rejected(self):
        self.patch['operations'] = [{'op': 'remove_direction_from_columns',
            'tensor': 'w', 'columns': [0], 'direction': [0, 0], 'strength': 1}]
        with self.assertRaises(ValueError):
            self.run_patch()
        self.patch['operations'] = [dict(self.op, factor=1e40)]
        with self.assertRaises(ValueError):
            self.run_patch()
