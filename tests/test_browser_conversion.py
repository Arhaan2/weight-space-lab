"""Offline checks for deployed tensor auditing; never loads an inference model."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from weight_space_lab.browser_conversion import decode_q4, load_tensor, tensor_index


class BrowserConversionTests(unittest.TestCase):
    def test_q4_decodes_nibble_order_signed_offset_and_group_scale(self):
        weight = np.array([[0x76543210, 0xFEDCBA98, 0x77777777, 0x77777777,
                            0x88888888, 0x77777777, 0x77777777, 0x77777777]], dtype=np.uint32)
        scale = np.array([[0.5, 2]], dtype=np.float16)
        decoded = decode_q4(weight, scale)
        np.testing.assert_array_equal(decoded[0,:16],np.arange(-7,9,dtype=np.float16)*0.5)
        np.testing.assert_array_equal(decoded[0,32:40],np.full(8,2,dtype=np.float16))

    def test_bf16_storage_expands_to_float32(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'data.bin').write_bytes(np.array([0x3F80,0xC000],dtype='<u2').tobytes())
            actual=load_tensor(temp,{'dataPath':'data.bin','byteOffset':0,'nbytes':4,'dtype':'float32','format':'f32-to-bf16','shape':[2]})
            np.testing.assert_array_equal(actual,np.array([1,-2],dtype=np.float32))

    def _cache(self, root):
        raw=np.array([1,2],dtype='<f2').tobytes()
        Path(root,'data.bin').write_bytes(raw)
        cache={'metadata':{'ParamSize':1},'records':[{'dataPath':'data.bin','nbytes':4,'md5sum':hashlib.md5(raw).hexdigest(),'records':[{'name':'weight','shape':[2],'dtype':'float16','format':'raw','byteOffset':0,'nbytes':4}]}]}
        Path(root,'tensor-cache.json').write_text(json.dumps(cache))
        return cache

    def test_changed_shard_cannot_pass_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            self._cache(temp)
            self.assertEqual(set(tensor_index(temp)),{'weight'})
            Path(temp,'data.bin').write_bytes(b'xxxx')
            with self.assertRaisesRegex(ValueError,'digest'):tensor_index(temp)

    def test_overlapping_or_duplicate_records_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cache=self._cache(temp)
            cache['metadata']['ParamSize']=2
            duplicate=dict(cache['records'][0]['records'][0]);duplicate['name']='other'
            cache['records'][0]['records'].append(duplicate)
            Path(temp,'tensor-cache.json').write_text(json.dumps(cache))
            with self.assertRaisesRegex(ValueError,'overlaps'):tensor_index(temp)

if __name__=='__main__':unittest.main()
