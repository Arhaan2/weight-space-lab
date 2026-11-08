"""Local CPU weight conversion using the pinned MLC loader and quantizer.

The 0.26.dev6 converter unnecessarily exports the full graph to enumerate
parameters, exposing an unrelated upstream RNN-state TIR validation failure.
This preparation-only adapter enumerates the same model Parameters directly.
It never builds or runs an inference engine and never changes the model library.
Every resulting parameter must match the pinned WebGPU library's metadata.
"""
from __future__ import annotations
import argparse
import json
import time
from types import SimpleNamespace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wasm-metadata', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError('Conversion output must be new')
    import tvm
    from mlc_llm.model.qwen35.qwen35_model import Qwen35LMHeadModel
    from mlc_llm.model import MODELS
    from mlc_llm.quantization import QUANTIZATION
    from mlc_llm.interface.convert_weight import convert_weight

    original_export = Qwen35LMHeadModel.export_tvm
    Qwen35LMHeadModel.export_tvm = lambda self, **kwargs: (None, [
        (name, SimpleNamespace(shape=param.shape, dtype=str(param.dtype)))
        for name, param in self.named_parameters()], [])
    started = time.time()
    try:
        convert_weight(config=source/'config.json', quantization=QUANTIZATION['q4f16_1'],
                       model=MODELS['qwen3_5'], device=tvm.cpu(), source=source/'model.safetensors.index.json',
                       source_format='huggingface-safetensor', output=output)
    finally:
        Qwen35LMHeadModel.export_tvm = original_export
    cache = json.loads((output/'tensor-cache.json').read_text())
    expected = json.loads(Path(args.wasm_metadata).read_text())['params']
    observed = {p['name']: {'shape':p['shape'], 'dtype':p['dtype']} for s in cache['records'] for p in s['records']}
    contract = {p['name']: {'shape':p['shape'], 'dtype':p['dtype']} for p in expected}
    if observed != contract:
        raise ValueError('Converted parameters differ from pinned WebGPU library contract')
    (output.parent/(output.name+'-receipt.json')).write_text(json.dumps({
        'schema_version':1, 'source':str(source), 'output':str(output),
        'preparation_seconds':time.time()-started, 'target_generations':0,
        'device':'cpu', 'quantization':'q4f16_1', 'parameter_contract_verified':True,
        'parameter_count':len(observed),
        'adapter':'direct named_parameters enumeration instead of unrelated full-graph export',
    },indent=2)+'\n')

if __name__=='__main__':
    main()
