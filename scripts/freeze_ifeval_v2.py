#!/usr/bin/env python3
"""Audit and seal a private official IFEval subset without model execution.

The earlier authored-set sealer is retained behind --historical-authored for
reproducing the superseded preparation, never as an automatic fallback.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_space_lab.ifeval_v2_dataset import seal, seal_official, verify_frozen
from weight_space_lab.ifeval_data import write_new

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--tokenizer')
    p.add_argument('--ifeval-root')
    p.add_argument('--environment')
    p.add_argument('--bundled-audit')
    p.add_argument('--historical-authored', action='store_true')
    p.add_argument('--source-reference', default='configs/benchmark_sources.json')
    p.add_argument('--source-cache', default='artifacts/phase04-benchmark-cache')
    p.add_argument('--exposure-inventory')
    p.add_argument('--grading-policy', default='artifacts/ifeval-browser-v1/grading-policy.json')
    p.add_argument('--public-summary')
    a=p.parse_args()
    if a.verify_only:
        m=verify_frozen(a.root)
        print(json.dumps({'verified':True,'manifest_sha256':m['sha256'],'counts':m['counts']}))
    else:
        if not all([a.tokenizer,a.ifeval_root,a.environment,a.public_summary]):
            p.error('sealing requires tokenizer, ifeval-root, environment and public-summary')
        if a.historical_authored:
            if not a.bundled_audit:
                p.error('historical authored sealing requires bundled-audit')
            result=seal(a.root,tokenizer_path=a.tokenizer,ifeval_root=a.ifeval_root,environment_path=a.environment,bundled_audit_path=a.bundled_audit)
        else:
            if not a.exposure_inventory:
                p.error('official sealing requires an audited exposure-inventory')
            result=seal_official(a.root, source_reference=a.source_reference, source_cache=a.source_cache,
                exposure_path=a.exposure_inventory, tokenizer_path=a.tokenizer, ifeval_root=a.ifeval_root,
                environment_path=a.environment, grading_policy_path=a.grading_policy)
        write_new(Path(a.public_summary),result)
        print(json.dumps(result,indent=2))
