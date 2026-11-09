#!/usr/bin/env python3
"""Seal an evaluator-private V2 draft offline, without any model execution."""
import argparse
import json
from pathlib import Path
from weight_space_lab.ifeval_v2_dataset import seal, verify_frozen
from weight_space_lab.ifeval_data import write_new

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--tokenizer')
    p.add_argument('--ifeval-root')
    p.add_argument('--environment')
    p.add_argument('--bundled-audit')
    p.add_argument('--public-summary')
    a=p.parse_args()
    if a.verify_only:
        m=verify_frozen(a.root)
        print(json.dumps({'verified':True,'manifest_sha256':m['sha256'],'counts':m['counts']}))
    else:
        if not all([a.tokenizer,a.ifeval_root,a.environment,a.bundled_audit,a.public_summary]):
            p.error('sealing requires tokenizer, ifeval-root, environment, bundled-audit and public-summary')
        result=seal(a.root,tokenizer_path=a.tokenizer,ifeval_root=a.ifeval_root,environment_path=a.environment,bundled_audit_path=a.bundled_audit)
        write_new(Path(a.public_summary),result)
        print(json.dumps(result,indent=2))
