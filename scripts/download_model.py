#!/usr/bin/env python3
"""Download the exact SD snapshot used in the note; verify all saved hashes."""
import argparse,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads((ROOT/'provenance/sd_model.json').read_text())
snapshot_download(repo_id=m['downloaded_model_id'],revision=m['revision'],allow_patterns=[f['path'] for f in m['files']],local_dir=str(a.output),local_dir_use_symlinks=False)
for row in m['files']:
    h=hashlib.sha256()
    with (a.output/row['path']).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==row['sha256'],row['path']
print('Verified pinned SD model:',a.output)
