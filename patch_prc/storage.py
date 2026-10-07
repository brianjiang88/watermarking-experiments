import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def immutable(path, obj):
    path = Path(path)
    if path.exists():
        if read(path) != obj: raise ValueError(f'Configuration changed: {path}. Use a new directory.')
    else: dump(path, obj)


def npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    tmp.replace(path)


def png(path, image):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    image.save(tmp, format='PNG'); tmp.replace(path)


def seal(path, files): dump(path, {str(Path(f).name): sha(f) for f in files})


def complete(path):
    path = Path(path)
    if not path.exists(): return False
    for name, expected in read(path).items():
        if sha(path.parent / name) != expected: raise ValueError(f'Checksum mismatch: {path.parent / name}')
    return True


@contextlib.contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Another process owns {path}')
        yield
