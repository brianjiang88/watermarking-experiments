"""Reference PRC embedding, deterministic benchmark streams, patch scores."""
import contextlib
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {1:(1,1), 4:(2,2), 16:(4,4), 32:(4,8), 64:(8,8)}
VARIANTS = [0,1,4,16,32,64]


def reference():
    sys.path.insert(0, str(ROOT/'third_party/gunn'))
    from src import prc, pseudogaussians
    return prc, pseudogaussians


def slices(count):
    rows, cols = GRIDS[count]
    return [(slice(None), slice(y*64//rows, (y+1)*64//rows), slice(x*64//cols, (x+1)*64//cols)) for y in range(rows) for x in range(cols)]


def seed(stream, i, count, patch_index=0, offset=0):
    base = dict(key=3000000000, encode=3100000000, message=3200000000,
                unmarked=3300000000, generation=3400000000, recovery=3500000000)
    return base[stream] + VARIANTS.index(count)*1000000 + i*1000 + patch_index + offset


@contextlib.contextmanager
def seeded_prc(value):
    prc, _ = reference()
    state = np.random.get_state(); np.random.seed(value)
    original = prc.GF.Random; rng = np.random.default_rng(value)
    with patch.object(prc.GF, 'Random', side_effect=lambda shape: original(shape, seed=rng)):
        try: yield
        finally: np.random.set_state(state)


def keygen(count, offset=0):
    prc, _ = reference(); keys=[]
    for j in range(count):
        with seeded_prc(seed('key',0,count,j,offset)):
            keys.append(prc.KeyGen(16384//count, message_length=512//count, false_positive_rate=1e-5, t=3))
    return keys


def initial_latent(split, i, count, keys, offset=0):
    if split == 'calibration':
        return np.random.default_rng(210000000+10000*i+offset).normal(size=(4,64,64)), np.empty((0,0),dtype=np.uint8)
    if count == 0:
        return np.random.default_rng(seed('unmarked',i,0,offset=offset)).normal(size=(4,64,64)), np.empty((0,0),dtype=np.uint8)
    prc, pg = reference(); z=np.empty((4,64,64),dtype=np.float64); messages=[]
    for j,(sl,(ek,_)) in enumerate(zip(slices(count),keys)):
        msg=np.random.default_rng(seed('message',i,count,j,offset)).integers(0,2,512//count,dtype=np.uint8)
        with seeded_prc(seed('encode',i,count,j,offset)):
            part=pg.sample(prc.Encode(ek,msg)).numpy()
        z[sl]=part.reshape(z[sl].shape);messages.append(msg)
    return z,np.stack(messages)


def evidence(dk, recovered):
    _,h,pad,_,noise,_,_,_,t=dk
    posterior=erf(np.asarray(recovered,dtype=np.float64)/np.sqrt(2*1.5*2.5))
    p=(1-2*noise)*(1-2*np.asarray(pad,dtype=float))*posterior
    pi=np.prod(p[h.indices.reshape(h.shape[0],t)],axis=1)
    lp=np.log((1+pi)/2);lm=np.log((1-pi)/2);both=lp+lm
    const=.5*np.sum(lp**2+lm**2-.5*both**2)
    return 0. if const<=0 else float((lp.sum()-.5*both.sum())/np.sqrt(2*const))


def score(recovered, keys):
    vals=[evidence(dk,np.ascontiguousarray(recovered[sl]).reshape(-1)) for sl,(_,dk) in zip(slices(len(keys)),keys)]
    return dict(mean_score=float(np.mean(vals)),patch_scores=vals,original_any=bool(max(vals)>=np.sqrt(np.log(1e5))))
