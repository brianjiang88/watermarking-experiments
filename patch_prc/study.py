"""Minimal, resumable PRC patch experiment. One process per GPU."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from . import attacks, storage as io, watermark as wm
ROOT=Path(__file__).resolve().parents[1]

def setup(config, out):
    cfg=io.read(config); cfg['prompts_sha256']=io.sha(ROOT/'data/prompts.json')
    for item in cfg['attacks']:
        if not item['name'].replace('_','').isalnum(): raise ValueError('Attack names must be simple filenames')
        attacks.validate(item)
    assert len({v['name'] for v in cfg['attacks']})==len(cfg['attacks'])
    assert set(cfg['layouts']).issubset(wm.GRIDS)
    prompts=io.read(ROOT/'data/prompts.json')
    assert 1<=cfg['calibration_count']<=len(prompts['calibration'])
    assert 1<=cfg['test_count']<=len(prompts['test'])
    out.mkdir(parents=True,exist_ok=True)
    io.immutable(out/'config.json',cfg)
    (out/'keys').mkdir(exist_ok=True)
    for count in cfg['layouts']:
        path=out/'keys'/f'p{count}.pkl';seal=path.with_suffix('.sha256')
        if path.exists():
            assert seal.exists() and seal.read_text().strip()==io.sha(path)
        else:
            keys=wm.keygen(count,cfg['seed_offset'])
            with path.open('wb') as f:pickle.dump(keys,f)
            path.chmod(0o600);seal.write_text(io.sha(path)+'\n')
    io.dump(out/'setup.json',{'status':'ready','attacks':[a['name'] for a in cfg['attacks']],'layouts':cfg['layouts']})

def keys(out,count):
    p=out/'keys'/f'p{count}.pkl'
    assert io.sha(p)==p.with_suffix('.sha256').read_text().strip()
    with p.open('rb') as f:return pickle.load(f)

def cache_jobs(cfg):
    prompts=io.read(ROOT/'data/prompts.json')
    for split,n in [('calibration',cfg['calibration_count']),('test',cfg['test_count'])]:
        counts=[0] if split=='calibration' else [0]+cfg['layouts']
        for rec in prompts[split][:n]:
            for count in counts:yield split,rec,count

def generated_path(cache,split,i,count):return cache/split/f'{i:04d}'/f'p{count}'

def check_cache(cfg,cache):
    with io.lock(cache/'.cache_setup.lock'):
        io.immutable(cache/'cache_config.json', {
            'model':cfg['model'], 'seed_offset':cfg['seed_offset'],
            'prompts_sha256':io.sha(ROOT/'data/prompts.json'),
            'source':'fixed PRC image generation, 50 steps, guidance 3, FP32'})

def generate(cfg,out,cache,device_index):
    from .pipeline import Pipeline
    import torch
    torch.cuda.set_device(device_index)
    check_cache(cfg,cache)
    pipe=None; done=0
    for job_index,(split,rec,count) in enumerate(cache_jobs(cfg)):
        if job_index%cfg['num_devices']!=device_index:continue
        i=rec['index'];folder=generated_path(cache,split,i,count);folder.mkdir(parents=True,exist_ok=True)
        s=folder/'generated.sha256'
        if s.exists():
            assert s.read_text().strip()==io.sha(folder/'clean.png');continue
        kk=[] if count==0 else keys(out,count)
        z,msg=wm.initial_latent(split,i,count,kk,cfg['seed_offset'])
        if pipe is None:pipe=Pipeline(cfg['model'])
        gseed=wm.seed('generation',i,count,offset=cfg['seed_offset']) if split=='test' else 210000000+10000*i+90+cfg['seed_offset']
        image=pipe.generate(z,rec['prompt'],gseed)
        io.png(folder/'clean.png',image)
        if cfg['save_initial_latents']:io.npz(folder/'initial.npz',z=z,message=msg)
        s.write_text(io.sha(folder/'clean.png')+'\n');done+=1
        print('generated',split,i,count,flush=True)
    print('Generated new images:',done,flush=True)

def recovery_jobs(cfg):
    for split,rec,count in cache_jobs(cfg):
        for spec in cfg['attacks']:
            yield split,rec,count,spec

def recover(cfg,out,cache,device_index):
    from .pipeline import Pipeline
    import torch
    torch.cuda.set_device(device_index)
    check_cache(cfg,cache)
    pipe=None;done=0
    for job_index,(split,rec,count,spec) in enumerate(recovery_jobs(cfg)):
        if job_index%cfg['num_devices']!=device_index:continue
        i=rec['index'];folder=out/'recoveries'/spec['name']/split/f'{i:04d}'/f'p{count}'
        folder.mkdir(parents=True,exist_ok=True);seal=folder/'complete.json'
        if io.complete(seal):continue
        src=generated_path(cache,split,i,count)/'clean.png'
        if not src.exists():raise FileNotFoundError(f'Generate images first: {src}')
        image=attacks.apply(Image.open(src).convert('RGB'),spec)
        if pipe is None:pipe=Pipeline(cfg['model'])
        rseed=wm.seed('recovery',i,count,offset=cfg['seed_offset']) if split=='test' else 210000000+10000*i+91+cfg['seed_offset']
        a,r=pipe.recover(image,rseed)
        files=[folder/'attacked.png',folder/'recovered.npz']
        io.png(files[0],image);io.npz(files[1],r=r,a=a);io.seal(seal,files)
        done+=1;print('recovered',spec['name'],split,i,count,flush=True)
    print('New recoveries:',done,flush=True)

def auc(pos,neg):
    pos=np.asarray(pos);neg=np.asarray(neg)
    return float((pos[:,None]>neg[None,:]).mean()+.5*(pos[:,None]==neg[None,:]).mean())

def evaluate(cfg,out):
    results=out/'results';results.mkdir(exist_ok=True)
    for spec in cfg['attacks']:
        attack=spec['name'];entries=[];cal=[];summary=[]
        for count in cfg['layouts']:
            kk=keys(out,count)
            scores=[]
            for i in range(cfg['calibration_count']):
                f=out/'recoveries'/attack/'calibration'/f'{i:04d}'/'p0'
                assert io.complete(f/'complete.json')
                with np.load(f/'recovered.npz') as z:s=wm.score(z['r'],kk)
                scores.append(s['mean_score']);cal.append({'layout':count,'index':i,'score':s['mean_score']})
            cut=float(max(scores))
            for i in range(cfg['test_count']):
                for label,variant in [('marked',count),('unmarked',0)]:
                    f=out/'recoveries'/attack/'test'/f'{i:04d}'/f'p{variant}'
                    assert io.complete(f/'complete.json')
                    with np.load(f/'recovered.npz') as z:s=wm.score(z['r'],kk)
                    entries.append({'index':i,'layout':count,'label':label,'threshold':cut,'detected':bool(s['mean_score']>cut),**s})
            marked=np.array([r['mean_score'] for r in entries if r['layout']==count and r['label']=='marked'])
            null=np.array([r['mean_score'] for r in entries if r['layout']==count and r['label']=='unmarked'])
            summary.append({'layout':count,'grid':wm.GRIDS[count],'threshold':cut,'n':cfg['test_count'],'detected':int((marked>cut).sum()),'false_positives':int((null>cut).sum()),'auroc':auc(marked,null)})
        target=results/attack;target.mkdir(exist_ok=True)
        io.dump(target/'calibration.json',cal);io.dump(target/'per_image.json',entries);io.dump(target/'summary.json',summary)
        print(attack,json.dumps(summary),flush=True)

def reproduce():
    refs=io.read(ROOT/'data/reference_scores.json');summary=io.read(ROOT/'data/reference_summary.json');cal=io.read(ROOT/'data/reference_calibration.json')
    assert len(refs)==1000 and len(cal)==100
    for item in summary:
        m=item['layout'];pos=np.array([r['mean_score'] for r in refs if r['layout']==m and r['label']=='marked']);neg=np.array([r['mean_score'] for r in refs if r['layout']==m and r['label']=='unmarked']);cut=item['threshold']
        assert len(pos)==len(neg)==100
        assert sum(pos>cut)==item['detected'] and sum(neg>cut)==item['false_positives']
        assert abs(auc(pos,neg)-item['auroc'])<1e-12
        assert abs(max(r['mean_score'] for r in cal if r['layout']==m)-cut)<1e-12
        print(f"{m:>2} patches: {item['detected']:>2}/100 detected, {item['false_positives']}/100 false positives, AUROC {item['auroc']:.4f}")

def verify_prc():
    prc,pg=wm.reference()
    with wm.seeded_prc(20261001):
        ek,dk=prc.KeyGen(256,message_length=8,false_positive_rate=1e-5,t=3)
        code=prc.Encode(ek,np.arange(8,dtype=np.uint8)%2)
        latent=pg.sample(code)
    posterior=pg.recover_posteriors(latent,variances=1.5)
    packed=wm.evidence(dk,latent.numpy())
    original=bool(prc.Detect(dk,posterior))
    assert original==(packed>=np.sqrt(np.log(1e5)))
    print('Packaged score agrees with reference PRC detector:',original)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['reproduce','verify-prc','setup','generate','recover','evaluate','all'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/reference.json');p.add_argument('--output',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='reproduce':reproduce();return
    if a.command=='verify-prc':verify_prc();return
    if a.output is None: p.error('--output is required')
    cfg=io.read(a.config);cache=a.cache or a.output/'image_cache'
    with io.lock(a.output/f'.lock_{a.device}' if a.command in ['generate','recover'] else a.output/'.lock_main'):
        if a.command=='setup':setup(a.config,a.output);return
        expected=dict(cfg,prompts_sha256=io.sha(ROOT/'data/prompts.json'))
        if io.read(a.output/'config.json')!=expected:raise ValueError('Run setup first, or use a new output directory for changed settings')
        if a.command in ['generate','all']:generate(cfg,a.output,cache,a.device)
        if a.command in ['recover','all']:recover(cfg,a.output,cache,a.device)
        if a.command in ['evaluate','all']:evaluate(cfg,a.output)
if __name__=='__main__':main()
