"""The study's fixed FP32 SD2.1 pipeline; no sampler replacement."""
from pathlib import Path
import random
import sys
import numpy as np
import torch
from .optimized_recovery import optimized_recovery
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/gunn'))
from inversion import stable_diffusion_pipe,generate
from src.optim_utils import transform_img


class Pipeline:
    def __init__(self, model):
        if not torch.cuda.is_available(): raise RuntimeError('Generation/recovery requires a CUDA GPU. Use reproduce for CPU-only analysis.')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        self.pipe=stable_diffusion_pipe(solver_order=1,model_id=str(model),cache_dir=str(Path(model).parent/'hf'))
        self.pipe.set_progress_bar_config(disable=True)
        with torch.no_grad():
            e=self.pipe.encode_prompt('','cuda',1,True,None)
            self.emb=torch.cat([e[1],e[0]])

    def generate(self,z,prompt,seed):
        image,actual_prompt,actual=generate(prompt=prompt,init_latents=torch.from_numpy(z[None]).cuda(),gen_seed=seed,guidance_scale=3.,num_inference_steps=50,solver_order=1,pipe=self.pipe)
        assert actual_prompt==prompt and np.array_equal(actual.cpu().numpy()[0],z)
        return image

    def recover(self,image,seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        x=transform_img(image).unsqueeze(0).cuda().float()
        with optimized_recovery(self.pipe):a=self.pipe.decoder_inv(x).detach().cpu().numpy()[0]
        with torch.no_grad():
            r=self.pipe.forward_diffusion(latents=torch.tensor(a.astype(np.float32)[None],device='cuda'),text_embeddings=self.emb,guidance_scale=3.,num_inference_steps=50,inverse_opt=False,inv_order=0).cpu().numpy()[0]
        assert a.shape==r.shape==(4,64,64) and np.isfinite(a).all() and np.isfinite(r).all()
        return a,r
