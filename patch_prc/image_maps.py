"""Fixed nonnegative interpolation matrices; image quantization is separate."""
import numpy as np
from scipy import sparse


def axis_matrix(n, crop=0.0, shift=0.0):
    if not 0 <= crop < n/2:
        raise ValueError('crop must be in [0,n/2)')
    coords=crop+(np.arange(n)+0.5)*(n-2*crop)/n-0.5+shift
    coords=np.clip(coords,0,n-1)
    lo=np.floor(coords).astype(int); hi=np.minimum(lo+1,n-1); w=coords-lo
    return sparse.csr_matrix((np.r_[1-w,w],(np.r_[np.arange(n),np.arange(n)],np.r_[lo,hi])),shape=(n,n))


def apply_matrix(chw, theta, reference_size=512):
    """theta=(crop, horizontal shift, vertical shift), in 512px image units."""
    a=np.asarray(chw,dtype=np.float64)
    assert a.ndim==3 and a.shape[1]==a.shape[2]
    n=a.shape[1]; c,dx,dy=np.array(theta,dtype=float)*n/reference_size
    sy=axis_matrix(n,c,dy);sx=axis_matrix(n,c,dx)
    return np.stack([(sx@(sy@channel).T).T for channel in a])


def render(image, theta):
    from PIL import Image
    a=np.asarray(image.convert('RGB'),dtype=float).transpose(2,0,1)/255
    b=apply_matrix(a,theta)
    assert b.min()>=-1e-12 and b.max()<=1+1e-12
    return Image.fromarray(np.rint(np.clip(b,0,1)*255).astype(np.uint8).transpose(1,2,0))


def max_axis_displacement(theta,n=512):
    c,dx,dy=theta
    # Upper bound before boundary extension; exact over the unclamped grid.
    return c*(n-1)/n+max(abs(dx),abs(dy))


def soft_consistency(clean,recovered,tau=0.5):
    a=np.tanh(np.asarray(clean,dtype=float)/tau)
    b=np.tanh(np.asarray(recovered,dtype=float)/tau)
    return float(np.mean(a*b)/max(np.mean(a*a),1e-12))
