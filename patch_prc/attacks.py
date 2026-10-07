"""Image transformations. Return a same-size RGB PIL image; never use keys.

To add a transformation, add a branch in apply(). Each config entry needs a
unique name. Changing this module requires a new --output directory; the old
--cache of generated images can still be reused.
"""
import io
import numpy as np
from PIL import Image, ImageFilter
from .image_maps import render


def apply(image, spec):
    image = image.convert('RGB')
    kind = spec['kind']
    if kind == 'identity':
        return image.copy()
    if kind == 'crop_resize':
        c = float(spec['pixels'])
        if not 0 <= c < 256:
            raise ValueError('pixels must be in [0,256)')
        # Exact historical half-pixel bilinear matrix; no antialias filter.
        return render(image, (c, 0, 0))
    if kind == 'translation':
        # Positive dx/dy moves visible content right/down; replicate edges.
        return render(image, (0, -float(spec.get('dx', 0)), -float(spec.get('dy', 0))))
    if kind == 'jpeg':
        quality = int(spec['quality'])
        if not 1 <= quality <= 100:
            raise ValueError('JPEG quality must be 1..100')
        b = io.BytesIO(); image.save(b, format='JPEG', quality=quality)
        b.seek(0)
        return Image.open(b).convert('RGB')
    if kind == 'gaussian_blur':
        radius = float(spec['radius'])
        if radius < 0: raise ValueError('radius must be nonnegative')
        return image.filter(ImageFilter.GaussianBlur(radius))
    raise ValueError(f'Unknown transformation: {kind}')


def validate(spec):
    apply(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)), spec)
