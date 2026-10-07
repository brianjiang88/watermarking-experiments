import unittest
import numpy as np
from PIL import Image
from patch_prc import attacks,watermark as wm
from patch_prc.study import reproduce

class PackageCPUChecks(unittest.TestCase):
    def test_reproduce_saved_table(self):
        reproduce()

    def test_patch_partition_covers_all_coordinates(self):
        for m in wm.GRIDS:
            coverage=np.zeros((4,64,64),dtype=int)
            for sl in wm.slices(m):coverage[sl]+=1
            self.assertTrue(np.all(coverage==1))

    def test_transforms_and_direction(self):
        a=np.zeros((512,512,3),dtype=np.uint8)
        a[250:260,250:260]=255
        image=Image.fromarray(a)
        right=np.asarray(attacks.apply(image,{'kind':'translation','dx':6,'dy':0}))
        self.assertEqual(int(right[255,261,0]),255)
        self.assertEqual(int(right[255,255,0]),0)
        crop=attacks.apply(image,{'kind':'crop_resize','pixels':6})
        self.assertEqual(crop.size,(512,512))
        self.assertTrue(np.array_equal(np.asarray(attacks.apply(image,{'kind':'identity'})),a))

if __name__=='__main__':unittest.main()
