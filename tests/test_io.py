import numpy as np
import pytest
from qgbmc.data.io import load_npz,save_npz,preprocess

def test_roundtrip_and_noise(tmp_path):
    views=[np.arange(60,dtype=np.float32).reshape(12,5),np.ones((12,3),dtype=np.float32)]
    path=tmp_path/'d.npz';save_npz(path,views,np.arange(12))
    got,labels=load_npz(path)
    assert np.array_equal(labels,np.arange(12))
    a=preprocess(got,1,[1],7);b=preprocess(got,1,[1],7)
    assert np.array_equal(a[0][1],b[0][1])
    assert not a[3][0].any() and a[3][1].all()
    assert np.isfinite(a[0][1]).all()

def test_bad_alignment(tmp_path):
    path=tmp_path/'bad.npz';save_npz(path,[np.ones((4,2)),np.ones((5,2))])
    with pytest.raises(ValueError):load_npz(path)

def test_sample_gaussian_noise_is_row_level():
    views=[np.arange(60,dtype=np.float32).reshape(12,5),np.ones((12,3),dtype=np.float32)]
    changed,_,_,masks=preprocess(views,noise_rate=.5,noise_views=(0,),seed=7,
                                 noise_mode='sample_gaussian')
    assert masks[0].shape == (12,)
    assert 0 < masks[0].sum() < 12
    assert masks[1].shape == (12,) and not masks[1].any()
    assert np.isfinite(changed[0]).all()

def test_unknown_noise_mode_rejected():
    with pytest.raises(ValueError):
        preprocess([np.ones((4,2)),np.ones((4,3))],noise_mode='unknown')
