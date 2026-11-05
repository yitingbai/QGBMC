import math
import numpy as np
import pytest
import torch
from qgbmc.models.quality import quality_score,quality_scores,modulation,scale_gradient
from qgbmc.models.losses import directional_loss,sca_loss,objective
from qgbmc.models.network import QGBMC
from qgbmc.engine.trainer import batches
from qgbmc.evaluation.metrics import clustering_metrics

def test_uniform_entropy_bound():
    z=torch.zeros(5,3,requires_grad=True);q,h=quality_score(z,gamma=3)
    assert h.item()==pytest.approx(math.log(4))
    assert q.item()==pytest.approx(math.exp(-3))
    q.backward();assert torch.isfinite(z.grad).all()

def test_small_batch_rejected():
    with pytest.raises(ValueError):quality_score(torch.ones(2,3))

def test_relative_quality_spans_expected_bounds():
    embeddings=[torch.zeros(5,3),torch.randn(5,3),torch.arange(15,dtype=torch.float32).reshape(5,3)]
    q,_=quality_scores(embeddings,gamma=2,mode='relative')
    assert q.max().item()==pytest.approx(1.)
    assert q.min().item()==pytest.approx(math.exp(-2),rel=1e-5)

def test_gradient_routing_preserves_reconstruction():
    z=torch.tensor([2.],requires_grad=True)
    loss=z.square().sum()+scale_gradient(z,torch.tensor(.25)).square().sum()
    loss.backward();assert z.grad.item()==pytest.approx(5.)

def test_threshold_and_dominance():
    q=torch.tensor([.9,.6,.3]);m=modulation(q,tau=.6,rho=2)
    assert m[0].item()==1
    assert m[1].item()>1
    assert m[2].item()==pytest.approx(.25)

def test_selective_formula_and_zero():
    a=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);c=torch.tensor(.4)
    got=directional_loss(a,a,c,.5)
    expected=.4*(np.log(np.exp(.8)+1)-.8)
    assert got.item()==pytest.approx(expected,rel=1e-6)
    zero=directional_loss(a,a,torch.tensor(0.),.5);zero.backward()
    assert zero.item()==0 and torch.equal(a.grad,torch.zeros_like(a))

def test_weighted_sca_matches_weighted_infonce():
    a=torch.tensor([[1.,0.],[0.,1.]])
    c=torch.tensor(.4)
    got=directional_loss(a,a,c,.5,mode='weighted')
    expected=.4*(np.log(np.exp(2.)+1)-2.)
    assert got.item()==pytest.approx(expected,rel=1e-6)

def test_gradient_diagnostics_are_finite_and_route_by_branch():
    import json
    cfg=json.load(open('configs/smoke.json'))
    cfg.update({'quality_mode':'relative','gamma':2.,'tau':math.exp(-1),
                'sca_mode':'weighted','log_gradient_norms':True})
    model=QGBMC([5,7],[8],3)
    loss,details=objective(model,[torch.randn(6,5),torch.randn(6,7)],cfg)
    loss.backward()
    assert set(['align_grad_raw_rms','align_grad_modulated_rms','decay_branch']) <= set(details)
    assert all(torch.isfinite(details[key]).all()
               for key in ['align_grad_raw_rms','align_grad_modulated_rms','decay_branch'])

def test_pair_normalization():
    z=[torch.randn(6,3),torch.randn(6,3)];q=torch.ones(2)
    expected=directional_loss(z[0],z[1],q[0],.5)+directional_loss(z[1],z[0],q[0],.5)
    assert torch.allclose(sca_loss(z,q,.5),expected)

def test_label_permutation():
    scores=clustering_metrics([0,0,1,1,2,2],[9,9,7,7,4,4])
    assert all(x==pytest.approx(1) for x in scores.values())

@pytest.mark.parametrize('n',[3,31,32,33,34,35,65,66])
def test_batch_no_drop(n):
    chunks=batches(n,32,torch.Generator().manual_seed(0))
    assert all(len(x)>=3 for x in chunks)
    assert sorted(torch.cat(chunks).tolist())==list(range(n))

@pytest.mark.parametrize('variant',['full','uniform_quality','shuffled_quality',
                                    'wo_vqe','wo_qgm','wo_sca'])
def test_all_branches_finite(variant):
    import json
    cfg=json.load(open('configs/smoke.json'));cfg['variant']=variant
    model=QGBMC([5,7],[8],3)
    loss,_=objective(model,[torch.randn(6,5),torch.randn(6,7)],cfg);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_differentiable_quality_branch():
    import json
    cfg=json.load(open('configs/smoke.json'));cfg['quality_gradient']=True
    model=QGBMC([5,7],[8],3)
    loss,_=objective(model,[torch.randn(6,5),torch.randn(6,7)],cfg);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
