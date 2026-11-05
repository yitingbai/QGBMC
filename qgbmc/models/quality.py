"""VQE: paper equations 8–14. Mini-batch topology, natural logarithms."""
import math
import torch

def quality_score(z, alpha=1.0, gamma=3.0):
    n = z.shape[0]
    if n < 3:
        raise ValueError('VQE needs at least 3 samples: log(B-1) must be positive')
    if alpha <= 0 or gamma <= 0:
        raise ValueError('alpha and gamma must be positive')
    dist = torch.cdist(z, z).square()
    logits = -(alpha + 1) / 2 * torch.log1p(alpha * dist)
    mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, -torch.inf)
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    # Replace diagonal -inf before multiplication, avoiding 0 * -inf.
    entropy = -(p * logp.masked_fill(mask, 0)).sum(dim=1).mean()
    normalized = (entropy / math.log(n - 1)).clamp(0, 1)
    return torch.exp(-gamma * normalized), entropy

def quality_scores(embeddings, alpha=1.0, gamma=3.0, mode='absolute'):
    """Return one score per view while preserving the original absolute mode.

    relative mode maps the lowest observed view entropy in a mini-batch to 1
    and the highest to exp(-gamma). It is an explicit experimental candidate,
    not a claim that the submitted equations already specified this mapping.
    """
    observed = [quality_score(z, alpha, gamma) for z in embeddings]
    entropy = torch.stack([item[1] for item in observed])
    if mode == 'absolute':
        quality = torch.stack([item[0] for item in observed])
    elif mode == 'relative':
        span = entropy.max() - entropy.min()
        if span.detach().abs().item() < 1e-12:
            quality = torch.ones_like(entropy)
        else:
            normalized = ((entropy - entropy.min()) / span).clamp(0, 1)
            quality = torch.exp(-gamma * normalized)
    else:
        raise ValueError('quality_mode must be absolute or relative')
    return quality, entropy

def modulation(q, tau=0.6, rho=2.0, kappa=1.0, beta=1.0):
    if not 0 < tau <= 1 or min(rho, kappa, beta) <= 0:
        raise ValueError('invalid QGM parameters')
    q = q.detach()
    acc = beta * (1 - torch.log(q.clamp_min(1e-12) / q.max().clamp_min(1e-12)))
    dec = kappa * (q / tau).pow(rho)
    return torch.where(q >= tau, acc, dec)

def scale_gradient(z, factor):
    """Eq.20: identity forward, factor * I backward; factor is a control."""
    return z.detach() + factor.detach() * (z - z.detach())
