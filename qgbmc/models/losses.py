import torch
import torch.nn.functional as F
from .quality import quality_score, quality_scores, modulation, scale_gradient

def directional_loss(source, target, reliability, temperature, mode='gated_positive'):
    """Directional InfoNCE with an explicit manuscript or corrected mode.

    ``gated_positive`` reproduces manuscript Eqs. 29–31. ``weighted`` follows
    the mutual-information expression in Eq. 28 by weighting an ordinary
    InfoNCE loss without shrinking only its positive logit.  Keeping both
    modes makes the algorithmic change auditable instead of silently changing
    the submitted method.
    """
    logits = F.normalize(source, dim=1) @ F.normalize(target, dim=1).T / temperature
    if mode == 'gated_positive':
        diagonal = torch.eye(len(source), dtype=torch.bool, device=source.device)
        logits = torch.where(diagonal, logits * reliability, logits)
    elif mode != 'weighted':
        raise ValueError('sca_mode must be gated_positive or weighted')
    return reliability * (torch.logsumexp(logits, dim=1) - logits.diag()).mean()

def sca_loss(z, q, temperature=0.5, mode='gated_positive'):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    m = len(z)
    loss = z[0].new_zeros(())
    # Eq.36 sums BOTH directions inside an ORDERED pair sum.
    # Each unordered pair is counted twice; retain that normalization.
    for s in range(m):
        for v in range(s + 1, m):
            c = (q[s] * q[v]).sqrt()
            loss = loss + 2 * (directional_loss(z[s], z[v], c, temperature, mode)
                               + directional_loss(z[v], z[s], c, temperature, mode))
    return loss / (m * (m - 1))

def objective(model, views, cfg):
    z, fused, recon = model(views)
    q, entropy = quality_scores(
        [v.detach() for v in z], cfg['alpha'], cfg['gamma'],
        cfg.get('quality_mode', 'absolute'))
    variant = cfg['variant']
    if variant in {'wo_vqe', 'uniform_quality'}:
        controls = torch.ones_like(q)
    elif variant == 'shuffled_quality':
        # Deterministic non-identity permutation. It preserves the batch's quality
        # distribution while breaking the score-to-view assignment.
        controls = torch.roll(q, shifts=1, dims=0)
    else:
        controls = q
    scales = (torch.ones_like(q) if variant == 'wo_qgm' else
              modulation(controls, cfg['tau'], cfg['rho'], cfg['kappa'], cfg['beta']))
    aligned = [scale_gradient(v, scales[i]) for i, v in enumerate(z)]
    # Optional derivative through Q still passes through the same scaled nodes.
    if cfg['quality_gradient'] and variant not in {'wo_vqe', 'uniform_quality', 'shuffled_quality'}:
        gates = torch.stack([quality_score(v, cfg['alpha'], cfg['gamma'])[0] for v in aligned])
    else:
        gates = controls
    # wo_sca = ordinary bidirectional InfoNCE, not removing all alignment.
    if variant == 'wo_sca':
        gates = torch.ones_like(gates)
    align = sca_loss(aligned, gates, cfg['temperature'],
                     cfg.get('sca_mode', 'gated_positive'))
    gradient_diagnostics = {}
    if cfg.get('log_gradient_norms', False):
        # Isolate the routing effect at each view-specific latent node.  The
        # quality controls are detached so these diagnostics measure only the
        # explicit QGM scale, not an optional derivative through the quality
        # estimator itself. RMS makes the value comparable across batch sizes.
        diagnostic_gates = gates.detach()
        diagnostic_mode = cfg.get('sca_mode', 'gated_positive')
        diagnostic_modulated = sca_loss(
            aligned, diagnostic_gates, cfg['temperature'], diagnostic_mode)
        modulated_grads = torch.autograd.grad(
            diagnostic_modulated, z, retain_graph=True, create_graph=False)
        diagnostic_raw = sca_loss(
            z, diagnostic_gates, cfg['temperature'], diagnostic_mode)
        raw_grads = torch.autograd.grad(
            diagnostic_raw, z, retain_graph=True, create_graph=False)
        gradient_diagnostics = {
            'align_grad_raw_rms': torch.stack(
                [grad.detach().square().mean().sqrt() for grad in raw_grads]),
            'align_grad_modulated_rms': torch.stack(
                [grad.detach().square().mean().sqrt() for grad in modulated_grads]),
        }
    reconstruction_mode = cfg.get('reconstruction_normalization', 'per_sample_sum')
    if reconstruction_mode == 'per_sample_sum':
        reconstruction = sum((x - xr).square().sum(dim=1).mean()
                             for x, xr in zip(views, recon))
    elif reconstruction_mode == 'per_feature_mean':
        reconstruction = sum((x - xr).square().mean() for x, xr in zip(views, recon))
    else:
        raise ValueError('unknown reconstruction_normalization')
    regularization = sum(p.square().sum() for p in model.parameters())
    total = reconstruction + cfg['lambda_sca'] * align + cfg['gamma_reg'] * regularization
    return total, {'reconstruction': reconstruction, 'sca': align,
                   'regularization': regularization, 'quality': q, 'entropy': entropy,
                   'modulation': scales,
                   'decay_branch': (controls < cfg['tau']).to(q.dtype),
                   **gradient_diagnostics}
