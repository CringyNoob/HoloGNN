"""
src/loss.py
===========
HoloGNN Custom Loss Functions — Version 4.0
---------------------------------------------
AntisymmetricLoss
    Enforces the physical antisymmetry constraint of ΔΔG predictions from the
    V4.0 Siamese forward pass.

    The mathematical formulation is:

        L = (dG_wt_to_mt + dG_mt_to_wt)² + (dG_pred − dG_exp)²
             ──────────────────────────────   ───────────────────
                  Antisymmetry Term               Fidelity Term

    Term 1 — Antisymmetry Term
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Physical law: the free-energy change from WT→MT is exactly equal and
    opposite to the change from MT→WT.  In other words:

        ΔΔG(WT→MT) = −ΔΔG(MT→WT)
        ⟹  ΔΔG(WT→MT) + ΔΔG(MT→WT) = 0

    The model should learn this for free because the SiameseStabilityHead
    computes f(z_mt − z_wt), so running it in both directions should produce
    opposite signs — but only if the learned MLP is antisymmetric at
    convergence.  The Antisymmetry Term _directly penalises_ deviations from
    this physical constraint during training, acting as a hard inductive bias
    rather than hoping the network discovers it implicitly.

    Term 2 — Fidelity Term
    ~~~~~~~~~~~~~~~~~~~~~~
    A standard MSE regression loss between the forward prediction and the
    experimentally measured ΔΔG value.  This term ensures the model does not
    simply predict zero for all inputs to trivially satisfy Term 1.

    The two terms are naturally balanced: both are squared errors, so they
    exist on the same numeric scale and no weighting coefficient is needed in
    the base implementation.  An optional `alpha` argument is provided to
    up-weight the antisymmetry term during early training if desired.
"""

import torch
import torch.nn as nn


class AntisymmetricLoss(nn.Module):
    """
    Antisymmetric ΔΔG loss for the V4.0 Siamese training loop.

    Loss = (dG_wt_to_mt + dG_mt_to_wt)² + (dG_pred − dG_exp)²
           ─────────────────────────────   ─────────────────────
               Antisymmetry constraint        Regression fidelity

    Args:
        alpha : float (default 1.0)
            Weight applied to the Antisymmetry Term relative to the
            Fidelity Term.  Keep at 1.0 for equal weighting.
            Increase (e.g., 2.0) during early training to aggressively
            enforce the physical constraint before fine-tuning fidelity.

    Inputs to forward()
    -------------------
    dG_wt_to_mt : Tensor (B, 1)
        Model prediction for the forward mutation (WT → Mutant).
    dG_mt_to_wt : Tensor (B, 1)
        Model prediction for the reverse mutation (Mutant → WT).
    dG_exp      : Tensor (B,)
        Experimental ΔΔG ground-truth values (from MegaScale / FireProtDB).

    Returns
    -------
    loss        : Scalar tensor — mean loss over the batch.
    components  : dict with keys 'antisymmetry' and 'fidelity' for logging.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def forward(
        self,
        dG_wt_to_mt: torch.Tensor,
        dG_mt_to_wt: torch.Tensor,
        dG_exp:      torch.Tensor,
    ):
        # Flatten to (B,) for element-wise operations
        pred_fwd = dG_wt_to_mt.squeeze(-1)   # (B,)
        pred_rev = dG_mt_to_wt.squeeze(-1)   # (B,)

        # ------------------------------------------------------------------
        # Term 1: Antisymmetry constraint
        #   Penalises (dG_wt_to_mt + dG_mt_to_wt)²
        #   At perfect antisymmetry this sum is zero → term vanishes.
        # ------------------------------------------------------------------
        antisymmetry_term = (pred_fwd + pred_rev) ** 2        # (B,)

        # ------------------------------------------------------------------
        # Term 2: Regression fidelity
        #   Standard MSE between the forward prediction and ground truth.
        #   We use the forward direction (WT→MT) as the primary prediction.
        # ------------------------------------------------------------------
        fidelity_term = (pred_fwd - dG_exp) ** 2             # (B,)

        # ------------------------------------------------------------------
        # Combined loss  (mean over batch)
        # ------------------------------------------------------------------
        loss = torch.mean(self.alpha * antisymmetry_term + fidelity_term)

        # Return component means for TensorBoard / tqdm logging
        components = {
            "antisymmetry": torch.mean(antisymmetry_term).item(),
            "fidelity":     torch.mean(fidelity_term).item(),
        }

        return loss, components


# =============================================================================
# [V6] Calibrated-uncertainty losses
# =============================================================================
def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic Gaussian negative log-likelihood (mean over batch).

        NLL = 0.5 * [ logσ² + (y − μ)² / σ² ]    (constant term dropped)

    Training a model to minimise this yields a *calibrated* predictive variance,
    so μ ± 1.96σ is a genuine ~95% confidence interval (rather than a heuristic
    band). ``mu``, ``logvar`` and ``target`` are broadcast to a common shape.
    """
    mu      = mu.squeeze(-1)
    logvar  = logvar.squeeze(-1)
    target  = target.squeeze(-1) if target.dim() > 1 else target
    inv_var = torch.exp(-logvar)
    return torch.mean(0.5 * (logvar + inv_var * (target - mu) ** 2))


class HeteroscedasticAntisymmetricLoss(nn.Module):
    """
    Antisymmetric ΔΔG loss with a calibrated (heteroscedastic) fidelity term.

    Loss = alpha * (μ_fwd + μ_rev)²  +  GaussianNLL(μ_fwd, logσ², ΔΔG_exp)
           └─── antisymmetry (≈0 with the StabilityScoreHead) ───┘

    Inputs to forward():
        dG_fwd_mu  : (B, 1) forward ΔΔG mean.
        dG_rev_mu  : (B, 1) reverse ΔΔG mean.
        dG_logvar  : (B, 1) predicted log-variance of the forward ΔΔG.
        dG_exp     : (B,)   experimental ΔΔG.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def forward(self, dG_fwd_mu, dG_rev_mu, dG_logvar, dG_exp):
        mu_fwd = dG_fwd_mu.squeeze(-1)
        mu_rev = dG_rev_mu.squeeze(-1)
        antisymmetry = torch.mean((mu_fwd + mu_rev) ** 2)
        nll = gaussian_nll(dG_fwd_mu, dG_logvar, dG_exp)
        loss = self.alpha * antisymmetry + nll
        components = {
            "antisymmetry": antisymmetry.item(),
            "nll":          nll.item(),
        }
        return loss, components
