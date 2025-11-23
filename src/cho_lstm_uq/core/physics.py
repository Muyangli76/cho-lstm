import torch

def carbon_closure_eps_seq(p_next_raw: torch.Tensor, p_prev_raw: torch.Tensor,
                           flows_raw: torch.Tensor, cres: torch.Tensor) -> torch.Tensor:
    """
    Residual ε for liquid-phase carbon balance (seq).
    Port your final formula here; below is a safe placeholder.
    """
    # flows_raw[..., :] = [dt, Fin_over_V_1ph, CinC_mmolC_L, CTR_mmolC_L_h]
    dt   = flows_raw[..., 0]
    FinV = flows_raw[..., 1]
    CinC = flows_raw[..., 2]
    CTR  = flows_raw[..., 3]
    d_acc = (p_next_raw - p_prev_raw).sum(dim=-1)
    return d_acc - dt * FinV * (CinC) + dt * CTR + cres  # TODO: midpoint DIC etc.
