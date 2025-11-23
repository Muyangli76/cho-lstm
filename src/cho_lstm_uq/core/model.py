from typing import Tuple, Optional
import torch, torch.nn as nn
import torch.nn.functional as F

class StrictCSeq2Seq(nn.Module):
    """
    Encoder/decoder LSTM with:
      - pools head
      - c_res head
      - optional heteroscedastic X/Ab (mu/logvar)
    """
    def __init__(self, in_dim: int, out_pools: int, n_drv_aux: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 het_xab: bool = True, softplus_var: bool = False,
                 logv_min: float = -10.0, logv_max: float = 3.0):
        super().__init__()
        self.out_pools, self.n_drv_aux = out_pools, n_drv_aux
        self.het_xab, self.softplus_var = het_xab, softplus_var
        self.logv_min, self.logv_max = logv_min, logv_max

        self.enc = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec = nn.LSTM(out_pools + n_drv_aux, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)

        self.head_pools = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_pools))
        self.head_cres  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

        if het_xab:
            self.head_xab_mu = nn.Linear(hidden, 2)
            self.head_xab_lv = nn.Linear(hidden, 2)
        else:
            self.head_xab = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def _stabilize_logv(self, raw_lv):
        if self.softplus_var:
            v = F.softplus(raw_lv) + 1e-6
            return torch.log(v)
        return torch.clamp(raw_lv, self.logv_min, self.logv_max)

    def forward(self, enc_all_sc, dec_all_sc, y_tf_sc, tf_ratio: float = 1.0):
        """Minimal forward signature; port your loop logic later."""
        # TODO: port your decoding loop; for now just raise to avoid silent misuse
        raise NotImplementedError("Port forward() from your working script.")

class ScaleInputAdapter(nn.Module):
    """Discrete scale embedding → residual added to encoder inputs."""
    def __init__(self, num_scales: int, in_dim: int, bottleneck: int = 16, p_drop: float = 0.05):
        super().__init__()
        self.emb = nn.Embedding(num_scales, bottleneck)
        self.enc_proj = nn.Linear(bottleneck, in_dim, bias=False)
        self.g = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(p_drop)

    def forward(self, enc_sc, dec_sc, scale_ids):
        enc_add = self.drop(self.enc_proj(self.emb(scale_ids))).unsqueeze(1)
        return enc_sc + self.g * enc_add, dec_sc
