import torch as T
import torch.nn as nn

from torch.nn import functional as F
from einops import rearrange


def f_p_sum(A, n=2):
    A = T.pow(A, n)
    return T.sum(A, dim=1)


class DTMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_g_l, f_r_l):
        loss = 0.
        for x, y in zip(f_g_l, f_r_l):
            whitened_x = F.layer_norm(x, x.shape[1:])
            loss += F.mse_loss(whitened_x, y)
        return loss / len(f_g_l)
    

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_g_l, f_r_l):
        loss = 0.
        for x, y in zip(f_g_l, f_r_l):
            q_g_l = f_p_sum(x)
            q_r_l = f_p_sum(y)

            q_g_l = rearrange(q_g_l, "b h w -> b (h w)")
            q_r_l = rearrange(q_r_l, "b h w -> b (h w)")

            loss += F.mse_loss(q_g_l / T.norm(q_g_l), q_r_l / T.norm(q_r_l))
        return loss / len(f_g_l)
    

class DTLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.mse_loss = DTMSELoss()
        self.at_loss = ATLoss()
        self.gamma = gamma

    def forward(self, f_g_l, f_r_l):
        mse = self.mse_loss(f_g_l, f_r_l)
        at = self.at_loss(f_g_l, f_r_l)

        return mse + self.gamma * at
