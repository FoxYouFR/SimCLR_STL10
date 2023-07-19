import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.modules.loss as loss

class InfoNCE(loss.MSELoss):
    def __init__(self, net, temp):
        super(InfoNCE, self).__init__()
        self.net = net
        self.temp = temp

    def forward(self, batch):
        return info_nce_loss(batch, self.net, self.temp)

def info_nce_loss(batch, net, temp):
    datapoints, _ = batch
    datapoints = torch.cat(datapoints, dim=0)
    # Encode all datapoints
    feats = net(datapoints)
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # Compute loss
    cos_sim = cos_sim / temp
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    return nll