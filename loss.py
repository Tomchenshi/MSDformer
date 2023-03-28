import torch
import numpy as np


class HLoss(torch.nn.Module):
    def __init__(self, la1, la2, sam=True, gra=True):
        super(HLoss, self).__init__()
        self.lamd1 = la1
        self.lamd2 = la2
        self.sam = sam
        self.gra = gra

        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()

    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = self.lamd1 * cal_sam(y, gt)
        loss3 = self.lamd2 * self.gra(cal_gradient(y), cal_gradient(gt))
        loss = loss1 + loss2 + loss3
        return loss

def cal_sam(Itrue, Ifake):
  esp = 1e-6
  InnerPro = torch.sum(Itrue*Ifake,1,keepdim=True)
  len1 = torch.norm(Itrue, p=2,dim=1,keepdim=True)
  len2 = torch.norm(Ifake, p=2,dim=1,keepdim=True)
  divisor = len1*len2
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*esp
  cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
  sam = torch.acos(cosA)
  sam = torch.mean(sam) / np.pi
  return sam


def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g


def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g


def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g