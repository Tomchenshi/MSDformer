import torch
import math
import torch.nn as nn
from common import *
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MSDformer(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_DCTM,  conv=default_conv):
        super(MSDformer, self).__init__()
        self.head = MSAMG(n_subs, n_ovls, n_colors, n_feats)
        self.body = nn.ModuleList()
        self.N = n_DCTM
        self.middle = nn.ModuleList()
        for i in range(self.N):
            self.body.append(DCTM(n_feats, 6, False))
            self.middle.append(conv(n_feats, n_feats, 3))
        self.skip_conv = conv(n_colors, n_feats, 3)
        self.upsample = Upsampler(conv, scale, n_feats)
        self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x, lms):
        x = self.head(x)
        xi = self.body[0](x)
        for i in range(1,self.N):
            xi = self.body[i](xi)
            xi = self.middle[i](xi)

        y = x + xi
        y = self.upsample(y)
        y = y + self.skip_conv(lms)
        y = self.tail(y)
        return y


class MSAMG(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_feats, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.n_feats = n_feats
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.IG = DCAM(n_subs, n_feats)
        self.spc = nn.ModuleList()
        self.middle = nn.ModuleList()
        for n in range(self.G):
            self.spc.append(ResAttentionBlock(conv, n_feats, 1, res_scale=0.1))
            self.middle.append(conv(n_feats, n_subs, 1))
        self.tail = conv(n_colors, n_feats, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.IG(xi)
            xi = self.spc[g](xi)
            xi = self.middle[g](xi)
            y[:, sta_ind:end_ind, :, :] = xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.tail(y)
        return y


class DCB(nn.Module):
    def __init__(self, n_subs, n_feats, conv=default_conv):
        super(DCB, self).__init__()
        self.dconv1 = conv(n_subs, n_feats, 3, dilation=1)
        self.dconv2 = conv(n_subs, n_feats, 3, dilation=3)
        self.dconv3 = conv(n_subs, n_feats, 3, dilation=5)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.dconv2(x))
        x3 = self.act(self.dconv3(x))
        y = x1 + x2 + x3
        return y


class DCAM(nn.Module):
    def __init__(self, n_subs, n_feats, conv=default_conv):
        super(DCAM, self).__init__()
        self.dcb = DCB(n_subs, n_feats)
        self.spa = ResBlock(conv, n_feats, 3, res_scale=0.1)

    def forward(self, x):
        y = self.dcb(x)
        y = self.spa(y)
        return y



class DMSA(nn.Module):
    """global spectral attention (DMSA)
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, bias):
        super(DMSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.deformconv(x)
        _, k, v = self.qkv(x).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class DCTM(nn.Module):
    """  Transformer Block:deformable convolution-based transformer module (DCTM)
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU, bias=False):
        super(DCTM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.num_heads = num_heads

        self.global_attn = DMSA(dim, num_heads, bias)

    def forward(self, x):

        B, C, H, W = x.shape   # B, C, H*W
        x = x.flatten(2).transpose(1, 2)   # B, H*W, C
        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C HW
        x = self.global_attn(x)  # global spectral self-attention

        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
