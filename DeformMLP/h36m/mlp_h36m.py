import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

def conv_3xnxn(inp, oup, kernel_size=1, stride=1):
    return nn.Conv2d(inp, oup, kernel_size=1, padding=0, stride=1)


def conv_1xnxn(inp, oup, kernel_size=1, stride=1):
    return nn.Conv2d(inp, oup, kernel_size=1, padding=0, stride=1)

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
class DynaMixerOpx(nn.Module):
    def __init__(self, dim=22, seq_len=10, num_head=22, reduced_dim=2):#192,32,8,2
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape#32,192
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x

class DynaMixerOpy(nn.Module):
    def __init__(self, dim=110, seq_len=22, num_head=10, reduced_dim=2):#192,32,8,2
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape#32,192
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x

class DynaMixerBlock(nn.Module):
    def __init__(self, dim, resolution_t=12, resolution_j=24, num_head_t=8, num_head_j=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.resolution_t = resolution_t
        self.resolution_j = resolution_j
        self.num_head_t = num_head_t
        self.num_head_j = num_head_j
        self.mix_h = DynaMixerOpx(dim, resolution_t, self.num_head_t, reduced_dim=reduced_dim)
        self.mix_w = DynaMixerOpy(dim, resolution_j, self.num_head_j, reduced_dim=reduced_dim)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        t = self.mix_h(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        j = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (t + j + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        #x = t * a[0] + j * a[1] + c * a[2]
        x = t * a[0] + j * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MorphFC_ST(nn.Module):
    def __init__(self, dim, segment_dim_j=8, segment_dim_t=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim_j = segment_dim_j
        self.segment_dim_t = segment_dim_t
        self.mlp_t = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_j = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp = MlpBlock(dim, dim)
        self.se = SE_Block(segment_dim_t, 10)
        # init weight problem
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        S = C // self.segment_dim_j#5
        S_t = C // self.segment_dim_t#11

        t = x.reshape(B, T, J, self.segment_dim_t, S_t).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim_t, J, T * S_t)
        t = self.mlp(t).reshape(B, self.segment_dim_t, J, T, S_t).permute(0, 3, 2, 1, 4).reshape(B, T, J, C)

        j = x.reshape(B, T, J, self.segment_dim_j, S).permute(0, 1, 3, 2, 4).reshape(B, T, self.segment_dim_j, J * S)
        j = self.mlp(j).reshape(B, T, self.segment_dim_j, J, S).permute(0, 1, 3, 2, 4).reshape(B, T, J, C)

        c = self.mlp(x)

        a = (t + j + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = t * a[0] + j * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MorphFC_T(nn.Module):
    def __init__(self, dim, segment_dim_t=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        global t_stride
        self.segment_dim_t = segment_dim_t
        dim2 = dim
        self.mlp_t = MlpBlock(dim, dim)
        self.se = SE_Block(self.segment_dim_t, 10)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        S = C // self.segment_dim_t

        # T
        t = x.reshape(B, T, J, self.segment_dim_t, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim_t, J,
                                                                                   T * S)
        t = self.mlp_t(t).reshape(B, self.segment_dim_t, J, T, S).permute(0, 3, 2, 1, 4).reshape(B, T, J, C)
        x = t
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class MorphFC_S(nn.Module):
    def __init__(self, dim, segment_dim_j=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        global t_stride
        self.segment_dim_j = segment_dim_j
        dim2 = dim
        self.mlp_j = MlpBlock(dim, dim)
        self.se = SE_Block(10, 10)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        S = C // self.segment_dim_j

        # S
        j = x.reshape(B, T, J, self.segment_dim_j, S).permute(0, 1, 3, 2, 4).reshape(B, T, self.segment_dim_j, J * S)
        j = self.mlp_j(j).reshape(B, T, self.segment_dim_j, J, S).permute(0, 1, 3, 2, 4).reshape(B, T, J, C)
        x = j
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim_j, segment_dim_t, num_head_j, num_head_t, reduced_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=MorphFC_ST):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.t_norm1 = norm_layer(dim)
        self.t_fc = MorphFC_T(dim, segment_dim_t=segment_dim_t, qkv_bias=qkv_bias, qk_scale=None,
                              attn_drop=attn_drop)
        self.s_fc = MorphFC_S(dim, segment_dim_j=segment_dim_j, qkv_bias=qkv_bias, qk_scale=None,
                              attn_drop=attn_drop)
        self.fc = mlp_fn(dim, segment_dim_j=segment_dim_j, segment_dim_t=segment_dim_t, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.attn = DynaMixerBlock(dim, resolution_j=segment_dim_j, resolution_t=segment_dim_t, num_head_j=num_head_j,
                                   num_head_t=num_head_t,
                                   reduced_dim=reduced_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.se = SE_Block(segment_dim_t, 10)
        self.skip_lam = skip_lam
        init_values = 0.1
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):  # B,T,J,C

        x_t = x + self.drop_path(self.t_fc(x)) / self.skip_lam
        x_s = x_t + self.drop_path(self.s_fc(x_t)) / self.skip_lam
        x_st = x + self.drop_path(self.fc(x_s)) / self.skip_lam
        x_c = x_st + self.drop_path(self.mlp(x_st)) / self.skip_lam

        return x_c

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj1 = conv_3xnxn(in_chans, embed_dim // 2, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.act = nn.GELU()
        self.proj2 = conv_1xnxn(embed_dim // 2, embed_dim, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, 1, 1)

    def forward(self, x):
        # x = self.proj1(x)
        # x = self.norm1(x)
        # x = self.act(x)
        # x = self.proj2(x)
        # x = self.norm2(x)
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(out_embed_dim)

    def forward(self, x):#B,T,J,C
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, T, J
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

def mish(x):
    return (x*torch.tanh(F.softplus(x)))

class MlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, activation='gelu', regularization=0,
                 initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        #self.mlp_bn_dim = mlp_bn_dim
        self.regularization = 0.1
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)

        self.reg1 = nn.Dropout(0.1)
        self.reg2 = nn.Dropout(0.1)
        self.act1 = mish  # nn.Mish()


    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)

        x = self.reg1(x)
        x = self.fc2(x)

        x = self.reg2(x)

        return x

class MorphMLP(nn.Module):
    """ MorphMLP
    """

    def __init__(self, pre_len):
        super().__init__()
        in_chans = 3
        layers = [1, 1, 1, 1]
        transitions = [False, False, False, False]

        print("\nDepending on the input/output length, you may need to change some parameters at:")
        print(" ", __file__, "\n")

        segment_dim_j = [22, 22, 22, 22]
        segment_dim_t = [10, 10, 10, 10]
        mlp_dim_t = [11, 11, 11, 11]
        mlp_dim_j = [5, 5, 5, 5]
        embed_dims = [110, 110, 110, 110]

        out_dim = 25
        # out_dim = 30
        # out_dim = 90

        # segment_dim_j = [22, 22, 22, 22]
        # segment_dim_t = [50, 50, 50, 50]
        # mlp_dim_t = [11, 11, 11, 11]
        # mlp_dim_j = [25, 25, 25, 25]
        # embed_dims = [550, 550, 550, 550]

        mlp_ratios = [3, 3, 3, 3]
        reduced_dim = [2, 2, 2, 2]
        patch_size = 7
        qkv_bias = False
        qk_scale = None
        attn_drop_rate = 0.1
        drop_path_rate = 0.1
        norm_layer = Aff
        #mlp_fn = [MorphFC_S] * 3 + [MorphFC_S2]
        mlp_fn = [MorphFC_ST] * 4
        skip_lam = 1.0
        self.pre_len = pre_len
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule
        # for item in dpr:
        #     print(item)

        # stage1
        self.blocks1 = nn.ModuleList([])
        for i in range(layers[0]):
            self.blocks1.append(
                PermutatorBlock(embed_dims[0], segment_dim_j[0], segment_dim_t[0], mlp_dim_j[0], mlp_dim_t[0], reduced_dim[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop_rate, drop_path=dpr[i], skip_lam=skip_lam, mlp_fn=mlp_fn[0])
            )
        if transitions[0] or embed_dims[0] != embed_dims[1]:
            patch_size = 2 if transitions[0] else 1
            self.patch_embed2 = Downsample(embed_dims[0], embed_dims[1], patch_size)

        else:
            self.patch_embed2 = nn.Identity()
        # stage2
        self.blocks2 = nn.ModuleList([])
        for i in range(layers[1]):
            self.blocks2.append(
                PermutatorBlock(embed_dims[1], segment_dim_j[1], segment_dim_t[1], mlp_dim_j[1], mlp_dim_t[1], reduced_dim[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop_rate, drop_path=dpr[i + layers[0]], skip_lam=skip_lam,
                                mlp_fn=mlp_fn[1])
            )
        if transitions[1] or embed_dims[1] != embed_dims[2]:
            patch_size = 2 if transitions[1] else 1
            self.patch_embed3 = Downsample(embed_dims[1], embed_dims[2], patch_size)

        else:
            self.patch_embed3 = nn.Identity()
        # stage3
        self.blocks3 = nn.ModuleList([])
        for i in range(layers[2]):
            self.blocks3.append(
                PermutatorBlock(embed_dims[2], segment_dim_j[2], segment_dim_t[2], mlp_dim_j[2], mlp_dim_t[2], reduced_dim[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop_rate, drop_path=dpr[i + layers[0] + layers[1]], skip_lam=skip_lam,
                                mlp_fn=mlp_fn[2])
            )
        if transitions[2] or embed_dims[2] != embed_dims[3]:
            patch_size = 2 if transitions[2] else 1
            self.patch_embed4 = Downsample(embed_dims[2], embed_dims[3], patch_size)

        else:
            self.patch_embed4 = nn.Identity()
        # stage4
        self.blocks4 = nn.ModuleList([])
        for i in range(layers[3]):
            self.blocks4.append(
                PermutatorBlock(embed_dims[3], segment_dim_j[3], segment_dim_t[3], mlp_dim_j[3], mlp_dim_t[3], reduced_dim[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop_rate, drop_path=dpr[i + layers[0] + layers[1] + layers[2]],
                                skip_lam=skip_lam, mlp_fn=mlp_fn[3])
            )

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.fc_out = nn.Linear(embed_dims[-1], 3)
        self.conv_out = nn.Conv1d(segment_dim_t[0], out_dim, 1, stride=1)
        self.conv2d_out = nn.Conv2d(segment_dim_t[0], self.pre_len, 1, stride=1)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_fc.mlp_t.weight' in name:
                nn.init.constant_(p, 0)
            if 't_fc.mlp_t.bias' in name:
                nn.init.constant_(p, 0)
            if 't_fc.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_fc.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        # B,C,T,J -> B,T,J,C
        x = x.permute(0, 2, 3, 1)

        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)

        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)

        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)

        for blk in self.blocks4:
            x = blk(x)#B,T,J,440

        return x

    def forward(self, x):  # 224,224,T,3
        x = x.float()
        x = x.reshape(x.shape[0], x.shape[1], -1, 3)#B,T,J,C
        x = x.permute(0, 3, 1, 2)
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.fc_out(x)
        x = self.conv_out(x.reshape(x.shape[0], x.shape[1], -1))

        return x



