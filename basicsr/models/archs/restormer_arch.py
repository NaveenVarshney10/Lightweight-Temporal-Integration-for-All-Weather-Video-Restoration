## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Modified with Gated Multi-Resolution SE + Temporal Skip Connections

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Squeeze-and-Excitation Block
##########################################################################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


##########################################################################
## Depthwise Separable Convolution
##########################################################################
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                    padding=kernel_size//2, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA) - Spatial Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
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


##########################################################################
## Spatial Attention TransformerBlock (Document 2 style)
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Temporal Attention Module (for skip connections)
##########################################################################
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_frames, bias, LayerNorm_type):
        super(TemporalAttention, self).__init__()
        self.num_frames = num_frames
        T = num_frames

        self.norm_t = LayerNorm(T, LayerNorm_type)
        self.attn_t = Attention(dim=T, num_heads=1, bias=bias)

    def forward(self, x):
        B_T, C, H, W = x.shape
        T = self.num_frames
        B = B_T // T

        # Rearrange for temporal processing
        temporal_input = rearrange(x, '(b t) c h w -> c (b t) h w', b=B, t=T)
        temporal_output = self.attn_t(self.norm_t(temporal_input))
        x_out = rearrange(temporal_output, 'c (b t) h w -> (b t) c h w', b=B, t=T)
        
        return x_out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer with Gated SE + Temporal Skip Connections --------
##########################################################################
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim = 48,
                 num_blocks = [4,6,6,8],
                 num_refinement_blocks = 4,
                 heads = [1,2,4,8],
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 LayerNorm_type = 'WithBias',
                 dual_pixel_task = False,
                 num_frames = 10,
                 se_reduction = 4
                ):

        super(Restormer, self).__init__()
        self.num_frames = num_frames

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # SE Blocks for each encoder level
        self.se_level1 = SEBlock(dim, reduction=se_reduction)
        self.se_level2 = SEBlock(int(dim*2**1), reduction=se_reduction)
        self.se_level3 = SEBlock(int(dim*2**2), reduction=se_reduction)
        self.se_level4 = SEBlock(int(dim*2**3), reduction=se_reduction)

        # Downsample modules for cascade path
        self.down_cascade_1_2 = Downsample(dim)
        self.down_cascade_2_3 = Downsample(int(dim*2**1))
        self.down_cascade_3_4 = Downsample(int(dim*2**2))

        # Depthwise Separable Conv for image path
        self.img_conv_L2 = DepthwiseSeparableConv(12, int(dim*2**1), kernel_size=3, bias=bias)
        self.img_conv_L3 = DepthwiseSeparableConv(48, int(dim*2**2), kernel_size=3, bias=bias)
        self.img_conv_L4 = DepthwiseSeparableConv(192, int(dim*2**3), kernel_size=3, bias=bias)

        # Encoder blocks (Spatial Attention)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # Shared Temporal Attention for skip connections
        self.temporal_skip = TemporalAttention(dim=self.num_frames, num_frames=self.num_frames, bias=bias, LayerNorm_type=LayerNorm_type)

        # Decoder
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # ========== ENCODER: Gated Multi-Resolution SE Architecture ==========
        
        # --- Level 1: Simple path ---
        original_features = self.patch_embed(inp_img)
        inp_enc_level1 = self.se_level1(original_features)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        # --- Level 2: Gated path with skip connection ---
        path_a_L2 = self.down_cascade_1_2(out_enc_level1)
        img_L2 = F.pixel_unshuffle(inp_img, 2)
        path_b_L2 = self.img_conv_L2(img_L2)
        
        gated_L2 = (path_a_L2 * path_b_L2) + path_a_L2  # Element-wise multiply + skip
        inp_enc_level2 = self.se_level2(gated_L2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        # --- Level 3: Gated path with skip connection ---
        path_a_L3 = self.down_cascade_2_3(out_enc_level2)
        img_L3 = F.pixel_unshuffle(inp_img, 4)
        path_b_L3 = self.img_conv_L3(img_L3)
        
        gated_L3 = (path_a_L3 * path_b_L3) + path_a_L3  # Element-wise multiply + skip
        inp_enc_level3 = self.se_level3(gated_L3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # --- Level 4: Gated path with skip connection ---
        path_a_L4 = self.down_cascade_3_4(out_enc_level3)
        img_L4 = F.pixel_unshuffle(inp_img, 8)
        path_b_L4 = self.img_conv_L4(img_L4)
        
        gated_L4 = (path_a_L4 * path_b_L4) + path_a_L4  # Element-wise multiply + skip
        inp_enc_level4 = self.se_level4(gated_L4)
        latent = self.latent(inp_enc_level4)
        
        # ========== DECODER: Temporal Skip Connections ==========
        
        # Apply temporal attention to encoder outputs for skip connections
        skip_enc_level3 = self.temporal_skip(out_enc_level3)
        skip_enc_level2 = self.temporal_skip(out_enc_level2)
        skip_enc_level1 = self.temporal_skip(out_enc_level1)
                            
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, skip_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, skip_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, skip_enc_level1], 1) 
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(original_features)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1