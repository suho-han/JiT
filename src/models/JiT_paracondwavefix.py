# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import torch
import torch.nn as nn

from src.models.JiT_paracondwave import Attention, BottleneckPatchEmbed, CrossAttention, FinalLayer, HaarSplitter, SwiGLUFFN, TimestepEmbedder, modulate
from util.model_util import RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class JiTBlockWaveFix(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, cond_weight=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_weight = cond_weight or {"cond": "fixed", "low_cond": "fixed", "high_cond": "fixed"}
        self.cond_mode = self.cond_weight.get("cond", "fixed")
        self.low_cond_mode = self.cond_weight.get("low_cond", "fixed")
        self.high_cond_mode = self.cond_weight.get("high_cond", "fixed")

        if self.cond_mode == "learnable":
            self.cond_w = nn.Parameter(torch.ones(hidden_size))
        elif self.cond_mode == "learnable_0":
            self.cond_w = nn.Parameter(torch.zeros(hidden_size))

        if self.low_cond_mode == "learnable":
            self.low_cond_w = nn.Parameter(torch.ones(hidden_size))
        elif self.low_cond_mode == "learnable_0":
            self.low_cond_w = nn.Parameter(torch.zeros(hidden_size))

        if self.high_cond_mode == "learnable":
            self.high_cond_w = nn.Parameter(torch.ones(hidden_size))
        elif self.high_cond_mode == "learnable_0":
            self.high_cond_w = nn.Parameter(torch.zeros(hidden_size))

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 15 * hidden_size, bias=True))

    def _get_weight_tensor(self, mode, local_w, shared_w, ref):
        if mode == "shared":
            weight = shared_w
        elif mode in ["learnable", "learnable_0"]:
            weight = local_w
        else:
            weight = ref.new_ones(self.hidden_size)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        return weight

    def forward(self, x, c, cond, low_cond, high_cond, feat_rope=None, shared_cond_w=None, shared_low_cond_w=None, shared_high_cond_w=None):
        shift_msa, scale_msa, gate_msa, shift_cond, scale_cond, gate_cond, \
            shift_low_cond, scale_low_cond, gate_low_cond, shift_high_cond, scale_high_cond, gate_high_cond, \
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(15, dim=-1)
            )
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_for_cross = x.clone()

        cond = modulate(self.norm2(cond), shift_cond, scale_cond)
        low_cond = modulate(self.norm2(low_cond), shift_low_cond, scale_low_cond)
        high_cond = modulate(self.norm2(high_cond), shift_high_cond, scale_high_cond)

        x = x + gate_msa.unsqueeze(1) * self.attn(x, rope=feat_rope)
        cond = cond + gate_cond.unsqueeze(1) * self.cross_attn(x_for_cross, cond, rope=feat_rope)
        low_cond = low_cond + gate_low_cond.unsqueeze(1) * self.cross_attn(x_for_cross, low_cond, rope=feat_rope)
        high_cond = high_cond + gate_high_cond.unsqueeze(1) * self.cross_attn(x_for_cross, high_cond, rope=feat_rope)

        w_c = self._get_weight_tensor(self.cond_mode, getattr(self, "cond_w", None), shared_cond_w, cond)
        w_lc = self._get_weight_tensor(self.low_cond_mode, getattr(self, "low_cond_w", None), shared_low_cond_w, cond)
        w_hc = self._get_weight_tensor(self.high_cond_mode, getattr(self, "high_cond_w", None), shared_high_cond_w, cond)

        # Normalize stream weights with softmax to prevent scale collapse among three paths.
        weights = torch.softmax(torch.stack([w_c, w_lc, w_hc], dim=0), dim=0)

        x = x + weights[0].unsqueeze(1) * cond
        x = x + weights[1].unsqueeze(1) * low_cond
        x = x + weights[2].unsqueeze(1) * high_cond

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class JiT_ParaCondWaveFix(nn.Module):
    """
    JiT ParaCondWave variant with softmax-normalized condition stream weights.
    """

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        cond_channels=0,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        cond_weight=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.cond_weight = cond_weight or {
            "cond": "fixed",
            "low_cond": "fixed",
            "high_cond": "fixed",
        }

        if self.cond_weight.get("cond") == "shared":
            self.shared_cond_w = nn.Parameter(torch.ones(hidden_size))
        if self.cond_weight.get("low_cond") == "shared":
            self.shared_low_cond_w = nn.Parameter(torch.ones(hidden_size))
        if self.cond_weight.get("high_cond") == "shared":
            self.shared_high_cond_w = nn.Parameter(torch.ones(hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.cond_embedder = BottleneckPatchEmbed(input_size, patch_size, cond_channels, bottleneck_dim, hidden_size, bias=True)
        self.low_cond_embedder = BottleneckPatchEmbed(input_size, patch_size, cond_channels, bottleneck_dim, hidden_size, bias=True)
        self.high_cond_embedder = BottleneckPatchEmbed(input_size, patch_size, cond_channels, bottleneck_dim, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len)

        self.blocks = nn.ModuleList(
            [
                JiTBlockWaveFix(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    cond_weight=self.cond_weight,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.splitter = HaarSplitter()
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, cond=None):
        t_emb = self.t_embedder(t)
        c = t_emb

        if cond is None and self.cond_channels > 0:
            cond = x.new_zeros((x.size(0), self.cond_channels, x.size(2), x.size(3)))

        x = self.x_embedder(x)
        x += self.pos_embed

        cond, low_cond, high_cond = self.splitter(cond)

        cond = self.cond_embedder(cond)
        low_cond = self.low_cond_embedder(low_cond)
        high_cond = self.high_cond_embedder(high_cond)
        cond += self.pos_embed
        low_cond += self.pos_embed
        high_cond += self.pos_embed

        for block in self.blocks:
            kwargs = {}
            if hasattr(self, "shared_cond_w"):
                kwargs["shared_cond_w"] = self.shared_cond_w
            if hasattr(self, "shared_low_cond_w"):
                kwargs["shared_low_cond_w"] = self.shared_low_cond_w
            if hasattr(self, "shared_high_cond_w"):
                kwargs["shared_high_cond_w"] = self.shared_high_cond_w

            x = block(x, c, cond, low_cond, high_cond, self.feat_rope, **kwargs)

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output
