# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import torch
import torch.nn as nn

from src.models.JiT_paracond import Attention, BottleneckPatchEmbed, CrossAttention, FinalLayer, SwiGLUFFN, TimestepEmbedder, modulate
from util.model_util import RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class JiTBlockFilm(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, cond_weight=None):
        super().__init__()
        self.cond_weight = cond_weight or {"cond": "fixed"}
        self.cond_mode = self.cond_weight.get("cond", "fixed")

        if self.cond_mode == "learnable":
            self.cond_w = nn.Parameter(torch.ones(1))
        elif self.cond_mode == "zero_init":
            self.cond_w = nn.Parameter(torch.zeros(1))

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

        # FiLM modulation applied after cross-attended condition update.
        self.film_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c, cond, feat_rope=None, shared_cond_w=None):
        shift_msa, scale_msa, gate_msa, shift_cond, scale_cond, gate_cond, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_for_cross = x.clone().detach()

        cond = modulate(self.norm2(cond), shift_cond, scale_cond)

        x = x + gate_msa.unsqueeze(1) * self.attn(x, rope=feat_rope)
        cond = cond + gate_cond.unsqueeze(1) * self.cross_attn(x_for_cross, cond, rope=feat_rope)

        film_shift, film_scale = self.film_modulation(c).chunk(2, dim=-1)
        cond = modulate(cond, film_shift, film_scale)

        w_c = shared_cond_w if self.cond_mode == "shared" else (self.cond_w if self.cond_mode in ["learnable", "zero_init"] else 1.0)
        x = x + w_c * cond

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class JiT_ParaCondFiLM(nn.Module):
    """
    JiT ParaCond variant with FiLM modulation on cross-attended condition features.
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
        self.cond_weight = cond_weight or {"cond": "fixed"}

        if self.cond_weight.get("cond") == "shared":
            self.shared_cond_w = nn.Parameter(torch.ones(1))

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        self.cond_embedder = BottleneckPatchEmbed(
            input_size, patch_size, cond_channels, bottleneck_dim, hidden_size, bias=True
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        self.blocks = nn.ModuleList(
            [
                JiTBlockFilm(
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
            nn.init.constant_(block.film_modulation[-1].weight, 0)
            nn.init.constant_(block.film_modulation[-1].bias, 0)

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
        cond = self.cond_embedder(cond)
        cond += self.pos_embed

        for block in self.blocks:
            kwargs = {}
            if hasattr(self, "shared_cond_w"):
                kwargs["shared_cond_w"] = self.shared_cond_w
            x = block(x, c, cond, self.feat_rope, **kwargs)

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output
