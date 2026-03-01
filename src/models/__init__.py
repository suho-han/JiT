import autorootcwd

from src.models.JiT import JiT
from src.models.JiT_condimg import JiT_CondImg
from src.models.JiT_paracond import JiT_ParaCond
from src.models.JiT_paracondwave import JiT_ParaCondWave

_BASE_SPECS = {
    "B": {"depth": 12, "hidden_size": 768, "num_heads": 12, "bottleneck_dim": 128},
    "L": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "bottleneck_dim": 128},
    "H": {"depth": 32, "hidden_size": 1280, "num_heads": 16, "bottleneck_dim": 256},
}
_PATCH_SIZES = (16, 32)

JiT_models = {}


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_CondImg_B_16(**kwargs):
    return JiT_CondImg(depth=12, hidden_size=768, num_heads=12,
                       bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_CondImg_B_32(**kwargs):
    return JiT_CondImg(depth=12, hidden_size=768, num_heads=12,
                       bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_CondImg_L_16(**kwargs):
    return JiT_CondImg(depth=24, hidden_size=1024, num_heads=16,
                       bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_CondImg_L_32(**kwargs):
    return JiT_CondImg(depth=24, hidden_size=1024, num_heads=16,
                       bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_CondImg_H_16(**kwargs):
    return JiT_CondImg(depth=32, hidden_size=1280, num_heads=16,
                       bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_CondImg_H_32(**kwargs):
    return JiT_CondImg(depth=32, hidden_size=1280, num_heads=16,
                       bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCond_B_16(**kwargs):
    return JiT_ParaCond(depth=12, hidden_size=768, num_heads=12,
                        bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCond_B_32(**kwargs):
    return JiT_ParaCond(depth=12, hidden_size=768, num_heads=12,
                        bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCond_L_16(**kwargs):
    return JiT_ParaCond(depth=24, hidden_size=1024, num_heads=16,
                        bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCond_L_32(**kwargs):
    return JiT_ParaCond(depth=24, hidden_size=1024, num_heads=16,
                        bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCond_H_16(**kwargs):
    return JiT_ParaCond(depth=32, hidden_size=1280, num_heads=16,
                        bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCond_H_32(**kwargs):
    return JiT_ParaCond(depth=32, hidden_size=1280, num_heads=16,
                        bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCondWave_B_16(**kwargs):
    return JiT_ParaCondWave(depth=12, hidden_size=768, num_heads=12,
                            bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCondWave_B_32(**kwargs):
    return JiT_ParaCondWave(depth=12, hidden_size=768, num_heads=12,
                            bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCondWave_L_16(**kwargs):
    return JiT_ParaCondWave(depth=24, hidden_size=1024, num_heads=16,
                            bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCondWave_L_32(**kwargs):
    return JiT_ParaCondWave(depth=24, hidden_size=1024, num_heads=16,
                            bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


def JiT_ParaCondWave_H_16(**kwargs):
    return JiT_ParaCondWave(depth=32, hidden_size=1280, num_heads=16,
                            bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_ParaCondWave_H_32(**kwargs):
    return JiT_ParaCondWave(depth=32, hidden_size=1280, num_heads=16,
                            bottleneck_dim=256, in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
    'JiT_CondImg-B/16': JiT_CondImg_B_16,
    'JiT_CondImg-B/32': JiT_CondImg_B_32,
    'JiT_CondImg-L/16': JiT_CondImg_L_16,
    'JiT_CondImg-L/32': JiT_CondImg_L_32,
    'JiT_CondImg-H/16': JiT_CondImg_H_16,
    'JiT_CondImg-H/32': JiT_CondImg_H_32,
    'JiT_ParaCond-B/16': JiT_ParaCond_B_16,
    'JiT_ParaCond-B/32': JiT_ParaCond_B_32,
    'JiT_ParaCond-L/16': JiT_ParaCond_L_16,
    'JiT_ParaCond-L/32': JiT_ParaCond_L_32,
    'JiT_ParaCond-H/16': JiT_ParaCond_H_16,
    'JiT_ParaCond-H/32': JiT_ParaCond_H_32,
    'JiT_ParaCondWave-B/16': JiT_ParaCondWave_B_16,
    'JiT_ParaCondWave-B/32': JiT_ParaCondWave_B_32,
    'JiT_ParaCondWave-L/16': JiT_ParaCondWave_L_16,
    'JiT_ParaCondWave-L/32': JiT_ParaCondWave_L_32,
    'JiT_ParaCondWave-H/16': JiT_ParaCondWave_H_16,
    'JiT_ParaCondWave-H/32': JiT_ParaCondWave_H_32,
}
