import re

import torch
from sae_lens import SAE

# Thin SAELens adapter for inference-time latent edits.
# No SAE training or ranking pipeline is kept in this minimal setup.


class SAELensLatentEdit:
    def __init__(self, layer_idx, sae, feature_ids, mode="suppress", alpha=1.0):
        self.layer_idx = int(layer_idx)
        self.sae = sae
        self.feature_ids = [int(x) for x in feature_ids]
        self.mode = str(mode)
        self.alpha = float(alpha)

    def apply(self, x):
        # Edit in latent space, then decode back to model residual space.
        self.sae.to(device=x.device, dtype=x.dtype)
        z = self.sae.encode(x)
        ids = torch.tensor(self.feature_ids, device=z.device, dtype=torch.long)
        if self.mode == "suppress":
            z[:, ids] = 0.0
        elif self.mode == "amplify":
            z[:, ids] = z[:, ids] + self.alpha
        elif self.mode == "scale":
            z[:, ids] = z[:, ids] * self.alpha
        elif self.mode == "set":
            z[:, ids] = self.alpha
        else:
            raise ValueError(f"Unknown edit mode: {self.mode}")
        return self.sae.decode(z)


def parse_feature_ids(spec):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def load_pretrained_sae(sae_release, sae_id, device):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=str(sae_release),
        sae_id=str(sae_id),
        device=str(device),
    )
    sae.eval()
    return sae, cfg_dict, sparsity


def resolve_sae_layer(sae, sae_id, layer):
    if layer is not None and str(layer) != "auto":
        return int(layer)
    if hasattr(sae, "cfg") and hasattr(sae.cfg, "hook_layer"):
        return int(sae.cfg.hook_layer)
    # Fallback for releases where layer is encoded only in sae_id text.
    m = re.search(r"layer_(\d+)", str(sae_id))
    assert m is not None
    return int(m.group(1))


def make_sae_edit(
    device,
    sae_release,
    sae_id,
    feature_ids,
    mode="suppress",
    alpha=1.0,
    layer=None,
):
    sae, cfg_dict, sparsity = load_pretrained_sae(
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    layer_idx = resolve_sae_layer(sae=sae, sae_id=sae_id, layer=layer)
    edit = SAELensLatentEdit(
        layer_idx=layer_idx,
        sae=sae,
        feature_ids=feature_ids,
        mode=mode,
        alpha=alpha,
    )
    return edit, layer_idx, cfg_dict, sparsity
