"""Load pretrained HUWIE-Net checkpoint into HuWieNetPINN.

The HUWIE-Net backbone parameter names match the official UIE-Lab/HUWIE-Net repo
exactly (i2im_*, pim_*, con*, in*, dark_channel, atmospheric_light layers), so no
key remapping is needed.  The only weights that stay random are the PINN branch
(`pinn.*`), which has no pretrained counterpart.

We load with strict=False so the PINN keys are tolerated as missing.
"""
import torch


def load_pretrained(model, ckpt_path, verbose=True):
    """Load backbone weights; report what loaded / stayed random.

    Returns dict(loaded, missing, unexpected). Raises if the load looks wrong
    (e.g. nothing matched).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The official HUWIE-Net checkpoint is a plain state_dict.
    # Some checkpoints wrap it under 'model' or 'params' key.
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "params" in ckpt:
            state = ckpt["params"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    loaded = len(state) - len(unexpected)

    pinn_missing = [k for k in missing if k.startswith("pinn.")]
    non_pinn_missing = [k for k in missing if not k.startswith("pinn.")]

    if verbose:
        print(f"[pretrained] {loaded}/{len(state)} checkpoint tensors loaded")
        print(f"[pretrained] {len(missing)} missing in model "
              f"({len(pinn_missing)} are PINN branch, expected)")
        print(f"[pretrained] {len(unexpected)} unexpected (skipped)")
        if non_pinn_missing:
            print(f"[pretrained] WARNING non-PINN missing keys: {non_pinn_missing[:8]}"
                  f"{' ...' if len(non_pinn_missing) > 8 else ''}")
        if unexpected:
            print(f"[pretrained] WARNING unexpected keys: {unexpected[:8]}"
                  f"{' ...' if len(unexpected) > 8 else ''}")

    if loaded == 0:
        raise RuntimeError(
            "Pretrained load matched 0 tensors — checkpoint does not match "
            "HuWieNetPINN architecture."
        )
    return {"loaded": loaded, "missing": missing, "unexpected": unexpected}
