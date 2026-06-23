"""Config loading: base.yaml + per-variant override -> merged dict."""
import os

import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(_HERE, "configs")


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(variant, configs_dir=None):
    """Load base.yaml then merge configs/<variant>.yaml (e.g. variant='A4')."""
    configs_dir = configs_dir or CONFIGS_DIR
    with open(os.path.join(configs_dir, "base.yaml")) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(configs_dir, f"{variant}.yaml")) as f:
        var = yaml.safe_load(f)
    cfg = _deep_merge(base, var)
    cfg["variant"] = var.get("variant", variant)
    cfg["out_dir"] = os.path.join(cfg["paths"]["outputs"], cfg["variant"])
    return cfg


def save_snapshot(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
