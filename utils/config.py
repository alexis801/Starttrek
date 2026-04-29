##
## EPITECH PROJECT, 2026
## star-trek
## File description:
## config.py
##

# path : obj pour manipuler les chemins
# gere les differnets config
from pathlib import Path
import yaml

def load_config(path):

    # conversion path path() to use method
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # lecture (r) avec context manager
    # safe load pour la securite du parsing yaml
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # si le fichier est vide
    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    return config

def merge_configs(base, override):

    # fusione deux configs recursive.
    # Les valeurs override ecrasent base.

    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def load_merged_config(default_path, override_path):

    # charge .yaml overide specifique et fusion
    base = load_config(default_path)
    override = load_config(override_path)

    return merge_configs(base, override)