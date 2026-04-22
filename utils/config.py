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