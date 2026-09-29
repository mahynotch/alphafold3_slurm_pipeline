"""Site configuration written by ``install.sh``.

Lookup order: ``$AF3_SLURM_CONFIG`` if set, otherwise ``config.yaml`` next to
this file. Required keys are ``db``, ``env`` and ``parameter``. ``af3_dir``
(AlphaFold3 source checkout) and ``hmmer_bin`` are optional so configs written
by the old conda installer keep working.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "AF3_SLURM_CONFIG"
REQUIRED_KEYS = ("db", "env", "parameter")
OPTIONAL_KEYS = ("af3_dir", "hmmer_bin")


@dataclass
class Config:
    db: str = ""
    env: str = ""
    parameter: str = ""
    af3_dir: str | None = None
    hmmer_bin: str | None = None
    _path: Path = field(init=False, repr=False)

    def __init__(self, path: str | Path | None = None):
        config_path = Path(path) if path is not None else self.default_path()
        self._path = config_path
        self._load_config(config_path)

    @staticmethod
    def default_path() -> Path:
        override = os.environ.get(CONFIG_ENV_VAR)
        if override:
            return Path(override)
        return Path(__file__).resolve().parent / "config.yaml"

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline config not found at {path}. Run ./install.sh first, "
                f"or point ${CONFIG_ENV_VAR} at a config file "
                "(see config.example.yaml)."
            )
        data = yaml.safe_load(path.read_text()) or {}
        missing_keys = set(REQUIRED_KEYS) - data.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"Missing required config keys in {path}: {missing}")

        self.db = data["db"]
        self.env = data["env"]
        self.parameter = data["parameter"]
        self.af3_dir = data.get("af3_dir")
        self.hmmer_bin = data.get("hmmer_bin")

    @property
    def python_bin(self) -> Path:
        return Path(self.env) / "bin" / "python"

    @property
    def hmmer_bin_dir(self) -> Path:
        """Directory holding jackhmmer & co. The old installer put them in env/bin."""
        return Path(self.hmmer_bin) if self.hmmer_bin else Path(self.env) / "bin"

    @property
    def run_alphafold_script(self) -> Path:
        """AlphaFold3's run_alphafold.py (new layout) or the legacy env/bin copy."""
        if self.af3_dir:
            return Path(self.af3_dir) / "run_alphafold.py"
        return Path(self.env) / "bin" / "run_alphafold"

    def to_dict(self) -> dict[str, str]:
        data = {key: getattr(self, key) for key in REQUIRED_KEYS}
        data.update({key: getattr(self, key) for key in OPTIONAL_KEYS if getattr(self, key)})
        return data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
            raise KeyError(f"Unknown config key: {key}")
        setattr(self, key, value)

    def save(self, path: str | Path | None = None) -> Path:
        target_path = Path(path) if path is not None else self._path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        self._path = target_path
        return target_path


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load the site config once per process."""
    return Config()
