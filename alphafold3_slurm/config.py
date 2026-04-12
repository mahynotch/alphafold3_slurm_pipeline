from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    db: str = ""
    env: str = ""
    parameter: str = ""
    _path: Path = field(init=False, repr=False)

    def __init__(self, path: str | Path | None = None):
        config_path = Path(path) if path is not None else self.default_path()
        self._path = config_path
        self._load_config(config_path)

    @staticmethod
    def default_path() -> Path:
        return Path(resources.files("alphafold3_slurm").joinpath("config.yaml"))

    def _load_config(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text()) or {}
        missing_keys = {"db", "env", "parameter"} - data.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"Missing required config keys: {missing}")

        self.db = data["db"]
        self.env = data["env"]
        self.parameter = data["parameter"]

    def to_dict(self) -> dict[str, str]:
        return {
            "db": self.db,
            "env": self.env,
            "parameter": self.parameter,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        if key not in self.to_dict():
            raise KeyError(f"Unknown config key: {key}")
        setattr(self, key, value)

    def save(self, path: str | Path | None = None) -> Path:
        target_path = Path(path) if path is not None else self._path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        self._path = target_path
        return target_path
