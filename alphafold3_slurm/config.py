import importlib.resources as pkg_resources
import yaml


class Config:
    def __init__(self):
        self._config = {}
        self._load_config()

    def _load_config(self):
        with pkg_resources.open_text("alphafold3_slurm", "config.yaml") as f:
            self._config = yaml.safe_load(f)

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key):
        return self._config.get(key)

    def set(self, key, value):
        self._config[key] = value

    def save(self):
        with pkg_resources.open_text("alphafold3_slurm", "config.json") as f:
            yaml.dump(self._config, f)


if __name__ == "__main__":
    config = Config()
    print(config.get("key"))
    config.set("key", "value")
    config.save()
