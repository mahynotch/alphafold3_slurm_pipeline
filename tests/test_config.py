import tempfile
import unittest
from pathlib import Path

from alphafold3_slurm.config import Config


class ConfigTest(unittest.TestCase):
    def test_save_persists_yaml_to_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("db: /db\nenv: /env\nparameter: /param\n")

            config = Config(config_path)
            config.set("db", "/new-db")
            saved_path = config.save()

            self.assertEqual(saved_path, config_path)
            reloaded = Config(config_path)
            self.assertEqual(reloaded.db, "/new-db")
            self.assertEqual(reloaded.env, "/env")
            self.assertEqual(reloaded.parameter, "/param")

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("db: /db\nenv: /env\nparameter: /param\n")
            config = Config(config_path)
            with self.assertRaises(KeyError):
                config.set("unknown", "value")

    def test_save_creates_parent_directories_for_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("db: /db\nenv: /env\nparameter: /param\n")

            config = Config(config_path)
            nested_path = Path(temp_dir) / "nested" / "dir" / "config.yaml"
            saved_path = config.save(nested_path)

            self.assertEqual(saved_path, nested_path)
            self.assertTrue(nested_path.exists())
            reloaded = Config(nested_path)
            self.assertEqual(reloaded.db, "/db")


if __name__ == "__main__":
    unittest.main()
