"""Test helpers: point the pipeline at a throwaway config."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphafold3_slurm.config import CONFIG_ENV_VAR, get_config


def use_temp_config(test: unittest.TestCase, *, new_layout: bool = True, force_flag: bool = True) -> Path:
    """Write a config under a temp dir, activate it for ``test``, return the root."""
    root = Path(tempfile.mkdtemp())
    test.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
    lines = [f"env: {root / 'venv'}", f"db: {root / 'db'}", f"parameter: {root / 'params'}"]
    if new_layout:
        af3_dir = root / "alphafold3"
        af3_dir.mkdir()
        flags = "flags.DEFINE_bool(\n    'force_output_dir',\n" if force_flag else ""
        (af3_dir / "run_alphafold.py").write_text(f"# fake\n{flags}")
        lines += [f"af3_dir: {af3_dir}", f"hmmer_bin: {root / 'hmmer' / 'bin'}"]
    config_path = root / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n")

    env_patch = patch.dict(os.environ, {CONFIG_ENV_VAR: str(config_path)})
    env_patch.start()
    test.addCleanup(env_patch.stop)
    get_config.cache_clear()
    test.addCleanup(get_config.cache_clear)
    return root
