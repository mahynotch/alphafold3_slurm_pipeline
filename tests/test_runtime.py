import unittest
from pathlib import Path

from alphafold3_slurm.config import get_config
from alphafold3_slurm.runtime import build_af3_command, runtime_exports

from support import use_temp_config


class RuntimeExportsTest(unittest.TestCase):
    def test_a100_disables_triton_gemm(self) -> None:
        exports = runtime_exports("a100")
        self.assertIn('export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"', exports)
        self.assertNotIn("LA_FLAGS=", exports.replace("XLA_FLAGS=", ""))

    def test_v100_disables_custom_kernel_fusion(self) -> None:
        exports = runtime_exports("v100")
        self.assertIn("--xla_disable_hlo_passes=custom-kernel-fusion-rewriter", exports)

    def test_unknown_gpu_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runtime_exports("h100x")


class BuildCommandTest(unittest.TestCase):
    def test_new_layout_uses_env_python_and_hmmer_dir(self) -> None:
        root = use_temp_config(self)
        command = build_af3_command(get_config(), '"$json"', Path("/out"), num_cpu=8)
        self.assertTrue(
            command.startswith(f"{root / 'venv/bin/python'} {root / 'alphafold3/run_alphafold.py'} ")
        )
        self.assertIn('--json_path="$json"', command)
        self.assertIn(f"--hmmbuild_binary_path={root / 'hmmer/bin/hmmbuild'}", command)
        self.assertIn("--jackhmmer_n_cpu=8", command)
        self.assertIn("--force_output_dir", command)

    def test_v100_forces_xla_attention_only_for_inference(self) -> None:
        use_temp_config(self)
        cfg = get_config()
        inference = build_af3_command(cfg, "in.json", Path("/out"), gpu_type="v100")
        features = build_af3_command(cfg, "in.json", Path("/out"), gpu_type="v100", run_inference=False)
        self.assertIn("--flash_attention_implementation=xla", inference)
        self.assertNotIn("--flash_attention_implementation", features)
        self.assertIn("--norun_inference", features)

    def test_stage_flags(self) -> None:
        use_temp_config(self)
        command = build_af3_command(
            get_config(), "in.json", Path("/out"), run_data_pipeline=False, num_diffusion_samples=3
        )
        self.assertIn("--norun_data_pipeline", command)
        self.assertIn("--num_diffusion_samples=3", command)
        self.assertNotIn("--jackhmmer_n_cpu", command)

    def test_paths_with_spaces_are_quoted(self) -> None:
        use_temp_config(self)
        command = build_af3_command(get_config(), "in.json", Path("/my out"))
        self.assertIn("--output_dir='/my out'", command)

    def test_legacy_layout_and_old_alphafold(self) -> None:
        root = use_temp_config(self, new_layout=False)
        command = build_af3_command(get_config(), "in.json", Path("/out"))
        self.assertTrue(command.startswith(f"{root / 'venv/bin/run_alphafold'} "))
        self.assertIn(f"--jackhmmer_binary_path={root / 'venv/bin/jackhmmer'}", command)
        # AF3 < 3.0.2 (no run_alphafold script found / no flag) must not get the flag.
        self.assertNotIn("--force_output_dir", command)

    def test_force_output_dir_omitted_when_unsupported(self) -> None:
        use_temp_config(self, force_flag=False)
        command = build_af3_command(get_config(), "in.json", Path("/out"))
        self.assertNotIn("--force_output_dir", command)


if __name__ == "__main__":
    unittest.main()
