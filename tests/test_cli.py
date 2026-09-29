import unittest

from alphafold3_slurm.cli import oligomer, pulldown
from alphafold3_slurm.cli._common import resolve_job_type
from alphafold3_slurm.cli.submit_input import _render_script
from alphafold3_slurm.config import get_config  # noqa: F401  (cache cleared by helper)

from support import use_temp_config

PULLDOWN_BASE = [
    "--bait_type", "protein", "--bait_input", "b.fa",
    "--prey_type", "protein", "--prey_input", "p.fa",
    "--destination", "out",
]


class CliTest(unittest.TestCase):
    def test_exactly_one_stage_required(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_job_type(pulldown.parsing(PULLDOWN_BASE))
        with self.assertRaises(SystemExit):
            resolve_job_type(pulldown.parsing(PULLDOWN_BASE + ["--make_feature", "--make_both"]))

    def test_feature_path_only_with_make_complex(self) -> None:
        args = pulldown.parsing(PULLDOWN_BASE + ["--make_both", "--feature_path", "f"])
        with self.assertRaises(SystemExit):
            resolve_job_type(args)
        args = pulldown.parsing(PULLDOWN_BASE + ["--make_complex", "--feature_path", "f"])
        self.assertEqual(resolve_job_type(args), "make_complex")

    def test_defaults_preserved(self) -> None:
        args = pulldown.parsing(PULLDOWN_BASE + ["--make_feature"])
        self.assertEqual((args.time, args.mem, args.max_jobs, args.gpu_type), (300, 64, 1990, "a100"))
        args = oligomer.parsing(
            ["--input_type", "protein", "--input", "a.fa", "--destination", "o", "--make_both"]
        )
        self.assertEqual(args.time, 600)

    def test_gpu_type_is_validated(self) -> None:
        with self.assertRaises(SystemExit):
            pulldown.parsing(PULLDOWN_BASE + ["--make_both", "--gpu_type", "k80"])

    def test_submit_input_script(self) -> None:
        from pathlib import Path

        use_temp_config(self)
        script = _render_script(Path("/in/x.json"), Path("/out"), 90, 32, gpu_type="v100")
        self.assertIn("#SBATCH --time=01:30:00", script)
        self.assertIn("#SBATCH --constraint=v100", script)
        self.assertIn("#SBATCH --job-name=AF3_x", script)
        self.assertIn("--json_path=/in/x.json", script)
        self.assertIn("--flash_attention_implementation=xla", script)
        self.assertNotIn("LA_FLAGS='", script)


if __name__ == "__main__":
    unittest.main()
