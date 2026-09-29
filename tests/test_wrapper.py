import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphafold3_slurm.wrapper import Alphafold3Multimer, Alphafold3WrapperMonomer

from support import use_temp_config


class WrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_root = use_temp_config(self)

    def _write_fasta(self, path: Path, records: list[tuple[str, str]]) -> None:
        path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))

    def test_exact_mode_requires_equal_input_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.fasta"
            second = temp_path / "second.fasta"
            self._write_fasta(first, [("A1", "AAAA"), ("A2", "CCCC")])
            self._write_fasta(second, [("B1", "GGGG")])

            with self.assertRaises(ValueError):
                Alphafold3Multimer(
                    "job",
                    "both",
                    ["protein", "protein"],
                    [str(first), str(second)],
                    destination=str(temp_path / "out"),
                    exact=True,
                )

    def test_output_completion_checks_require_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Protein1", "AAAA")])
            wrapper = Alphafold3WrapperMonomer(
                "job",
                [str(fasta_path)],
                destination=str(temp_path / "out"),
            )

            model_name = "protein1"
            output_dir = wrapper.destination / model_name
            output_dir.mkdir(parents=True)
            self.assertFalse(wrapper._has_structure_output(model_name))
            self.assertFalse(wrapper._has_feature_output(model_name))

            (output_dir / f"{model_name}.cif").write_text("data")
            self.assertTrue(wrapper._has_structure_output(model_name))
            self.assertFalse(wrapper._has_feature_output(model_name))

            (output_dir / f"{model_name}_data.json").write_text("{}")
            self.assertTrue(wrapper._has_feature_output(model_name))

    def test_make_feature_inputs_skip_only_completed_feature_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Done", "AAAA"), ("Pending", "CCCC")])
            wrapper = Alphafold3Multimer(
                "job",
                "make_feature",
                ["protein"],
                [str(fasta_path)],
                destination=str(temp_path / "out"),
            )

            done_dir = wrapper.destination / "done"
            done_dir.mkdir(parents=True)
            (done_dir / "done_data.json").write_text("{}")
            pending_dir = wrapper.destination / "pending"
            pending_dir.mkdir(parents=True)

            wrapper.make_protein_features_inputs()

            generated = sorted(path.name for path in wrapper._get_inputs_dir().glob("job-*/*.json"))
            self.assertEqual(generated, ["pending.json"])

    def test_detailed_check_removes_failed_output_by_json_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Protein1", "AAAA")])
            wrapper = Alphafold3WrapperMonomer(
                "job",
                [str(fasta_path)],
                destination=str(temp_path / "out"),
            )

            failed_output_dir = wrapper.destination / "protein1"
            failed_output_dir.mkdir(parents=True)
            ibex_out_dir = wrapper.destination / "ibex_out"
            ibex_out_dir.mkdir(parents=True)
            json_path = wrapper._get_inputs_dir() / "job-0" / "protein1.json"
            log_path = ibex_out_dir / "job.out"
            log_path.write_text(
                "header\n"
                f"{json_path}\n"
                "both\n"
                "SOJ_indicator\n"
                "1\n"
                "EOJ_indicator\n"
            )

            wrapper.detailed_check()
            self.assertFalse(failed_output_dir.exists())

    def test_sbatch_submission_uses_checked_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Protein1", "AAAA")])
            wrapper = Alphafold3WrapperMonomer(
                "job",
                [str(fasta_path)],
                destination=str(temp_path / "out"),
            )
            script_path = temp_path / "run.slurm"
            script_path.write_text("#!/bin/bash\n")

            with patch("alphafold3_slurm.runtime.subprocess.run") as run_mock:
                run_mock.return_value.stdout = "Submitted batch job 123"
                run_mock.return_value.stderr = ""
                wrapper._sbatch_submit(script_path)

            run_mock.assert_called_once_with(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_print_script_uses_generated_job_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Protein1", "AAAA"), ("Protein2", "CCCC")])
            wrapper = Alphafold3WrapperMonomer(
                "job",
                [str(fasta_path)],
                destination=str(temp_path / "out"),
                max_jobs=2,
                overwrite=True,
            )
            wrapper.make_protein_monomer_inputs()

            script_path = wrapper.print_script()
            script_contents = script_path.read_text()

            self.assertIn("#SBATCH --array=0-1", script_contents)
            self.assertIn(str(wrapper.destination / "ibex_out" / "%x-%j.out"), script_contents)
            root = self.config_root
            self.assertIn(
                f"{root / 'venv' / 'bin' / 'python'} {root / 'alphafold3' / 'run_alphafold.py'}",
                script_contents,
            )
            self.assertIn(f"--jackhmmer_binary_path={root / 'hmmer' / 'bin' / 'jackhmmer'}", script_contents)
            self.assertIn(f"--nhmmer_binary_path={root / 'hmmer' / 'bin' / 'nhmmer'}", script_contents)
            self.assertIn('export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"', script_contents)
            self.assertIn("--force_output_dir", script_contents)
            self.assertIn(f"--jax_compilation_cache_dir={wrapper.destination / '.jax_cache'}", script_contents)
            self.assertNotIn("conda activate", script_contents)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", script_contents)
            self.assertNotIn("--flash_attention_implementation", script_contents)
            self.assertIn("run-both.slurm", str(script_path))
            self.assertTrue((wrapper.destination / "ibex_out").exists())

    def test_run_skips_submission_when_no_jobs_remain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / "proteins.fasta"
            self._write_fasta(fasta_path, [("Protein1", "AAAA")])
            wrapper = Alphafold3WrapperMonomer(
                "job",
                [str(fasta_path)],
                destination=str(temp_path / "out"),
            )

            completed_dir = wrapper.destination / "protein1"
            completed_dir.mkdir(parents=True)
            (completed_dir / "protein1.cif").write_text("data")
            stale_job_dir = wrapper._get_inputs_dir() / "job-0"
            stale_job_dir.mkdir(parents=True)
            (stale_job_dir / "stale.json").write_text("{}")

            with patch("alphafold3_slurm.runtime.subprocess.run") as run_mock:
                wrapper.run()

            run_mock.assert_not_called()
            self.assertEqual(list(wrapper._get_inputs_dir().glob("job-*/*.json")), [stale_job_dir / "stale.json"])


if __name__ == "__main__":
    unittest.main()
