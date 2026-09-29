import json
import tempfile
import unittest
from pathlib import Path

from alphafold3_slurm.input_utils import (
    build_homomultimer,
    build_multimer,
    prepare_output_dir,
    read_file_as_df,
    sanitize_string,
)


class InputUtilsTest(unittest.TestCase):
    def test_build_homomultimer_uses_single_sequence_with_all_chain_ids(self) -> None:
        payload = json.loads(build_homomultimer("complex", "ACDE", 3))
        self.assertEqual(len(payload["sequences"]), 1)
        self.assertEqual(payload["sequences"][0]["protein"]["id"], ["A", "B", "C"])
        self.assertEqual(payload["sequences"][0]["protein"]["sequence"], "ACDE")

    def test_build_multimer_accepts_twenty_six_chains(self) -> None:
        payload = json.loads(
            build_multimer(
                "max_complex",
                ["protein"] * 26,
                [f"SEQ{index}" for index in range(26)],
            )
        )
        chain_ids = [entry["protein"]["id"] for entry in payload["sequences"]]
        self.assertEqual(chain_ids[0], "A")
        self.assertEqual(chain_ids[-1], "Z")

    def test_build_multimer_rejects_more_than_twenty_six_chains(self) -> None:
        with self.assertRaises(ValueError):
            build_multimer("too_many", ["protein"] * 27, ["A"] * 27)

    def test_prepare_output_dir_requires_overwrite_for_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "inputs_make_feature"
            target.mkdir()
            (target / "stale.json").write_text("{}");

            with self.assertRaises(FileExistsError):
                prepare_output_dir(target)

            prepare_output_dir(target, overwrite=True)
            self.assertTrue(target.exists())
            self.assertFalse((target / "stale.json").exists())

    def test_read_file_as_df_supports_csv_tsv_and_sanitizes_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "proteins.csv"
            csv_path.write_text("id,sequence\nAlpha Beta,AAAA\n")
            tsv_path = temp_path / "more.tsv"
            tsv_path.write_text("id\tsequence\nGamma-Delta\tCCCC\n")

            csv_df = read_file_as_df(str(csv_path), "protein")
            tsv_df = read_file_as_df(str(tsv_path), "protein")

            self.assertEqual(csv_df[0, "id"], "alpha_beta")
            self.assertEqual(tsv_df[0, "id"], "gamma_delta")
            self.assertEqual(csv_df[0, "type"], "protein")
            self.assertEqual(tsv_df[0, "type"], "protein")

    def test_sanitize_string_restricts_character_set(self) -> None:
        self.assertEqual(sanitize_string("A B-C*D"), "a_b_cd")



class IdCollisionTest(unittest.TestCase):
    def test_conflicting_sequences_for_same_sanitized_id_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = Path(temp_dir) / "x.fasta"
            fasta.write_text(">P1-A\nAAAA\n>p1_a\nCCCC\n")
            with self.assertRaisesRegex(ValueError, "p1_a"):
                read_file_as_df(str(fasta), "protein")

    def test_identical_duplicates_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = Path(temp_dir) / "x.fasta"
            fasta.write_text(">P1\nAAAA\n>p1\nAAAA\n")
            self.assertEqual(len(read_file_as_df(str(fasta), "protein")), 2)


if __name__ == "__main__":
    unittest.main()
