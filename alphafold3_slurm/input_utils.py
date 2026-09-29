import shutil
import string
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from Bio import SeqIO
from pydantic import BaseModel, Field, Json


MoleculeType = Literal["protein", "rna", "dna", "ligand_ccd", "ligand_smiles"]
VALID_MOLECULE_TYPES = {"protein", "rna", "dna", "ligand_ccd", "ligand_smiles"}


class PTM_Model(BaseModel):
    ptmType: str
    ptmPosition: int


class XNA_Modification_Model(BaseModel):
    modificationType: str
    basePosition: int


class Protein_Model(BaseModel):
    id: str | list[str]
    sequence: str
    modifications: Optional[list[PTM_Model]] = None
    unpairedMsa: str | None = None
    pairedMsa: str | None = None
    templates: Optional[list[Json]] = None


class Ligand_Model(BaseModel):
    id: str | list[str]
    smiles: Optional[str] = None
    ccdCodes: Optional[list[str]] = None


class XNA_Model(BaseModel):
    id: str | list[str]
    sequence: str
    modifications: Optional[list[XNA_Modification_Model]] = None


class Sequence_Model(BaseModel):
    protein: Optional[Protein_Model] = None
    ligand: Optional[Ligand_Model] = None
    rna: Optional[XNA_Model] = None
    dna: Optional[XNA_Model] = None


class Input_Model(BaseModel):
    name: str
    modelSeeds: list[int] = Field(default=[1], description="At least one seed required")
    sequences: list[Sequence_Model] = Field(
        ..., description="List of sequences, refer to sequence model"
    )
    bondedAtomPairs: Optional[list[Json]] = Field(
        default=None, description="List of bonded atom pairs"
    )
    userCCD: Optional[str] = Field(default=None, description="User provided CCD code")
    dialect: str = Field(default="alphafold3", description="Dialect")
    version: int = Field(default=1, description="Version")


def _build_sequence_model(
    chain_id: str | list[str],
    molecule_type: MoleculeType,
    sequence: str,
) -> Sequence_Model:
    if molecule_type not in VALID_MOLECULE_TYPES:
        valid = ", ".join(sorted(VALID_MOLECULE_TYPES))
        raise ValueError(f"Invalid molecule type {molecule_type!r}. Expected one of: {valid}")

    if molecule_type == "protein":
        return Sequence_Model(protein=Protein_Model(id=chain_id, sequence=sequence))
    if molecule_type == "rna":
        return Sequence_Model(rna=XNA_Model(id=chain_id, sequence=sequence))
    if molecule_type == "dna":
        return Sequence_Model(dna=XNA_Model(id=chain_id, sequence=sequence))
    if molecule_type == "ligand_smiles":
        return Sequence_Model(ligand=Ligand_Model(id=chain_id, smiles=sequence))
    return Sequence_Model(ligand=Ligand_Model(id=chain_id, ccdCodes=[sequence]))


def build_protein_dimer(name: str, sequence1: str, sequence2: str) -> str:
    sequences = [
        _build_sequence_model("A", "protein", sequence1),
        _build_sequence_model("B", "protein", sequence2),
    ]
    return Input_Model(name=name, sequences=sequences).model_dump_json(exclude_none=True)


def build_homomultimer(name: str, sequence: str, n: int) -> str:
    if n < 1:
        raise ValueError("Number of copies should be greater than 0")
    if n > 26:
        raise ValueError("Number of copies should be less than or equal to 26")

    chain_ids = [chr(65 + index) for index in range(n)]
    sequences = [_build_sequence_model(chain_ids, "protein", sequence)]
    return Input_Model(name=name, sequences=sequences).model_dump_json(exclude_none=True)


def build_dimer(
    name: str,
    A_type: MoleculeType,
    A_sequence: str,
    B_type: MoleculeType,
    B_sequence: str,
) -> str:
    sequences = [
        _build_sequence_model("A", A_type, A_sequence),
        _build_sequence_model("B", B_type, B_sequence),
    ]
    return Input_Model(name=name, sequences=sequences).model_dump_json(exclude_none=True)


def build_monomer(name: str, type: MoleculeType, sequence: str) -> str:
    sequences = [_build_sequence_model("A", type, sequence)]
    return Input_Model(name=name, sequences=sequences).model_dump_json(exclude_none=True)


def build_multimer(
    name: str,
    type: list[MoleculeType],
    sequence: list[str],
    is_feature: bool = False,
) -> str:
    if len(type) != len(sequence):
        raise ValueError("The length of type and sequence should be the same")
    if len(type) > 26:
        raise ValueError("Number of copies should be less than or equal to 26")

    models = [
        _build_sequence_model(chr(65 + index), molecule_type, molecule_sequence)
        for index, (molecule_type, molecule_sequence) in enumerate(zip(type, sequence))
    ]
    version = 2 if is_feature else 1
    return Input_Model(name=name, sequences=models, version=version).model_dump_json(
        exclude_none=True
    )


def _read_fasta_as_df(fasta_path: str) -> pl.DataFrame:
    ids: list[str] = []
    sequences: list[str] = []
    with open(fasta_path, "r") as handle:
        for seq in SeqIO.parse(handle, "fasta"):
            ids.append(seq.id.split("|")[-1])
            sequences.append(str(seq.seq))
    return pl.DataFrame({"id": ids, "sequence": sequences})


def sanitize_string(value: object) -> str:
    lower_spaceless = str(value).lower().replace(" ", "_").replace("-", "_")
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(char for char in lower_spaceless if char in allowed_chars)


def sanitize_id_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("id").map_elements(sanitize_string, return_dtype=pl.String)
    )


def read_file_as_df(file_paths: str | list[str], type_input: MoleculeType) -> pl.DataFrame:
    print(f"Reading file {file_paths}...")
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    df_list: list[pl.DataFrame] = []
    for file_path in file_paths:
        suffix = Path(file_path).suffix.lower()
        if suffix in {".fasta", ".fa"}:
            data = _read_fasta_as_df(file_path)
        elif suffix == ".csv":
            data = pl.read_csv(file_path)
        elif suffix == ".tsv":
            data = pl.read_csv(file_path, separator="\t")
        else:
            raise ValueError("Invalid file format, please use fasta, fa, csv, or tsv")
        df_list.append(data)

    concat_data = pl.concat(df_list)
    concat_data = sanitize_id_column(concat_data)
    _check_id_collisions(concat_data)
    concat_data = concat_data.with_columns(pl.lit(type_input).alias("type"))
    print(f"Final size of the table: {concat_data.shape}")
    return concat_data


def _check_id_collisions(df: pl.DataFrame) -> None:
    """Fail if different sequences share an ID after sanitization.

    Output directories are named by ID, so such rows would silently overwrite
    or skip each other.
    """
    conflicts = (
        df.group_by("id")
        .agg(pl.col("sequence").n_unique().alias("n_sequences"))
        .filter(pl.col("n_sequences") > 1)
        .sort("id")
    )
    if not conflicts.is_empty():
        shown = ", ".join(conflicts["id"].head(10).to_list())
        raise ValueError(
            f"{len(conflicts)} ID(s) map to more than one sequence after sanitization "
            f"(lowercase, spaces/dashes -> '_', other symbols removed): {shown}. "
            "Rename them so each ID is unique."
        )


def filter_dataframes(df_list: list[pl.DataFrame]) -> list[pl.DataFrame]:
    if not df_list:
        return []

    renamed_dfs = []
    for index, df in enumerate(df_list):
        renamed_dfs.append(
            df.rename(
                {
                    "id": f"id_{index}",
                    "sequence": f"sequence_{index}",
                    "type": f"type_{index}",
                }
            )
        )

    concatenated_df = pl.concat(renamed_dfs, how="horizontal").drop_nulls().unique()

    result_dfs = []
    for index in range(len(df_list)):
        cols_to_select = [f"id_{index}", f"sequence_{index}", f"type_{index}"]
        result_dfs.append(
            concatenated_df.select(cols_to_select).rename(
                {
                    f"id_{index}": "id",
                    f"sequence_{index}": "sequence",
                    f"type_{index}": "type",
                }
            )
        )
    return result_dfs


def prepare_output_dir(destination: str | Path, overwrite: bool = False) -> Path:
    destination_path = Path(destination)
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination_path} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(destination_path)
        print(f"Deleted existing directory {destination_path}")

    destination_path.mkdir(parents=True, exist_ok=True)
    return destination_path
