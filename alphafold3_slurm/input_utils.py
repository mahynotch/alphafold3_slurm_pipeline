import string
from pydantic import BaseModel, Field, Json
from typing import Optional, Literal, List
from Bio import SeqIO
import polars as pl
import os, shutil


class PTM_Model(BaseModel):
    """
    Dataclass for modifications
    e.g.:
    {
        "ptmType": "HY3",
        "ptmPosition": 1
    }
    """

    ptmType: str
    ptmPosition: int


class XNA_Modification_Model(BaseModel):
    """
    Dataclass for modifications
    e.g.:
    {
        "modificationType": "2MG",
        "basePosition": 1
    }
    """

    modificationType: str
    basePosition: int


class Protein_Model(BaseModel):
    """
    Dataclass for protein sequences
    e.g.:
    {
        "id": "A",
        "sequence": "PVLSCGEWQL",
        "modifications": [
        {"ptmType": "HY3", "ptmPosition": 1},
        {"ptmType": "P1L", "ptmPosition": 5}
        ], #Optional
        unpairedMsa: "PVLSCGEWQL", #Optional
        pairedMsa: "PVLSCGEWQL", #Optional
        "templates": [...] #Optional
    }
    """

    id: str | list[str]
    sequence: str
    modifications: Optional[list[PTM_Model]] = None
    unpairedMsa: str = None
    pairedMsa: str = None
    templates: Optional[list[Json]] = None


class Ligand_Model(BaseModel):
    """
    Dataclass for ligand sequences
    {
        "id": "B",
        "smiles": "C1=CC=C(C=C1)C(=O)O", # SMILES string or...
        "ccdCodes": ["TYR"] # CCD code

    }
    """

    id: str | list[str]
    smiles: Optional[str] = None
    ccdCodes: Optional[list[str]] = None


class XNA_Model(BaseModel):
    """
    dataclass for RNA or DNA sequences
    {
        "id": "A",
        "sequence": "AGCU",
        "modifications": [
        {"modificationType": "2MG", "basePosition": 1},
        {"modificationType": "5MC", "basePosition": 4}
        ] #Optional
    }
    """

    id: str | list[str]
    sequence: str
    modifications: Optional[list[XNA_Modification_Model]] = None


class Sequence_Model(BaseModel):
    """
    Master class of all types of sequences, please only use one of the following:
    """

    protein: Optional[Protein_Model] = None
    ligand: Optional[Ligand_Model] = None
    rna: Optional[XNA_Model] = None
    dna: Optional[XNA_Model] = None


class Input_Model(BaseModel):
    """
    Dataclass for input json file
    e.g.: {
        "name": "Job name goes here",
        "modelSeeds": [1, 2],  # At least one seed required.
        "sequences": [
            {"protein": {...}},
            {"rna": {...}},
            {"dna": {...}},
            {"ligand": {...}}
        ],
        "bondedAtomPairs": [...],  # Optional
        "userCCD": "...",  # Optional
        "dialect": "alphafold3",  # Required
        "version": 1  # Required
    }
    """

    name: str
    modelSeeds: list[int] = Field(default=[1], description="At least one seed required")
    sequences: list[Sequence_Model] = Field(
        ..., description="List of sequences, refer to sequence model"
    )
    bondedAtomPairs: Optional[list[Json]] = Field(
        default=None, description="List of bonded atom pairs"
    )
    userCCD: Optional[str] = Field(default=None, description="User provided CCD code")
    dialect: str = Field(
        default="alphafold3", description="Dialect"
    )  # Please do not change this
    version: int = Field(default=1, description="Version")


def build_protein_dimer(name: str, sequence1: str, sequence2: str) -> str:
    """
    Function to build a dimer from two sequences

    :param name: name of the output file
    :param sequence1: sequence of the first protein
    :param sequence2: sequence of the second protein
    :return: json string of the input file
    """
    sequence_protein1 = Sequence_Model(
        protein=Protein_Model(id="A", sequence=sequence1)
    )
    sequence_protein2 = Sequence_Model(
        protein=Protein_Model(id="B", sequence=sequence2)
    )
    sequences = [sequence_protein1, sequence_protein2]
    # Test the dataclass
    input_str = Input_Model(name=name, sequences=sequences)
    return input_str.model_dump_json(exclude_none=True)


def build_homomultimer(name: str, sequence: str, n: int):
    """
    Function to build a homomultimer, support up to 26 copies

    :param name: name of the output file
    :param sequence: sequence of the protein
    :param n: number of copies
    :return: json string of the input file
    """
    if n < 1:
        raise ValueError("Number of copies should be greater than 1")
    elif n > 26:
        raise ValueError("Number of copies should be less than 26")
    input_id = [chr(65 + i) for i in range(n)]
    sequences = [
        Sequence_Model(protein=Protein_Model(id=input_id, sequence=sequence))
        for i in range(n)
    ]
    # Test the dataclass
    input_str = Input_Model(name=name, sequences=sequences)
    return input_str.model_dump_json(exclude_none=True)


def build_dimer(
    name: str,
    A_type: Literal["protein", "rna", "dna", "ligand_ccd", "ligand_smiles"],
    A_sequence,
    B_type: Literal["protein", "rna", "dna", "ligand_ccd", "ligand_smiles"],
    B_sequence,
) -> str:
    """
    Function to build a dimer complex

    :param name: name of the output file
    :param A_type: type of sequence A
    :param A_sequence: sequence of the first molecule
    :param B_type: type of sequence B
    :param B_sequence: sequence of the second molecule
    """
    match A_type:
        case "protein":
            sequence_A = Sequence_Model(
                protein=Protein_Model(id="A", sequence=A_sequence)
            )
        case "rna":
            sequence_A = Sequence_Model(rna=XNA_Model(id="A", sequence=A_sequence))
        case "dna":
            sequence_A = Sequence_Model(dna=XNA_Model(id="A", sequence=A_sequence))
        case "ligand_smiles":
            sequence_A = Sequence_Model(ligand=Ligand_Model(id="A", smiles=A_sequence))
        case "ligand_ccd":
            sequence_A = Sequence_Model(
                ligand=Ligand_Model(id="A", ccdCodes=[A_sequence])
            )
        case _:
            raise ValueError(
                f"Invalid type A, should be one of protein, rna, dna, ligand_ccd, ligand_smiles, but got {A_type}"
            )
    match B_type:
        case "protein":
            sequence_B = Sequence_Model(
                protein=Protein_Model(id="B", sequence=B_sequence)
            )
        case "rna":
            sequence_B = Sequence_Model(rna=XNA_Model(id="B", sequence=B_sequence))
        case "dna":
            sequence_B = Sequence_Model(dna=XNA_Model(id="B", sequence=B_sequence))
        case "ligand_smiles":
            sequence_B = Sequence_Model(ligand=Ligand_Model(id="B", smiles=B_sequence))
        case "ligand_ccd":
            sequence_B = Sequence_Model(
                ligand=Ligand_Model(id="B", ccdCodes=[B_sequence])
            )
        case _:
            raise ValueError(
                f"Invalid type B, should be one of protein, rna, dna, ligand_ccd, ligand_smiles, but got {B_type}"
            )
    sequences = [sequence_A, sequence_B]
    input_str = Input_Model(name=name, sequences=sequences)
    return input_str.model_dump_json(exclude_none=True)


def build_monomer(
    name: str,
    type: Literal["protein", "rna", "dna", "ligand_ccd", "ligand_smiles"],
    sequence,
):
    """
    Function to build a monomer

    :param name: name of the output file
    :param type: type of the sequence
    :param sequence: sequence of the molecule
    """
    match type:
        case "protein":
            sequence = Sequence_Model(protein=Protein_Model(id="A", sequence=sequence))
        case "rna":
            sequence = Sequence_Model(rna=XNA_Model(id="A", sequence=sequence))
        case "dna":
            sequence = Sequence_Model(dna=XNA_Model(id="A", sequence=sequence))
        case "ligand_smiles":
            sequence = Sequence_Model(ligand=Ligand_Model(id="A", smiles=sequence))
        case "ligand_ccd":
            sequence = Sequence_Model(ligand=Ligand_Model(id="A", ccdCodes=[sequence]))
        case _:
            raise ValueError("Invalid type")
    sequences = [sequence]
    input_str = Input_Model(name=name, sequences=sequences)
    return input_str.model_dump_json(exclude_none=True)

def build_multimer(
    name: str,
    type: List[Literal["protein", "rna", "dna", "ligand_ccd", "ligand_smiles"]],
    sequence: List[str],
    is_feature: bool = False,
    ):
    """
    Function to build a multimer

    :param name: name of the output file
    :param type: type of the sequences
    :param sequence: sequences of the molecules
    """
    if len(type) != len(sequence):
        raise ValueError("The length of type and sequence should be the same")
    elif len(type) > 26:
        raise ValueError("Number of copies should be less than 26")
    models = []
    id = "A"
    for i in range(len(type)):
        match type[i]:
            case "protein":
                sequence_model = Sequence_Model(protein=Protein_Model(id=id, sequence=sequence[i]))
            case "rna":
                sequence_model = Sequence_Model(rna=XNA_Model(id=id, sequence=sequence[i]))
            case "dna":
                sequence_model = Sequence_Model(dna=XNA_Model(id=id, sequence=sequence[i]))
            case "ligand_smiles":
                sequence_model = Sequence_Model(ligand=Ligand_Model(id=id, smiles=sequence[i]))
            case "ligand_ccd":
                sequence_model = Sequence_Model(ligand=Ligand_Model(id=id, ccdCodes=[sequence[i]]))
            case _:
                raise ValueError("Invalid type")
        models.append(sequence_model)
        id = chr(ord(id) + 1)
    input_str = Input_Model(name=name, sequences=models, version=2 if is_feature else 1)
    return input_str.model_dump_json(exclude_none=True)



def _read_fasta_as_df(fasta_path: str) -> pl.DataFrame:
    """
    Function to read fasta file

    :param fasta_path: path to the fasta file
    :return: list of sequences
    """
    fasta_list = SeqIO.parse(fasta_path, "fasta")
    id_list = []
    sequence_list = []
    for seq in fasta_list:
        id_list.append(seq.id.split("|")[-1])
        sequence_list.append(str(seq.seq))
    data = pl.DataFrame({"id": id_list, "sequence": sequence_list})
    return data

def sanitize_string(s):
    """
    String sanitazation function, from alphafold3 codebase.
    """
    s = str(s)
    lower_spaceless = s.lower().replace(' ', '_').replace('-', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(c for c in lower_spaceless if c in allowed_chars)

def sanitize_id_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitizes the 'id' column using the same process as the sanitised_name method:
    1. Convert to lowercase
    2. Replace spaces with underscores
    3. Keep only lowercase letters, digits, underscores, hyphens, and periods
    
    Args:
        df: Input DataFrame with an 'id' column
        
    Returns:
        DataFrame with sanitized 'id' column
    """
    return df.with_columns(
        pl.col("id").map_elements(sanitize_string, return_dtype=pl.Object)
    )


def read_file_as_df(file_paths: str | List[str], type_input: str) -> pl.DataFrame:
    """
    Function to read a file

    :param file_path: path to the file
    :return: list of sequences
    """
    print(f"Reading file {file_paths}...")
    if type(file_paths) == str:
        file_paths = [file_paths]
    df_list = []
    for file_path in file_paths:
        if file_path.endswith(".fasta") or file_path.endswith(".fa"):
            data = _read_fasta_as_df(file_path)
        elif file_path.endswith(".csv"):
            data = pl.read_csv(file_path)
        elif file_path.endswith(".tsv"):
            data = pl.read_csv(file_path, sep="\t")
        else:
            raise ValueError("Invalid file format, please use files end in fasta, fa, csv or tsv")
        df_list.append(data)
    concat_data: pl.DataFrame = pl.concat(df_list)
    concat_data = sanitize_id_column(concat_data)
    concat_data = concat_data.with_columns(pl.lit(type_input).alias("type"))
    print(f"Final size of the table: {concat_data.shape}")
    return concat_data

def filter_dataframes(df_list: List[pl.DataFrame]) -> List[pl.DataFrame]:
    """
    Process a list of Polars dataframes:
    1. Horizontally concatenate them
    2. Drop null values
    3. Keep only unique rows
    4. Split them back into a list of dataframes
    
    Args:
        df_list: List of Polars dataframes, each with 'id', 'sequence', and 'type' columns
        
    Returns:
        List of Polars dataframes after processing
    """
    import polars as pl
    
    # Check if the list is empty
    if not df_list:
        return []
    
    # Rename columns to make them unique
    renamed_dfs = []
    for i, df in enumerate(df_list):
        renamed_df = df.rename({
            "id": f"id_{i}",
            "sequence": f"sequence_{i}",
            "type": f"type_{i}"
        })
        renamed_dfs.append(renamed_df)
    
    # Horizontally concatenate the dataframes
    concatenated_df = pl.concat(renamed_dfs, how="horizontal")
    
    # Drop rows with null values
    concatenated_df = concatenated_df.drop_nulls()
    
    # Keep only unique rows
    concatenated_df = concatenated_df.unique()
    
    # Split the dataframe back into a list
    result_dfs = []
    for i in range(len(df_list)):
        cols_to_select = [f"id_{i}", f"sequence_{i}", f"type_{i}"]
        result_df = concatenated_df.select(cols_to_select).rename({
            f"id_{i}": "id",
            f"sequence_{i}": "sequence",
            f"type_{i}": "type"
        })
        result_dfs.append(result_df)
    
    return result_dfs


def check_exist(destination):
    """
    Function to check if the destination folder exists, and ask the user if they want to delete it.
    :param destination: path to the destination folder
    """
    if os.path.exists(destination):
        user_input = input(
            f"The folder {destination} already exists. Do you want to delete it? (Y/n): "
        )
        if user_input.lower() == "n":
            print(f"Folder {destination} not deleted. Exiting...")
            exit()
        else:
            shutil.rmtree(destination)
            print(f"Deleted folder {destination}")
            
