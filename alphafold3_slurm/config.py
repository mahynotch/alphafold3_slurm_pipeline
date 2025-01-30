from pydantic import BaseModel

class Config(BaseModel):
    """
    Config class for the alphafold3_slurm_pipeline, please modify the path according to your system.
    """
    parameter: str = "./parameter"
    # This is the path to the parameter directory, which contains the parameter files for the pipeline, you can ask for the parameter here: https://docs.google.com/forms/d/e/1FAIpQLSfWZAgo1aYk0O4MuAXZj8xRQ8DafeFJnldNOnh_13qAx2ceZw/viewform
    db: str = "/ibex/reference/KSL/alphafold/3.0.0"
    # This db dir is for Ibex system of KAUST only.