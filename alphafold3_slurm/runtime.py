"""How a Slurm job invokes AlphaFold3: environment exports and the command line.

GPU settings follow AlphaFold3 v3.0.4 (docker/Dockerfile and docs/performance.md):
Ampere+ GPUs disable Triton GEMM and keep the default Triton flash attention;
compute-capability 7.x GPUs (V100) must disable the custom-kernel-fusion pass and
use XLA attention, otherwise run_alphafold.py refuses to start.
"""

import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import Config

GPU_TYPES = ("a100", "v100")
HMMER_TOOLS = ("jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild")


@dataclass(frozen=True)
class GpuProfile:
    xla_flags: str
    flash_attention: str | None  # None keeps AlphaFold3's default (triton)


GPU_PROFILES = {
    "a100": GpuProfile("--xla_gpu_enable_triton_gemm=false", None),
    "v100": GpuProfile("--xla_disable_hlo_passes=custom-kernel-fusion-rewriter", "xla"),
}


def _gpu_profile(gpu_type: str) -> GpuProfile:
    try:
        return GPU_PROFILES[gpu_type]
    except KeyError as error:
        raise ValueError(f"Unsupported gpu_type {gpu_type!r}; expected one of {GPU_TYPES}") from error


def runtime_exports(gpu_type: str) -> str:
    """Shell ``export`` lines for a job running on ``gpu_type``."""
    profile = _gpu_profile(gpu_type)
    return "\n".join(
        [
            f'export XLA_FLAGS="{profile.xla_flags}"',
            "export XLA_PYTHON_CLIENT_PREALLOCATE=true",
            "export XLA_CLIENT_MEM_FRACTION=0.95",
        ]
    )


@lru_cache(maxsize=None)
def _script_text(script: Path) -> str:
    try:
        return script.read_text()
    except OSError:
        return ""


def supports_flag(cfg: Config, flag: str) -> bool:
    """Whether the configured run_alphafold.py defines ``flag`` (e.g. force_output_dir)."""
    return f"'{flag}'" in _script_text(cfg.run_alphafold_script)


def af3_executable(cfg: Config) -> str:
    """Shell words that start run_alphafold.py with the pipeline's interpreter."""
    if cfg.af3_dir:
        return f"{shlex.quote(str(cfg.python_bin))} {shlex.quote(str(cfg.run_alphafold_script))}"
    return shlex.quote(str(cfg.run_alphafold_script))


def build_af3_command(
    cfg: Config,
    json_path: str,
    output_dir: Path,
    *,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    gpu_type: str = "a100",
    num_diffusion_samples: int | None = None,
    num_cpu: int | None = None,
    compilation_cache_dir: Path | None = None,
) -> str:
    """Return the run_alphafold.py command line.

    ``json_path`` is inserted verbatim so callers can pass a shell variable
    such as ``"$json"``; every other path is shell-quoted here.
    """
    hmmer_dir = cfg.hmmer_bin_dir
    parts = [
        af3_executable(cfg),
        f"--json_path={json_path}",
        f"--output_dir={shlex.quote(str(output_dir))}",
        f"--model_dir={shlex.quote(str(cfg.parameter))}",
        f"--db_dir={shlex.quote(str(cfg.db))}",
    ]
    parts += [
        f"--{tool}_binary_path={shlex.quote(str(hmmer_dir / tool))}" for tool in HMMER_TOOLS
    ]
    if not run_data_pipeline:
        parts.append("--norun_data_pipeline")
    if not run_inference:
        parts.append("--norun_inference")
    if run_data_pipeline and num_cpu:
        parts += [f"--jackhmmer_n_cpu={num_cpu}", f"--nhmmer_n_cpu={num_cpu}"]
    if run_inference:
        flash_attention = _gpu_profile(gpu_type).flash_attention
        if flash_attention:
            parts.append(f"--flash_attention_implementation={flash_attention}")
        if num_diffusion_samples is not None:
            parts.append(f"--num_diffusion_samples={num_diffusion_samples}")
        if compilation_cache_dir is not None:
            parts.append(f"--jax_compilation_cache_dir={shlex.quote(str(compilation_cache_dir))}")
    if supports_flag(cfg, "force_output_dir"):
        # Without this, a re-run after a crash writes to <name>_<timestamp>/ and
        # the pipeline never sees the finished model.
        parts.append("--force_output_dir")
    return " ".join(parts)


def sbatch_submit(script_path: Path) -> str:
    """Submit ``script_path`` with sbatch and return its stdout.

    Exits with Slurm's own error message (e.g. an invalid account, QOS or
    constraint) instead of a bare CalledProcessError traceback.
    """
    try:
        completed = subprocess.run(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise SystemExit("sbatch not found: run this on a Slurm login node.") from error
    except subprocess.CalledProcessError as error:
        reason = (error.stderr or error.stdout or "").strip() or "no error message"
        raise SystemExit(
            f"sbatch rejected {script_path} (exit {error.returncode}):\n{reason}"
        ) from error
    if completed.stderr.strip():
        print(completed.stderr.strip())
    return completed.stdout.strip()
