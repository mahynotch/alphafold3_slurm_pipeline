#!/bin/bash
#SBATCH --job-name=AF3_install
#SBATCH --output=AF3_install_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# Install AlphaFold3 + this pipeline into a self-contained uv virtualenv.
#
# Follows AlphaFold3's own recipe (docker/Dockerfile): Python 3.12, HMMER 3.4
# with the --seq_limit patch, `uv sync --frozen` against the release's uv.lock,
# then `build_data`. No GPU is needed. Run it either way, from the repo root:
#
#   ./install.sh [options]           # in the current shell
#   sbatch install.sh [options]      # as a CPU batch job (log: AF3_install_<id>.out)
#
# Re-running is safe: finished steps are skipped, the config is rewritten.
set -euo pipefail

AF3_VERSION="v3.0.4"
HMMER_VERSION="3.4"
HMMER_SHA256="ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3"
UV_VERSION="0.9.24"
PYTHON_VERSION="3.12"
DEFAULT_MODULES="gcc/12.2.0"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"   # sbatch runs a spooled copy of this script
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PREFIX="${REPO_DIR}/af3_env"
DB_DIR="/ibex/reference/KSL/alphafold/3.0.0"
PARAM_DIR="${REPO_DIR}/parameter"
MODULES="${DEFAULT_MODULES}"
JOBS="${SLURM_CPUS_PER_TASK:-8}"

usage() {
    cat <<EOF
Usage: $0 [options]

  --prefix DIR        Install location (default: ${PREFIX})
  --params DIR        Directory containing the AF3 weights af3.bin[.zst] (default: ${PARAM_DIR})
  --db DIR            AlphaFold3 genetic databases (default: ${DB_DIR})
  --af3-version TAG   AlphaFold3 git tag or commit (default: ${AF3_VERSION})
  --modules "LIST"    Environment modules to load before compiling, "" for none
                      (default: "${DEFAULT_MODULES}"; ignored if 'module' is unavailable)
  -h, --help          Show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --params) PARAM_DIR="$2"; shift 2 ;;
        --db) DB_DIR="$2"; shift 2 ;;
        --af3-version) AF3_VERSION="$2"; shift 2 ;;
        --modules) MODULES="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -f "${REPO_DIR}/pyproject.toml" ] && [ -d "${REPO_DIR}/alphafold3_slurm" ] \
    || die "Run this from the pipeline repository root (got ${REPO_DIR})."

mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"
VENV="${PREFIX}/venv"          # own directory: `uv sync` may recreate it
AF3_DIR="${PREFIX}/alphafold3"
HMMER_DIR="${PREFIX}/hmmer"
CONFIG_PATH="${REPO_DIR}/alphafold3_slurm/config.yaml"

# Keep uv's cache and managed Python next to the env instead of in $HOME,
# which has a small quota on most clusters.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PREFIX}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PREFIX}/.python}"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_LINK_MODE=copy

log "Installing AlphaFold3 ${AF3_VERSION} into ${PREFIX}"

# --- 1. Compilers -----------------------------------------------------------
if [ -n "${MODULES}" ] && type module >/dev/null 2>&1; then
    log "Loading modules: ${MODULES}"
    set +u   # module scripts often reference unset variables
    # shellcheck disable=SC2086
    module load ${MODULES}
    set -u
fi
command -v g++ >/dev/null || die "A C++ compiler (g++) is required to build AlphaFold3 and HMMER."
command -v make >/dev/null || die "'make' is required to build HMMER."

# --- 2. uv --------------------------------------------------------------------
if command -v uv >/dev/null; then
    UV="$(command -v uv)"
else
    UV="${PREFIX}/.uv/bin/uv"
    if [ ! -x "${UV}" ]; then
        log "Installing uv ${UV_VERSION}"
        curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
            | env UV_INSTALL_DIR="${PREFIX}/.uv/bin" INSTALLER_NO_MODIFY_PATH=1 sh
    fi
fi
log "Using $("${UV}" --version)"

# --- 3. AlphaFold3 source -----------------------------------------------------
if [ ! -d "${AF3_DIR}/.git" ]; then
    log "Cloning AlphaFold3 ${AF3_VERSION}"
    git clone --quiet https://github.com/google-deepmind/alphafold3.git "${AF3_DIR}"
fi
if ! git -C "${AF3_DIR}" rev-parse --verify --quiet "${AF3_VERSION}^{commit}" >/dev/null; then
    git -C "${AF3_DIR}" fetch --quiet --tags origin
fi
git -C "${AF3_DIR}" -c advice.detachedHead=false checkout --quiet "${AF3_VERSION}"
log "AlphaFold3 source at $(git -C "${AF3_DIR}" describe --tags --always)"

# --- 4. HMMER with AlphaFold3's --seq_limit patch -----------------------------
if [ ! -x "${HMMER_DIR}/bin/jackhmmer" ]; then
    log "Building HMMER ${HMMER_VERSION}"
    BUILD_DIR="$(mktemp -d "${PREFIX}/hmmer-build.XXXXXX")"
    (
        cd "${BUILD_DIR}"
        curl -LsSf -o "hmmer-${HMMER_VERSION}.tar.gz" \
            "http://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz"
        echo "${HMMER_SHA256}  hmmer-${HMMER_VERSION}.tar.gz" | sha256sum --check --quiet
        tar zxf "hmmer-${HMMER_VERSION}.tar.gz"
        if [ -f "${AF3_DIR}/docker/jackhmmer_seq_limit.patch" ]; then
            patch --quiet -p0 < "${AF3_DIR}/docker/jackhmmer_seq_limit.patch"
        fi
        cd "hmmer-${HMMER_VERSION}"
        ./configure --quiet --prefix="${HMMER_DIR}"
        make --quiet -j"${JOBS}"
        make --quiet install
        make --quiet -C easel install
    )
    rm -rf "${BUILD_DIR}"
fi
hmmer_banner="$("${HMMER_DIR}/bin/jackhmmer" -h)"
echo "${hmmer_banner}" | sed -n 2p

# --- 5. Python environment ----------------------------------------------------
if [ ! -x "${VENV}/bin/python" ]; then
    log "Creating Python ${PYTHON_VERSION} virtualenv"
    "${UV}" venv --quiet --python "${PYTHON_VERSION}" "${VENV}"
fi

log "Installing AlphaFold3 and its locked dependencies (this compiles C++ code)"
(
    cd "${AF3_DIR}"
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" "${UV}" sync --frozen --no-dev
)

# Installed after `uv sync`, which removes packages that are not in AF3's lockfile.
log "Installing the Slurm pipeline"
"${UV}" pip install --quiet --python "${VENV}/bin/python" -e "${REPO_DIR}"

log "Building the chemical components database"
"${VENV}/bin/build_data"

# --- 6. Pipeline config -------------------------------------------------------
if [ -f "${CONFIG_PATH}" ]; then
    cp "${CONFIG_PATH}" "${CONFIG_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
fi
cat > "${CONFIG_PATH}" <<EOF
env: ${VENV}
af3_dir: ${AF3_DIR}
hmmer_bin: ${HMMER_DIR}/bin
db: ${DB_DIR}
parameter: ${PARAM_DIR}
EOF
log "Wrote ${CONFIG_PATH}"

# --- 7. Smoke checks ----------------------------------------------------------
log "Checking the installation"
"${VENV}/bin/python" - <<'EOF'
import importlib.metadata as md
import alphafold3, jax
from alphafold3_slurm.config import get_config
print(f"alphafold3 {md.version('alphafold3')}, jax {jax.__version__}")
print(f"pipeline config: {get_config().to_dict()}")
EOF
"${VENV}/bin/python" "${AF3_DIR}/run_alphafold.py" --helpshort >/dev/null
"${VENV}/bin/af3pulldown" --help >/dev/null

warnings=0
if ! compgen -G "${PARAM_DIR}/af3.bin*" >/dev/null; then
    echo "WARNING: no af3.bin[.zst] in ${PARAM_DIR}. Request the weights from DeepMind and"
    echo "         place them there, or re-run with --params DIR."
    warnings=1
fi
if [ ! -d "${DB_DIR}" ]; then
    echo "WARNING: database directory ${DB_DIR} not found; re-run with --db DIR."
    warnings=1
fi

"${UV}" cache prune --quiet || true

if [ "${warnings}" -eq 1 ]; then
    log "Done, with warnings above."
else
    log "Done."
fi
cat <<EOF

Activate the environment before using the pipeline commands:

    source ${VENV}/bin/activate

Jobs submitted by the pipeline use absolute paths, so they do not need it.
EOF
