#!/usr/bin/env bash
# Create the conda environment for kinfit_nn_training.
#
#   ./setup_env.sh                  # CPU build, env in /data/dust/user/$USER/conda/envs/kinfit-nn
#   ./setup_env.sh --gpu            # CUDA build (also works on a node without a GPU)
#   ./setup_env.sh --prefix <dir>   # other location
#
# Then:  conda activate /data/dust/user/$USER/conda/envs/kinfit-nn
#
# Needs conda or mamba.  If you have neither, install Miniforge first:
#   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   bash Miniforge3-Linux-x86_64.sh -b -p /data/dust/user/$USER/miniforge3
#   source /data/dust/user/$USER/miniforge3/etc/profile.d/conda.sh
# The environment goes on dust because torch alone is several GB, too big for the AFS home quota.
set -euo pipefail

PREFIX="/data/dust/user/${USER}/conda/envs/kinfit-nn"
GPU=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU=1; shift ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v mamba >/dev/null; then CONDA=mamba
elif command -v conda >/dev/null; then CONDA=conda
else echo "conda/mamba not found, see the instructions at the top of this script"; exit 1
fi

if [[ $GPU == 1 ]]; then
    # conda-forge selects the CUDA build of pytorch if it sees a GPU driver.
    # Pretend there is one so the env can be built on a CPU-only login node.
    export CONDA_OVERRIDE_CUDA=12.6
fi

echo "Creating environment in ${PREFIX} with ${CONDA} (GPU build: ${GPU})"
"$CONDA" env create --yes --prefix "$PREFIX" --file "$HERE/environment.yml"

"$PREFIX/bin/python" -c "
import torch, uproot, awkward, sklearn
print('torch', torch.__version__, '| CUDA build:', torch.version.cuda, '| GPU visible now:', torch.cuda.is_available())
print('uproot', uproot.__version__, '| awkward', awkward.__version__)
"
echo "Done. Activate with:  conda activate $PREFIX"
