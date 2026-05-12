set -eu
source ./tools/shell_utils.sh
load_project_config

if [ -n "${WORK_DIR-}" ]; then
	export ROOT_DIR=$WORK_DIR
elif [ -w /work ]; then
	export ROOT_DIR=/work
else
	export ROOT_DIR=${BASE_DIR}/work
fi

export ROOT=$ROOT_DIR
export STDOUT_DIR=$ROOT_DIR/stdout_logs
mkdir -p $STDOUT_DIR
mkdir -p "${ROOT_DIR}/datasets"

if command -v wandb >/dev/null 2>&1; then
	wandb login
fi

# Fix the hash seed for reproducibility
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M%S`
export CURRENT_DIR=`pwd`
export TAG=`basename ${CURRENT_DIR}`


echo "Setup : ${TAG}"

### for debugging ###
date
uname -a
which python
python --version
