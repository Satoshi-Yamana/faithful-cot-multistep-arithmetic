set -eu
source ./scripts/setup.sh

if [ $# -ge 2 ]; then
    config_file_path=$1
    GPU_ID_ARRAY="${@:2}"
	# If CUDA_VISIBLE_DEVICES already set by environment (e.g. scheduler), keep it
	if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
		GPU_IDS=${CUDA_VISIBLE_DEVICES}
	else
		if [ "${GPU_ID_ARRAY}" = "auto" ] || [ "${GPU_ID_ARRAY}" = "all" ] || [ "${GPU_ID_ARRAY}" = "ALL" ]; then
			if command -v nvidia-smi >/dev/null 2>&1; then
				GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
			else
				GPU_IDS=$(python - <<'PY'
import torch
c = torch.cuda.device_count()
print("") if c==0 else print(",".join(str(i) for i in range(c)))
PY
)
			fi
		else
			GPU_IDS=`python -c "print(\",\".join(\"${GPU_ID_ARRAY}\".split()))"`
		fi
	fi
else
    echo "Select config file path and gpu id"
    exit
fi

JSONNET_RESULTS=$(
    jsonnet $config_file_path \
	--ext-str TAG=${TAG} \
	--ext-str ROOT=${ROOT_DIR} \
	--ext-str CURRENT_DIR=${CURRENT_DIR} \
	--ext-str PATCH_START_POSITION=${PATCH_START_POSITION} \
	--ext-str PATCH_END_POSITION=${PATCH_END_POSITION} \
	--ext-str PATCH_START_LAYER=${PATCH_START_LAYER} \
	--ext-str PATCH_END_LAYER=${PATCH_END_LAYER} \
)


echo "Config file:\n${JSONNET_RESULTS}"

TEST_ARGS=`python ./tools/config2args.py ${JSONNET_RESULTS}`
echo $TEST_ARGS

cd $SOURCE_DIR
TEST_PROGRAM=`echo "python ./test.py ${TEST_ARGS} 2>&1 | tee ${STDOUT_DIR}/test_${TAG}_${DATE}.log"`
eval "CUDA_VISIBLE_DEVICES=$GPU_IDS $TEST_PROGRAM"
cd -
