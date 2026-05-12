set -eu
START_FILE_INDEX=${1-1}

zsh ./scripts/test.sh ./configs/decode_test_2step_config.jsonnet auto
zsh ./scripts/test.sh ./configs/decode_probing_train_2step_config.jsonnet auto

echo "Finish all preprocessing"
