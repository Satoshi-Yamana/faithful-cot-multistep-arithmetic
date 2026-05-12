set -eu
START_FILE_INDEX=${1-1}

find ./configs -type f -name "decode_*.jsonnet" | rev | sort | rev | tail -n +$START_FILE_INDEX | xargs -t -I % zsh ./scripts/test.sh % auto

echo "Finish all preprocessing"
