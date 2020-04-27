#!/bin/bash
VERSION=0.1.0
USAGE="Usage: copy_file.sh -t task_name -d dataset -m model_name -a alias -e epoch"

# --- Options processing -------------------------------------------
if [ $# == 0 ] ; then
    echo $USAGE
    exit 1;
fi

while getopts ":ht:d:m:a:s:e:" optname; do
    case "$optname" in
      "t")
        TASK=$OPTARG;
        ;;
      "d")
        DATASET=$OPTARG;
        ;;
      "m")
        MODEL=$OPTARG;
        ;;
      "a")
        ALIAS=$OPTARG;
        ;;
      "s")
        START_EPOCH=$OPTARG;
        ;;
      "e")
        END_EPOCH=$OPTARG;
        ;;
      "h")
        echo $USAGE
        exit 0;
        ;;
      "?")
        echo "Unknown option $OPTARG"
        exit 0;
        ;;
      ":")
        echo "No argument value for option $OPTARG"
        exit 0;
        ;;
      *)
        echo "Unknown error while processing options"
        exit 0;
        ;;
    esac
done

DIR=${TASK}/${DATASET}/${MODEL}/${ALIAS}

SOURCE="/content/models/${DIR}"
#TARGET="/content/log/logart/${DIR}/"
TARGET="/content/drive/My Drive/logart/models/${DIR}/"

mkdir -p "${TARGET}"

echo $DIR
echo $SOURCE
echo $TARGET
chmod 775 "${TARGET}"
stat "${TARGET}"

for EPOCH in $(seq $START_EPOCH $END_EPOCH); do
	cp ${SOURCE}/epoch${EPOCH}.pt "${TARGET}"
done
cp ${SOURCE}/x_encoder.p "${TARGET}"
cp ${SOURCE}/y_encoder.p "${TARGET}"


