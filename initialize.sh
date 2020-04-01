#!/bin/bash
VERSION=0.1.0
USAGE="Usage: initialize.sh"

mkdir -p "/content/log/"
mkdir -p "/content/result/"
TARGET="/content/"

SOURCE="/content/drive/My Drive/logart/data/"
cp -r "${SOURCE}" "${TARGET}"

SOURCE="/content/drive/My Drive/logart/Embedding/"
cp -r "${SOURCE}" "${TARGET}"
