#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=$1
DATA=$2
TOOLS=$3

$TOOLS/compute_image_mean $EXAMPLE/sp_train_lmdb \
  $DATA/image_mean.binaryproto

echo "Done."
