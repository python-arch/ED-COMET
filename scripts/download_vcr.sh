#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-$PWD/vcr1}"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading VCR images..."
curl -L -C - -o vcr1images.zip https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip

echo "Downloading VCR annotations..."
curl -L -C - -o vcr1annots.zip https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip

echo "Unzipping..."
unzip -q vcr1images.zip
unzip -q vcr1annots.zip

echo "Done. Data in: $DEST_DIR"
