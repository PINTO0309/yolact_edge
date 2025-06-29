#!/usr/bin/env bash
set -euo pipefail

# 個別のimagesXX.tar.gzを作成
for i in $(seq -w 00 11); do
  FOLDER="images${i}"
  ARCHIVE="${FOLDER}.tar.gz"
  echo "Archiving ${FOLDER} to ${ARCHIVE}"
  tar -czf "${ARCHIVE}" "${FOLDER}"
done

# 全てのimagesXX.tar.gzをtrain2017.tar.gzにまとめる
echo "Archiving all imagesXX.tar.gz to train2017.tar.gz"
tar -czf train2017.tar.gz images*.tar.gz

echo "Archiving complete."