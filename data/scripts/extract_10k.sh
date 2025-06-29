#!/usr/bin/env bash
set -euo pipefail
# set -x # デバッグトレースは不要になったのでコメントアウト

# ファイルディスクリプタの制限を増やす (念のため残します)
ulimit -n 65536

ZIP=train2014.zip          # 入力アーカイブ
CHUNK=10000                # 1 フォルダあたり枚数
PREFIX=images              # 出力ディレクトリ接頭辞
TMPDIR=/mnt/ram/7z_tmp     # RAM ディスク領域
mkdir -p "$TMPDIR"

echo "[1] build list..."
LC_ALL=C 7z l -slt "$ZIP" |
  awk -F'= ' '/^Path = / && /\.jpg$/{print $2}' |
  sort > "$TMPDIR/all.txt"

total_images=$(wc -l < "$TMPDIR/all.txt")
echo "DEBUG: ZIPファイルから抽出された画像の総数: $total_images 枚"

split -l "$CHUNK" -d -a 3 "$TMPDIR/all.txt" "$TMPDIR/list_"
chunks=$(ls "$TMPDIR"/list_* | wc -l)
echo "[2] $chunks 個のチャンクに分割されました (CHUNKサイズ: $CHUNK)"

n=0
for lst in "$TMPDIR"/list_*; do
  idx=$(printf '%02d' "$n")
  ram="/mnt/ram/extract$idx"
  out="${PREFIX}${idx}"

  echo "--- チャンク $idx の処理を開始します (リストファイル: $lst) ---"
  echo "[3] chunk $idx → RAM"
  mkdir -p "$ram" "$out" # mkdir -p を元に戻す

  #### 7-Zip（スイッチ→ZIP→@listfile の順！）####
  echo "DEBUG: 7z コマンドを実行中: 7z x -bb0 -aos -o\"$ram\" \"$ZIP\" @\"$lst\""
  7z x -bb0 -aos -o"$ram" "$ZIP" @"$lst"
  code=$?
  echo "DEBUG: 7z の終了コード: $code"
  if (( code > 1 )); then
    echo "エラー: チャンク $idx で 7z エラーが発生しました。終了コード: $code"
    exit $code
  fi

  #### rsync → 実ディスク ####
  echo "DEBUG: rsync コマンドを実行中: rsync -a --info=progress2 \"$ram\"/ \"$out\"/"
  rsync -a --info=progress2 "$ram"/ "$out"/
  rsync_code=$?
  echo "DEBUG: rsync の終了コード: $rsync_code"
  case $rsync_code in
    0|23|24)
      ;;
    *)
      echo "致命的なエラー: チャンク $idx で rsync エラーが発生しました。終了コード: $rsync_code"
      exit 3
      ;;
  esac

  echo "DEBUG: rsync 後の処理に進みます。"
  rm -rf "$ram"
  echo "DEBUG: rm -rf \"$ram\" 完了。"

  # ここを修正: n=$((n+1)) を使用してnを正しくインクリメント
  echo "DEBUG: n の値 (インクリメント前): $n"
  n=$((n+1)) # 正しいインクリメント方法
  echo "DEBUG: n の値 (インクリメント後): $n"
  echo "--- チャンク $idx の処理が完了しました ---"
  echo "DEBUG: 次のチャンク処理へ移行します。"
done

echo "[DONE] $n folders (images00 … images$(printf '%02d' $((n-1))))"
