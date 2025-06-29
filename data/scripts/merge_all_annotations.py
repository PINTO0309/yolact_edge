#!/usr/bin/env python3
import json
import glob
from pathlib import Path
from tqdm import tqdm

def merge_all_annotations(input_pattern, output_file):
    """
    COCOフォーマットのアノテーションファイルをマージする（全カテゴリ対象）
    """
    # ファイルリストを取得
    files = sorted(glob.glob(input_pattern))
    print(f"Found {len(files)} files matching pattern: {input_pattern}")
    
    # 最初のファイルを読み込んでベースとする
    with open(files[0], 'r') as f:
        merged_data = json.load(f)
    
    # カテゴリ情報を表示
    print(f"Number of categories: {len(merged_data['categories'])}")
    print("Categories:")
    for cat in merged_data['categories']:
        print(f"  - {cat['id']}: {cat['name']} ({cat['supercategory']})")
    
    # 残りのファイルを処理
    for file_path in tqdm(files[1:], desc="Processing files"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # imagesを追加
        merged_data['images'].extend(data['images'])
        
        # annotationsを追加
        merged_data['annotations'].extend(data['annotations'])
    
    # カテゴリ別の統計情報を計算
    category_counts = {}
    for cat in merged_data['categories']:
        category_counts[cat['id']] = 0
    
    for ann in merged_data['annotations']:
        cat_id = ann['category_id']
        if cat_id in category_counts:
            category_counts[cat_id] += 1
    
    # 統計情報を出力
    print(f"\nMerged statistics:")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print("\nAnnotations per category:")
    for cat in merged_data['categories']:
        cat_id = cat['id']
        count = category_counts.get(cat_id, 0)
        if count > 0:
            print(f"  - {cat['name']}: {count:,} annotations")
    
    # ファイルを保存
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"Successfully saved merged annotations to {output_file}")
    
    return len(merged_data['images']), len(merged_data['annotations'])

def main():
    # annotationsディレクトリの存在確認
    annotations_dir = Path('annotations')
    if not annotations_dir.exists():
        print("Error: annotations directory not found!")
        return
    
    # 訓練データの処理
    print("=== Processing training data ===")
    train_pattern = 'annotations/sama_coco_coco_format_train_*.json'
    train_output = 'annotations/instances_train2017.json'
    train_images, train_anns = merge_all_annotations(train_pattern, train_output)
    
    print("\n" + "="*50 + "\n")
    
    # 検証データの処理
    print("=== Processing validation data ===")
    val_pattern = 'annotations/sama_coco_coco_format_val_*.json'
    val_output = 'annotations/instances_val2017.json'
    val_images, val_anns = merge_all_annotations(val_pattern, val_output)
    
    # 最終統計
    print("\n" + "="*50)
    print("=== Final Summary ===")
    print(f"Training: {train_images:,} images, {train_anns:,} annotations")
    print(f"Validation: {val_images:,} images, {val_anns:,} annotations")

if __name__ == "__main__":
    main()