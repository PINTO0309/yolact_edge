#!/usr/bin/env python3
import json
import glob
from pathlib import Path
from tqdm import tqdm

def merge_person_annotations(input_pattern, output_file):
    """
    COCOフォーマットのアノテーションファイルをマージし、
    personカテゴリのみを抽出する
    """
    # ファイルリストを取得
    files = sorted(glob.glob(input_pattern))
    print(f"Found {len(files)} files matching pattern: {input_pattern}")
    
    # 最初のファイルを読み込んでベースとする
    with open(files[0], 'r') as f:
        merged_data = json.load(f)
    
    # personカテゴリのIDを取得（通常は1）
    person_category = None
    for cat in merged_data['categories']:
        if cat['name'] == 'person':
            person_category = cat
            person_cat_id = cat['id']
            break
    
    if not person_category:
        raise ValueError("Person category not found!")
    
    print(f"Person category ID: {person_cat_id}")
    
    # categoriesをpersonのみに変更
    merged_data['categories'] = [person_category]
    
    # 最初のファイルのannotationsをpersonのみでフィルタリング
    merged_data['annotations'] = [
        ann for ann in merged_data['annotations'] 
        if ann['category_id'] == person_cat_id
    ]
    
    # 残りのファイルを処理
    for file_path in tqdm(files[1:], desc="Processing files"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # imagesを追加
        merged_data['images'].extend(data['images'])
        
        # personのannotationsのみを追加
        person_annotations = [
            ann for ann in data['annotations'] 
            if ann['category_id'] == person_cat_id
        ]
        merged_data['annotations'].extend(person_annotations)
    
    # 統計情報を出力
    print(f"\nMerged statistics:")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total person annotations: {len(merged_data['annotations'])}")
    
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
    train_output = 'annotations/instances_train2017_person_only.json'
    train_images, train_anns = merge_person_annotations(train_pattern, train_output)
    
    print("\n" + "="*50 + "\n")
    
    # 検証データの処理
    print("=== Processing validation data ===")
    val_pattern = 'annotations/sama_coco_coco_format_val_*.json'
    val_output = 'annotations/instances_val2017_person_only.json'
    val_images, val_anns = merge_person_annotations(val_pattern, val_output)
    
    # 最終統計
    print("\n" + "="*50)
    print("=== Final Summary ===")
    print(f"Training: {train_images} images, {train_anns} person annotations")
    print(f"Validation: {val_images} images, {val_anns} person annotations")

if __name__ == "__main__":
    main()